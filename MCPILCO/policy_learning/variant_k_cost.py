"""
Variant K: Per-particle KL against the GFN's TIME-VARYING diffusion
marginals (cartpole).

This is the variant that finally uses the GFN as the diffusion sampler it
actually is. Instead of comparing every control step against the GFN's
frozen TERMINAL target (Variant J), Variant K extracts the GFN's
distribution at EVERY diffusion step and maps control-time to diffusion-time:

    control step k   ->   diffusion progress u = k / N_h  in [0, 1]
                     ->   diffusion step index = u * T_gfn
                     ->   GFN marginal  N(mu_p(k), diag(sigma_p^2(k)))

Then, per particle m:

    KL_m(k) = KL( p_GFN(k) || N(mu_GP_m(k), sigma^2_GP_m(k)) )

    L = sum_k  w_k * (1/M) sum_m  KL_m(k)
      + alpha * sum_k  slack_k

Why this fixes Variant J's blow-up
-----------------------------------
Variant J compares a WIDE, uncertain GP at step 10 against the GFN's TIGHT
terminal target (sigma_theta = 0.1). The trace term sigma_p^2 / sigma_GP^2
explodes because the demanded precision is unreachable that early.

Variant K compares the step-10 GP against the GFN's step-10 diffusion
marginal -- which is ALSO wide (the GFN is only partly denoised at step 10).
Two comparably-wide Gaussians -> KL is small and well-behaved. As both the
GP and the GFN narrow toward the terminal, the KL becomes a precise,
task-relevant signal exactly when it should.

The GFN diffusion starts at zeros, matching MC-PILCO's initial state
[0,0,0,0], so the START of both distributions coincides; the END coincides
because the GFN terminal target is the upright pose. The intermediate
diffusion marginals give a natural curriculum (mean sliding 0 -> pi, variance
growing then tightening) instead of a fixed unreachable goal.

Requires MC_PILCO_GPStats
--------------------------
The per-particle GP mean/variance are captured by MC_PILCO_GPStats and
stored on this cost object as gp_means_seq / gp_vars_seq before each call.
"""

import math
import torch

from policy_learning.gfn_prior import GFNPrior
from policy_learning.chance_constraint import cartpole_total_slack
from policy_learning.kl_cost import reverse_kl_gaussian_diag


class VariantK_Cost:
    """
    Per-particle KL against the GFN's time-varying diffusion marginals.

    Attributes set externally by MC_PILCO_GPStats.apply_policy:
        gp_means_seq : [N_h+1, M, 4]
        gp_vars_seq  : [N_h+1, M, 4]
    """

    def __init__(self,
                 checkpoint_path,
                 alpha=5.0,
                 epsilon=0.10,
                 weighting='uniform',
                 position_bound=2.4,
                 angle_bound=0.35,
                 num_ref_samples=512,
                 num_marginal_samples=2048,
                 dtype=torch.float64,
                 device=torch.device('cpu')):
        assert weighting in ('quadratic', 'linear', 'none', 'uniform'), (
            f"Unknown weighting '{weighting}'.")

        self.alpha = alpha
        self.epsilon = epsilon
        self.weighting = weighting
        self.position_bound = position_bound
        self.angle_bound = angle_bound
        self.dtype = dtype
        self.device = device

        self.gfn_prior = GFNPrior(
            checkpoint_path=checkpoint_path,
            num_ref_samples=num_ref_samples,
            dtype=dtype, device=device,
        )

        # ---- Extract the GFN's per-diffusion-step marginals (FROZEN) ----
        self.gfn_mu_steps, self.gfn_var_steps = \
            self.gfn_prior.get_diffusion_marginals(n_samples=num_marginal_samples)
        self.T_gfn = self.gfn_mu_steps.shape[0] - 1

        # These are set by MC_PILCO_GPStats before each cost call
        self.gp_means_seq = None
        self.gp_vars_seq = None

        # Logging buffers
        self.last_kl_per_step = None
        self.last_slack_per_step = None
        self.last_weights = None
        self.last_mu_p_traj = None
        self.last_var_p_traj = None

        print(f"[VariantK_Cost] per-particle KL vs GFN TIME-VARYING "
              f"diffusion marginals")
        print(f"[VariantK_Cost] alpha={alpha} epsilon={epsilon} "
              f"weighting='{weighting}'  T_gfn={self.T_gfn}")
        # Sanity print: GFN marginal at start, middle, end
        mid = self.T_gfn // 2
        for label, idx in [("start", 0), ("mid", mid), ("end", self.T_gfn)]:
            mu = self.gfn_mu_steps[idx].tolist()
            sd = (self.gfn_var_steps[idx] ** 0.5).tolist()
            print(f"[VariantK_Cost]   diffusion {label:>5} (step {idx:>3}): "
                  f"mu=[{', '.join(f'{m:+.2f}' for m in mu)}]  "
                  f"sigma=[{', '.join(f'{s:.2f}' for s in sd)}]")

    # ------------------------------------------------------------------ #
    # Build time-varying target by mapping control steps to diffusion    #
    # steps (linear interpolation in diffusion-step space).              #
    # ------------------------------------------------------------------ #
    def _target_at_control_steps(self, N_h_plus_1, dtype, device):
        T_gfn = self.T_gfn
        N_h = N_h_plus_1 - 1

        mu_steps = self.gfn_mu_steps.to(dtype=dtype, device=device)
        var_steps = self.gfn_var_steps.to(dtype=dtype, device=device)

        u = torch.arange(N_h_plus_1, dtype=dtype, device=device) / max(N_h, 1)
        idx_f = u * T_gfn
        idx_lo = torch.clamp(idx_f.floor().long(), 0, T_gfn)
        idx_hi = torch.clamp(idx_lo + 1, 0, T_gfn)
        frac = (idx_f - idx_lo.to(dtype)).unsqueeze(-1)            # [N_h+1, 1]

        mu_lo, mu_hi = mu_steps[idx_lo], mu_steps[idx_hi]
        var_lo, var_hi = var_steps[idx_lo], var_steps[idx_hi]

        mu_p_traj = mu_lo + frac * (mu_hi - mu_lo)                 # [N_h+1, 4]
        var_p_traj = var_lo + frac * (var_hi - var_lo)            # [N_h+1, 4]
        var_p_traj = torch.clamp(var_p_traj, min=1e-6)
        return mu_p_traj, var_p_traj

    # ------------------------------------------------------------------ #
    def cost_function(self, states_sequence, inputs_sequence, trial_index):
        """
        Args:
            states_sequence: [N_h+1, M, 4]  (used for chance constraints)
            inputs_sequence: [N_h+1, M, 1]  (unused)
            trial_index:     int

        Returns:
            cost_per_step: [N_h+1]
        """
        N_h_plus_1, M, D = states_sequence.shape
        assert D == 4, f"Expected state_dim=4, got {D}"
        N_h = N_h_plus_1 - 1

        dtype = states_sequence.dtype
        device = states_sequence.device

        # ---- Time-varying GFN target ----
        mu_p_traj, var_p_traj = self._target_at_control_steps(
            N_h_plus_1, dtype, device)                            # [N_h+1, 4]

        # ---- Per-particle GP Gaussians ----
        assert self.gp_means_seq is not None, (
            "gp_means_seq not set! Use MC_PILCO_GPStats.")
        assert self.gp_vars_seq is not None, (
            "gp_vars_seq not set! Use MC_PILCO_GPStats.")
        gp_means = self.gp_means_seq.to(dtype=dtype, device=device)
        gp_vars = self.gp_vars_seq.to(dtype=dtype, device=device)
        gp_vars = torch.clamp(gp_vars, min=1e-8)

        # ---- Per-step, per-particle reverse KL(p_GFN(k) || GP_m(k)) ----
        kl_per_step = torch.zeros(N_h_plus_1, dtype=dtype, device=device)
        for t in range(N_h_plus_1):
            kl_particle = reverse_kl_gaussian_diag(
                mu_p_traj[t], var_p_traj[t],   # GFN marginal at this step
                gp_means[t], gp_vars[t],       # per-particle GP Gaussian
            )  # [M]
            kl_particle = torch.clamp(kl_particle, min=0.0, max=1e6)
            kl_per_step[t] = kl_particle.mean()

        # ---- Time weighting ----
        t_frac = torch.arange(N_h_plus_1, dtype=dtype, device=device) / max(N_h, 1)
        if self.weighting == 'quadratic':
            weights = t_frac ** 2
        elif self.weighting == 'linear':
            weights = t_frac
        else:  # 'uniform' / 'none'
            weights = torch.ones_like(t_frac)
        weighted_kl = weights * kl_per_step

        # ---- Chance-constraint slack ----
        slack_per_step, _ = cartpole_total_slack(
            states_sequence,
            position_bound=self.position_bound,
            angle_bound=self.angle_bound,
            epsilon=self.epsilon,
        )

        cost_per_step = weighted_kl + self.alpha * slack_per_step

        # Logging
        self.last_kl_per_step = kl_per_step.detach()
        self.last_slack_per_step = slack_per_step.detach()
        self.last_weights = weights.detach()
        self.last_mu_p_traj = mu_p_traj.detach()
        self.last_var_p_traj = var_p_traj.detach()

        return cost_per_step

    def __call__(self, states_sequence, inputs_sequence, trial_index):
        cost_per_step = self.cost_function(states_sequence,
                                           inputs_sequence,
                                           trial_index)
        mean_cost = cost_per_step.sum()
        std_cost = torch.tensor(0.0,
                                dtype=cost_per_step.dtype,
                                device=cost_per_step.device)
        return mean_cost, std_cost
