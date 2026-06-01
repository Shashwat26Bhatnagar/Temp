"""
UR5 Variant K cost -- Per-particle KL against the GFN's TIME-VARYING
diffusion marginals.

Same idea as cartpole Variant K: map control step k to the corresponding
GFN diffusion step and compare each particle's GP Gaussian against the
GFN's diffusion marginal at that step.

    control step k -> diffusion progress k/N_h -> GFN marginal N(mu_p(k), Sigma_p(k))

    KL_m(k) = KL( p_GFN(k) || N(mu_GP_m(k), sigma^2_GP_m(k)) )
    L = sum_k w_k * (1/M) sum_m KL_m(k) + alpha*slack_k + beta*action_k

IMPORTANT CAVEAT for UR5
------------------------
The Phase-1 UR5 GFN was trained ONLY on the terminal target (the closed
circle's endpoint). Its diffusion path therefore goes  zeros -> terminal
pose, NOT along the q_ref(t) tracking circle. So this variant is a
GOAL-REACHING curriculum with growing-then-tightening tolerance, not
pointwise circle tracking. If you want circle tracking, use Variant H/J
(which slide the target mean along q_ref(t)).

Requires MC_PILCO_GPStats (stores gp_means_seq / gp_vars_seq).
"""

import torch

from policy_learning.ur5_gfn_prior import UR5GFNPrior
from policy_learning.ur5_chance_constraint import (
    ur5_joint_total_slack,
    UR5_Q_MIN_DEFAULT,
    UR5_Q_MAX_DEFAULT,
)
from policy_learning.kl_cost import reverse_kl_gaussian_diag


class UR5_VariantK_Cost:
    """
    Per-particle KL against the GFN's time-varying diffusion marginals (UR5).

    Attributes set externally by MC_PILCO_GPStats.apply_policy:
        gp_means_seq : [N_h+1, M, 12]
        gp_vars_seq  : [N_h+1, M, 12]
    """

    def __init__(self,
                 checkpoint_path,
                 q_ref,
                 dq_ref,
                 T_control,
                 alpha=5.0,
                 epsilon=0.10,
                 weighting='uniform',
                 beta=0.0,
                 u_max=None,
                 q_min=UR5_Q_MIN_DEFAULT,
                 q_max=UR5_Q_MAX_DEFAULT,
                 num_ref_samples=512,
                 num_marginal_samples=2048,
                 dtype=torch.float64,
                 device=torch.device('cpu')):
        assert weighting in ('uniform', 'linear', 'quadratic', 'none'), \
            f"Unknown weighting '{weighting}'."

        self.alpha     = alpha
        self.beta      = beta
        self.epsilon   = epsilon
        self.weighting = weighting
        self.q_min     = q_min
        self.q_max     = q_max
        self.dtype     = dtype
        self.device    = device

        if u_max is not None:
            self.u_max = torch.as_tensor(u_max, dtype=dtype, device=device)
        else:
            self.u_max = None

        self.gfn_prior = UR5GFNPrior(
            checkpoint_path=checkpoint_path,
            q_ref=q_ref, dq_ref=dq_ref, T_control=T_control,
            num_ref_samples=num_ref_samples,
            dtype=dtype, device=device,
        )

        # ---- Extract the GFN's per-diffusion-step marginals (FROZEN) ----
        self.gfn_mu_steps, self.gfn_var_steps = \
            self.gfn_prior.get_diffusion_marginals(n_samples=num_marginal_samples)
        self.T_gfn = self.gfn_mu_steps.shape[0] - 1

        # Set by MC_PILCO_GPStats
        self.gp_means_seq = None
        self.gp_vars_seq = None

        # Logging
        self.last_kl_per_step     = None
        self.last_action_per_step = None
        self.last_slack_per_step  = None
        self.last_weights         = None
        self.last_mu_p_traj       = None
        self.last_var_p_traj      = None

        print(f"[UR5_VariantK_Cost] per-particle KL vs GFN TIME-VARYING "
              f"diffusion marginals (GOAL-REACHING curriculum)")
        print(f"[UR5_VariantK_Cost] alpha={alpha} beta={beta} epsilon={epsilon} "
              f"weighting='{weighting}'  T_gfn={self.T_gfn}")
        mid = self.T_gfn // 2
        for label, idx in [("start", 0), ("mid", mid), ("end", self.T_gfn)]:
            mu = self.gfn_mu_steps[idx][:6].tolist()
            sd = (self.gfn_var_steps[idx][:6] ** 0.5).tolist()
            print(f"[UR5_VariantK_Cost]   diff {label:>5} (step {idx:>3}) q: "
                  f"mu=[{', '.join(f'{m:+.2f}' for m in mu)}]  "
                  f"sigma=[{', '.join(f'{s:.2f}' for s in sd)}]")

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
        frac = (idx_f - idx_lo.to(dtype)).unsqueeze(-1)

        mu_lo, mu_hi = mu_steps[idx_lo], mu_steps[idx_hi]
        var_lo, var_hi = var_steps[idx_lo], var_steps[idx_hi]

        mu_p_traj = mu_lo + frac * (mu_hi - mu_lo)                # [N_h+1, 12]
        var_p_traj = var_lo + frac * (var_hi - var_lo)
        var_p_traj = torch.clamp(var_p_traj, min=1e-6)
        return mu_p_traj, var_p_traj

    # ------------------------------------------------------------------ #
    def cost_function(self, states_sequence, inputs_sequence, trial_index):
        """
        Args:
            states_sequence: [N_h+1, M, 12]
            inputs_sequence: [N_h+1, M, 6]
            trial_index:     int

        Returns:
            cost_per_step: [N_h+1]
        """
        N_h_plus_1, P, D = states_sequence.shape
        assert D == 12, f"Expected state_dim=12, got {D}"
        N_h = N_h_plus_1 - 1

        dtype  = states_sequence.dtype
        device = states_sequence.device

        # ---- Time-varying GFN target ----
        mu_p_traj, var_p_traj = self._target_at_control_steps(
            N_h_plus_1, dtype, device)

        # ---- Per-particle GP Gaussians ----
        assert self.gp_means_seq is not None, "gp_means_seq not set! Use MC_PILCO_GPStats."
        assert self.gp_vars_seq is not None, "gp_vars_seq not set! Use MC_PILCO_GPStats."
        gp_means = self.gp_means_seq.to(dtype=dtype, device=device)
        gp_vars = self.gp_vars_seq.to(dtype=dtype, device=device)
        gp_means = torch.nan_to_num(gp_means, nan=0.0, posinf=1e3, neginf=-1e3)
        gp_means = torch.clamp(gp_means, min=-50.0, max=50.0)
        gp_vars = torch.nan_to_num(gp_vars, nan=1e-6, posinf=1e6, neginf=1e-12)
        gp_vars = torch.clamp(gp_vars, min=1e-8, max=1e6)

        kl_per_step = torch.zeros(N_h_plus_1, dtype=dtype, device=device)
        for t in range(N_h_plus_1):
            kl_particle = reverse_kl_gaussian_diag(
                mu_p_traj[t], var_p_traj[t],
                gp_means[t], gp_vars[t],
            )  # [M]
            kl_particle = torch.clamp(kl_particle, min=0.0, max=1e6)
            kl_per_step[t] = kl_particle.mean()

        # ---- Time weighting ----
        t_frac = torch.arange(N_h_plus_1, dtype=dtype, device=device) / max(N_h, 1)
        if self.weighting == 'quadratic':
            weights = t_frac ** 2
        elif self.weighting == 'linear':
            weights = t_frac
        else:
            weights = torch.ones_like(t_frac)
        weighted_kl = weights * kl_per_step

        # ---- Chance constraints ----
        slack_per_step, _ = ur5_joint_total_slack(
            states_sequence, q_min=self.q_min, q_max=self.q_max,
            epsilon=self.epsilon,
        )

        # ---- Optional action regularisation ----
        if self.beta > 0 and inputs_sequence is not None:
            if self.u_max is not None:
                u_norm = inputs_sequence / self.u_max.unsqueeze(0).unsqueeze(0)
            else:
                u_norm = inputs_sequence
            action_per_step = (u_norm ** 2).sum(dim=-1).mean(dim=1)
        else:
            action_per_step = torch.zeros(N_h_plus_1, dtype=dtype, device=device)

        cost_per_step = (weighted_kl
                         + self.alpha * slack_per_step
                         + self.beta * action_per_step)

        # Logging
        self.last_kl_per_step     = kl_per_step.detach()
        self.last_action_per_step = action_per_step.detach()
        self.last_slack_per_step  = slack_per_step.detach()
        self.last_weights         = weights.detach()
        self.last_mu_p_traj       = mu_p_traj.detach()
        self.last_var_p_traj      = var_p_traj.detach()

        return cost_per_step

    def __call__(self, states_sequence, inputs_sequence, trial_index):
        cost_per_step = self.cost_function(states_sequence,
                                           inputs_sequence, trial_index)
        mean_cost = cost_per_step.sum()
        std_cost = torch.tensor(0.0, dtype=cost_per_step.dtype,
                                device=cost_per_step.device)
        return mean_cost, std_cost
