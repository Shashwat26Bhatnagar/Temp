"""
Variant K (overwritten): per-particle KL between the GFN's LIVE conditional
transition and the GP's transition, with truncated BPTT.

This is the formulation that uses the GFN as the conditional transition kernel
P_F(s'|s,t) it actually is -- the network takes the current state and the
(physical->diffusion-mapped) time and OUTPUTS a Gaussian over the next state.
No frozen marginals, no precomputed targets.

Design (per the 4 requirements)
-------------------------------
1) REVERSE KL:  KL( p_GFN(.|s) || q_GP(.|s,a) )   (GFN = p, GP = q)
2) PER PARTICLE: each particle queries the GFN from its OWN current state;
   KL computed per particle then averaged.
3) TRUNCATED BPTT (middle ground): handled by MC_PILCO_GPStats(bptt_truncate=K)
   -- detach state every K steps. Not full BPTT, not per-step (K=1). Use K~10.
4) LIVE GFN CONDITIONAL: at control step k, feed the particle's current state
   x_k and diffusion time t = k/N_h into gfn.predict_next_state, then
       mu_GFN  = x_k + dt * pf_mean(x_k, t)
       var_GFN = dt * exp(pf_logvar(x_k, t))
   (dt = 1/trajectory_length is the GFN's own diffusion step.)

Time conversion
---------------
control step k  ->  diffusion time t = k / N_h   (same normalised [0,1] axis
the GFN was trained on; matches get_mu_at_steps' u = k/N_h convention).

The transition k -> k+1:
    target   = GFN conditional from x_k                  (detached, frozen net)
    estimate = GP conditional for x_{k+1}  = gp_means_seq[k+1], gp_vars_seq[k+1]
    KL(target || estimate)   -> reverse KL, GP variance in the denominator.

NOTE (denominator): reverse KL divides by the GP variance, which is tiny for
positions in the speed model. The per-particle KL is clamped and the truncated
BPTT stops any blow-up from propagating across the whole trajectory; if it
still misbehaves, switch kl_direction='forward'.

Requires MC_PILCO_GPStats (sets gp_means_seq / gp_vars_seq).
"""

import math
import torch

from policy_learning.gfn_prior import GFNPrior
from policy_learning.chance_constraint import cartpole_total_slack
from policy_learning.kl_cost import (
    reverse_kl_gaussian_diag,
    forward_kl_gaussian_diag,
)


class VariantK_Cost:
    """
    Per-particle KL vs the GFN's LIVE conditional transition (cartpole).

    Attributes set externally by MC_PILCO_GPStats.apply_policy:
        gp_means_seq : [N_h+1, M, 4]   GP next-state mean per particle
        gp_vars_seq  : [N_h+1, M, 4]   GP next-state variance per particle
    """

    def __init__(self,
                 checkpoint_path,
                 alpha=5.0,
                 epsilon=0.10,
                 weighting='uniform',
                 kl_direction='reverse',
                 use_state_weight=True,
                 position_bound=2.4,
                 angle_bound=0.35,
                 num_ref_samples=512,
                 dtype=torch.float64,
                 device=torch.device('cpu')):
        assert weighting in ('quadratic', 'linear', 'none', 'uniform'), (
            f"Unknown weighting '{weighting}'.")
        assert kl_direction in ('forward', 'reverse'), (
            f"Unknown kl_direction '{kl_direction}'.")

        self.alpha = alpha
        self.epsilon = epsilon
        self.weighting = weighting
        self.kl_direction = kl_direction
        self.use_state_weight = use_state_weight
        self.position_bound = position_bound
        self.angle_bound = angle_bound
        self.dtype = dtype
        self.device = device

        self.gfn_prior = GFNPrior(
            checkpoint_path=checkpoint_path,
            num_ref_samples=num_ref_samples,
            dtype=dtype, device=device,
        )
        # The LIVE network -- queried as a conditional transition kernel.
        self.gfn = self.gfn_prior.gfn_model
        self.gfn_dt = float(self.gfn.dt)
        assert not self.gfn.langevin, (
            "This variant assumes langevin=False so predict_next_state does "
            "not need the energy gradient.")

        # Set by MC_PILCO_GPStats
        self.gp_means_seq = None
        self.gp_vars_seq = None

        # Logging
        self.last_kl_per_step = None
        self.last_slack_per_step = None
        self.last_weights = None

        print(f"[VariantK_Cost] LIVE conditional GFN transition  "
              f"(state,t) -> N(next state)")
        print(f"[VariantK_Cost] kl_direction = {kl_direction.upper()}  "
              f"({'KL(GFN||GP), GP var in denom' if kl_direction=='reverse' else 'KL(GP||GFN), GFN var in denom'})")
        print(f"[VariantK_Cost] alpha={alpha} epsilon={epsilon} "
              f"weighting='{weighting}'  gfn_dt={self.gfn_dt:.4f}")
        print(f"[VariantK_Cost] state-probability weighting = "
              f"{'ON (w_m = softmax over particles of gp-density of s_k)' if use_state_weight else 'OFF (plain particle average)'}")
        # Sanity: query the GFN from zeros at t=0 and t=0.5
        self._sanity_print()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _diag_gaussian_logprob(x, mu, var):
        """log N(x | mu, diag(var)) per particle. x,mu,var: [M,D] -> [M]."""
        var = torch.clamp(var, min=1e-12)
        return -0.5 * (((x - mu) ** 2 / var)
                       + torch.log(2.0 * math.pi * var)).sum(dim=-1)

    def _dummy_logr(self, x, condition=None):
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

    def _gfn_predict(self, state_k, t_scalar):
        """
        Query the GFN conditional transition from `state_k` at diffusion
        time `t_scalar`. Returns (mu_GFN, var_GFN), both detached, in the
        dtype/device of `state_k`.

            mu_GFN  = s + dt * pf_mean(s, t)
            var_GFN = dt * exp(pf_logvar(s, t))
        """
        dtype, device = state_k.dtype, state_k.device
        with torch.no_grad():
            s32 = state_k.detach().to(dtype=torch.float32, device=device)
            pfs, _ = self.gfn.predict_next_state(s32, float(t_scalar),
                                                 self._dummy_logr)
            pf_mean, pf_logvar = self.gfn.split_params(pfs)
            mu_gfn = s32 + self.gfn_dt * pf_mean
            var_gfn = self.gfn_dt * torch.exp(pf_logvar)
        return (mu_gfn.to(dtype=dtype, device=device),
                var_gfn.to(dtype=dtype, device=device))

    def _sanity_print(self):
        z = torch.zeros(1, 4, dtype=self.dtype, device=self.device)
        for t in (0.0, 0.5, 0.95):
            mu, var = self._gfn_predict(z, t)
            print(f"[VariantK_Cost]   GFN(zeros, t={t:.2f}): "
                  f"mu=[{', '.join(f'{m:+.3f}' for m in mu[0].tolist())}]  "
                  f"sigma=[{', '.join(f'{s**0.5:.3f}' for s in var[0].tolist())}]")

    # ------------------------------------------------------------------ #
    def cost_function(self, states_sequence, inputs_sequence, trial_index):
        """
        Args:
            states_sequence: [N_h+1, M, 4]
            inputs_sequence: [N_h+1, M, 1]  (unused)
        Returns:
            cost_per_step: [N_h+1]
        """
        N_h_plus_1, M, D = states_sequence.shape
        assert D == 4, f"Expected state_dim=4, got {D}"
        N_h = N_h_plus_1 - 1

        dtype = states_sequence.dtype
        device = states_sequence.device

        assert self.gp_means_seq is not None, "Use MC_PILCO_GPStats."
        gp_means = self.gp_means_seq.to(dtype=dtype, device=device)
        gp_vars = torch.clamp(self.gp_vars_seq.to(dtype=dtype, device=device),
                              min=1e-8)

        kl_per_step = torch.zeros(N_h_plus_1, dtype=dtype, device=device)

        # Transition k -> k+1 for k = 0 .. N_h-1
        for k in range(N_h):
            t_diff = k / max(N_h, 1)
            # GFN conditional from the particle's CURRENT state (detached target)
            mu_gfn, var_gfn = self._gfn_predict(states_sequence[k], t_diff)
            var_gfn = torch.clamp(var_gfn, min=1e-8)

            # GP conditional for x_{k+1} (differentiable estimate)
            gp_mu = gp_means[k + 1]
            gp_v = gp_vars[k + 1]

            if self.kl_direction == 'reverse':
                # KL(GFN || GP) -- GP var in denominator
                kl = reverse_kl_gaussian_diag(mu_gfn, var_gfn, gp_mu, gp_v)
            else:
                # KL(GP || GFN) -- GFN var in denominator
                kl = forward_kl_gaussian_diag(gp_mu, gp_v, mu_gfn, var_gfn)

            kl = torch.clamp(kl, min=0.0, max=1e6)             # [M]

            # State-probability weighting: weight each particle's KL by the
            # GP density of its CURRENT state s_k (detached sample weights).
            #   w_m = softmax_m( log gp(s_k^m) ),  loss_k = sum_m w_m * KL_m
            if self.use_state_weight:
                log_w = self._diag_gaussian_logprob(
                    states_sequence[k].detach(),
                    gp_means[k].detach(), gp_vars[k].detach())   # [M]
                w = torch.softmax(log_w, dim=0)                  # [M], sums to 1
                kl_per_step[k + 1] = (w * kl).sum()
            else:
                kl_per_step[k + 1] = kl.mean()

        # ---- Time weighting ----
        t_frac = torch.arange(N_h_plus_1, dtype=dtype, device=device) / max(N_h, 1)
        if self.weighting == 'quadratic':
            weights = t_frac ** 2
        elif self.weighting == 'linear':
            weights = t_frac
        else:
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

        self.last_kl_per_step = kl_per_step.detach()
        self.last_slack_per_step = slack_per_step.detach()
        self.last_weights = weights.detach()

        return cost_per_step

    def __call__(self, states_sequence, inputs_sequence, trial_index):
        cost_per_step = self.cost_function(states_sequence,
                                           inputs_sequence, trial_index)
        mean_cost = cost_per_step.sum()
        std_cost = torch.tensor(0.0, dtype=cost_per_step.dtype,
                                device=cost_per_step.device)
        return mean_cost, std_cost
