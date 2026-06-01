"""
Variant J: Per-particle GP-vs-GFN reverse KL for cartpole.

Instead of collapsing all particles into a single moment-matched Gaussian
and computing KL between two Gaussian surrogates (as Variant C does),
Variant J uses the EXACT per-particle GP prediction Gaussian directly:

    L = sum_t  w_t * (1/M) sum_m  KL(p_GFN || GP_m(t))
      + alpha * sum_t  slack_t

where
    GP_m(t)  = N(mu_GP_m(t), diag(sigma^2_GP_m(t)))
               The GP's Gaussian prediction for particle m at time t.
               This is EXACT -- not a moment-matched approximation.
    p_GFN    = N(mu_p, diag(sigma^2_p))
               Frozen Phase-1 GFN target (analytical Gaussian).
    w_t      = (t / N_h)^2  (quadratic time weighting)
    slack_t  = soft chance-constraint slack at step t

Why this is better than Variant C
----------------------------------
Variant C fits a SINGLE Gaussian to all M particles (moment matching),
losing the mixture structure.  The moment-matched KL between the fitted
Gaussian and the target is a surrogate divergence that:
  * Underestimates the true divergence for multimodal particle clouds
  * Involves a subtle sign-correction trick (see earlier analysis)

Variant J uses each GP output DIRECTLY.  Since:
  * Each GP posterior is an EXACT Gaussian (not an approximation)
  * The GFN target is an EXACT Gaussian
the per-particle KL is exact -- no moment matching, no sign correction,
no mixture-of-Gaussians collapse.

Requires MC_PILCO_GPStats
--------------------------
The standard MC_PILCO.apply_policy discards GP means/variances.
MC_PILCO_GPStats captures them and stores:
    self.gp_means_seq : [N_h+1, M, D]
    self.gp_vars_seq  : [N_h+1, M, D]
on the cost function object before each cost evaluation.
"""

import math
import torch

from policy_learning.gfn_prior import GFNPrior
from policy_learning.chance_constraint import cartpole_total_slack
from policy_learning.kl_cost import reverse_kl_gaussian_diag


class VariantJ_Cost:
    """
    Variant J: per-particle GP-vs-GFN reverse KL for cartpole.

    Attributes set externally by MC_PILCO_GPStats.apply_policy:
        gp_means_seq : [N_h+1, M, 4]  per-particle GP state prediction mean
        gp_vars_seq  : [N_h+1, M, 4]  per-particle GP state prediction variance
    """

    def __init__(self,
                 checkpoint_path,
                 alpha=5.0,
                 epsilon=0.10,
                 weighting='quadratic',
                 position_bound=2.4,
                 angle_bound=0.35,
                 num_ref_samples=512,
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

        # These will be set by MC_PILCO_GPStats before each cost call
        self.gp_means_seq = None
        self.gp_vars_seq = None

        # Per-call logging buffers (detached)
        self.last_kl_per_step = None
        self.last_slack_per_step = None
        self.last_weights = None

        print(f"[VariantJ_Cost] per-particle GP-vs-GFN reverse KL")
        print(f"[VariantJ_Cost] alpha = {alpha}, epsilon = {epsilon}, "
              f"weighting = '{weighting}'")
        print(f"[VariantJ_Cost] position_bound = {position_bound} m, "
              f"angle_bound = {angle_bound} rad "
              f"({angle_bound * 180 / math.pi:.2f} deg around theta=pi)")

    # ------------------------------------------------------------------ #
    # Cost API                                                            #
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

        # ============================================================
        # 1) Per-particle reverse KL using EXACT GP Gaussians
        # ============================================================
        assert self.gp_means_seq is not None, (
            "gp_means_seq not set! Use MC_PILCO_GPStats instead of MC_PILCO.")
        assert self.gp_vars_seq is not None, (
            "gp_vars_seq not set! Use MC_PILCO_GPStats instead of MC_PILCO.")

        gp_means = self.gp_means_seq.to(dtype=dtype, device=device)  # [T, M, D]
        gp_vars = self.gp_vars_seq.to(dtype=dtype, device=device)    # [T, M, D]

        # Floor GP variance for numerical stability
        gp_vars = torch.clamp(gp_vars, min=1e-8)

        mu_p = self.gfn_prior.mu_p.to(dtype=dtype, device=device)          # [D]
        Sigma_p = self.gfn_prior.Sigma_p_diag.to(dtype=dtype, device=device)  # [D]

        # KL(p_GFN || GP_m) for each particle at each timestep
        # reverse_kl_gaussian_diag handles batched mu_q: [..., D] -> [...]
        kl_per_step = torch.zeros(N_h_plus_1, dtype=dtype, device=device)
        for t in range(N_h_plus_1):
            # mu_q: [M, D],  Sigma_q: [M, D]
            kl_per_particle = reverse_kl_gaussian_diag(
                mu_p, Sigma_p, gp_means[t], gp_vars[t]
            )  # [M]
            # Clamp individual KL values to avoid extreme outliers
            kl_per_particle = torch.clamp(kl_per_particle, min=0.0, max=1e6)
            kl_per_step[t] = kl_per_particle.mean()

        # ============================================================
        # 2) Time weighting
        # ============================================================
        t_frac = torch.arange(N_h_plus_1, dtype=dtype, device=device) / N_h
        if self.weighting == 'quadratic':
            weights = t_frac ** 2
        elif self.weighting == 'linear':
            weights = t_frac
        else:  # 'none' or 'uniform'
            weights = torch.ones_like(t_frac)

        weighted_kl = weights * kl_per_step

        # ============================================================
        # 3) Soft chance-constraint slack
        # ============================================================
        slack_per_step, _ = cartpole_total_slack(
            states_sequence,
            position_bound=self.position_bound,
            angle_bound=self.angle_bound,
            epsilon=self.epsilon,
        )

        # ============================================================
        # 4) Total cost per step
        # ============================================================
        cost_per_step = weighted_kl + self.alpha * slack_per_step

        # Logging (detached)
        self.last_kl_per_step = kl_per_step.detach()
        self.last_slack_per_step = slack_per_step.detach()
        self.last_weights = weights.detach()

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
