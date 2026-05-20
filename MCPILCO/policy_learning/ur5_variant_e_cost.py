"""
UR5 Variant E cost -- PER-PARTICLE cross-entropy with the time-varying
GFN target, plus chance constraints on joint position bounds.

    L = sum_t  w_t * (1/P) sum_i  [-log p_GFN(x_t^(i); t)]
      + alpha * sum_t  slack_t

where x_t^(i) is the i-th particle at time t and p_GFN(.; t) is the
time-varying GFN target  N( mu_p(t), diag(sigma_p^2) ).

For a diagonal Gaussian target the per-particle negative log-density is

    -log p_GFN(x_i) = 0.5 * sum_d (x_{i,d} - mu_{p,d})^2 / sigma_{p,d}^2
                      + const

Differences from earlier variants
---------------------------------

Variant D (W2 on moments)              -> particles collapsed to (mu_q, sigma_q)
                                          then closed-form W2 vs target.
                                          FAILED on UR5: "wide cloud with right
                                          mean" loophole + variance-mismatch
                                          dominated cost.

Variant E (per-particle cross-entropy) -> NO collapsing.  Each particle gets
                                          its own contribution.  The GFN's
                                          sigma_p acts as a PER-DIMENSION
                                          PRECISION WEIGHTING -- tight on
                                          positions, loose on velocities.

Why this should work for UR5
----------------------------
  * Each particle that drifts off-trajectory contributes positive cost.
    The mean-cancellation cheat is impossible.
  * Gradient flows through every particle independently (no moment fit
    in the middle that destroys per-particle information).
  * The GFN's sigma_p is used non-trivially: positions weighted ~1/0.05^2
    while velocities weighted ~1/0.20^2 -- about 25x stronger emphasis on
    position tracking, matching the task's structure.
  * Closed-form, cheap to compute (no log, no division by particle stats).
"""

import torch

from policy_learning.ur5_gfn_prior import UR5GFNPrior
from policy_learning.ur5_chance_constraint import (
    ur5_joint_total_slack,
    UR5_Q_MIN_DEFAULT,
    UR5_Q_MAX_DEFAULT,
)
from policy_learning.kl_cost import gaussian_moments_from_particles


class UR5_VariantE_Cost:
    """
    UR5 cost driven by per-particle cross-entropy under the time-varying
    GFN target, plus probabilistic safety on joint position bounds.

    Args:
        checkpoint_path: path to ur5_denoising_theta_*.pt
        q_ref:           [N_traj+1, 6] joint-position reference trajectory
        dq_ref:          [N_traj+1, 6] joint-velocity reference trajectory
        T_control:       trajectory duration (s)
        alpha:           weight on chance-constraint slack
        epsilon:         allowed violation probability
        weighting:       'quadratic' | 'linear' | 'none'
        sigma_p_q:       per-position-dim sigma in the target (rad). If None,
                         use the GFN's trained value (0.10). For tighter
                         tracking, pass e.g. 0.02-0.05.
        sigma_p_dq:      per-velocity-dim sigma in the target (rad/s).
                         If None, use 0.50 (GFN-trained).
        q_min, q_max:    UR5 joint limits (rad)
    """

    def __init__(self,
                 checkpoint_path,
                 q_ref,
                 dq_ref,
                 T_control,
                 alpha=5.0,
                 epsilon=0.10,
                 weighting='quadratic',
                 sigma_p_q=None,
                 sigma_p_dq=None,
                 q_min=UR5_Q_MIN_DEFAULT,
                 q_max=UR5_Q_MAX_DEFAULT,
                 num_ref_samples=512,
                 dtype=torch.float64,
                 device=torch.device('cpu')):
        assert weighting in ('quadratic', 'linear', 'none', 'uniform'), \
            f"Unknown weighting '{weighting}'."

        self.alpha     = alpha
        self.epsilon   = epsilon
        self.weighting = weighting
        self.q_min     = q_min
        self.q_max     = q_max
        self.dtype     = dtype
        self.device    = device

        self.gfn_prior = UR5GFNPrior(
            checkpoint_path=checkpoint_path,
            q_ref=q_ref,
            dq_ref=dq_ref,
            T_control=T_control,
            num_ref_samples=num_ref_samples,
            dtype=dtype, device=device,
        )

        # Optional sigma override -- same convention as Variant D.
        sigma = self.gfn_prior.sigma.clone()
        if sigma_p_q is not None:
            sigma[:6] = sigma_p_q
        if sigma_p_dq is not None:
            sigma[6:] = sigma_p_dq
        self.sigma_p = sigma                                  # [12]
        # Pre-compute inverse variance (precision diagonal) used at every step
        self.precision_diag = 1.0 / (self.sigma_p ** 2)       # [12]

        # Logging buffers
        self.last_per_particle_mean = None
        self.last_slack_per_step    = None
        self.last_weights           = None
        self.last_mu_p_traj         = None

        print(f"[UR5_VariantE_Cost] PER-PARTICLE cross-entropy + chance "
              f"constraints | alpha={alpha} epsilon={epsilon} "
              f"weighting='{weighting}'")
        print(f"[UR5_VariantE_Cost] sigma_p (positions, rad):  "
              f"{self.sigma_p[:6].tolist()}")
        print(f"[UR5_VariantE_Cost] sigma_p (velocities, rad/s): "
              f"{self.sigma_p[6:].tolist()}")
        print(f"[UR5_VariantE_Cost] q_min={list(q_min)}")
        print(f"[UR5_VariantE_Cost] q_max={list(q_max)}")

    # ------------------------------------------------------------------ #
    # Cost API expected by MC-PILCO                                      #
    # ------------------------------------------------------------------ #
    def cost_function(self, states_sequence, inputs_sequence, trial_index):
        """
        Args:
            states_sequence: [N_h+1, num_particles, 12]
            inputs_sequence: [N_h+1, num_particles, 6]   (unused)
            trial_index:     int

        Returns:
            cost_per_step: [N_h+1]
        """
        N_h_plus_1, P, D = states_sequence.shape
        assert D == 12, f"Expected state_dim=12, got {D}"
        N_h = N_h_plus_1 - 1

        dtype  = states_sequence.dtype
        device = states_sequence.device

        # Sanitise runaway particles (no division by sigma_q here, but
        # extreme states can still produce huge values).
        states_sanitized = torch.nan_to_num(states_sequence,
                                            nan=0.0, posinf=1e3, neginf=-1e3)
        states_sanitized = torch.clamp(states_sanitized, min=-50.0, max=50.0)

        # ============================================================
        # 1) Per-particle negative log-density under the GFN target
        # ============================================================
        mu_p_traj = self.gfn_prior.get_mu_at_steps(
            N_h, dtype=dtype, device=device)                  # [N_h+1, 12]
        precision = self.precision_diag.to(dtype=dtype,
                                           device=device)     # [12]

        # diff: [N_h+1, P, 12]
        diff = states_sanitized - mu_p_traj.unsqueeze(1)
        # Weighted squared error per particle, summed over dimensions.
        # = -log p_GFN(x) up to additive constant.
        per_particle_neg_log_p = 0.5 * (diff ** 2 * precision).sum(dim=-1)
        # Average across particles -> [N_h+1]
        divergence_per_step = per_particle_neg_log_p.mean(dim=1)

        # ============================================================
        # 2) Quadratic time weighting w_t = (t/N_h)^2
        # ============================================================
        t = torch.arange(N_h_plus_1, dtype=dtype, device=device) / N_h
        if self.weighting == 'quadratic':
            weights = t ** 2
        elif self.weighting == 'linear':
            weights = t
        else:
            weights = torch.ones_like(t)
        weighted_div_per_step = weights * divergence_per_step

        # ============================================================
        # 3) Chance-constraint slack on joint positions
        #    (still based on moment-fitted particle Gaussian, since
        #     Pr(constraint violated) inherently needs a distribution)
        # ============================================================
        slack_per_step, _ = ur5_joint_total_slack(
            states_sanitized,
            q_min=self.q_min, q_max=self.q_max,
            epsilon=self.epsilon,
        )

        # ============================================================
        # 4) Total cost per step
        # ============================================================
        cost_per_step = weighted_div_per_step + self.alpha * slack_per_step

        # Safety floor (per-particle cost is well-behaved, but the
        # chance-constraint slack uses fitted moments which could spike).
        cost_per_step = torch.nan_to_num(cost_per_step,
                                         nan=1e8, posinf=1e8, neginf=0.0)
        cost_per_step = torch.clamp(cost_per_step, max=1e8)

        # Logging (detached)
        self.last_per_particle_mean = divergence_per_step.detach()
        self.last_slack_per_step    = slack_per_step.detach()
        self.last_weights           = weights.detach()
        self.last_mu_p_traj         = mu_p_traj.detach()

        return cost_per_step

    def __call__(self, states_sequence, inputs_sequence, trial_index):
        cost_per_step = self.cost_function(states_sequence,
                                           inputs_sequence,
                                           trial_index)
        mean_cost = cost_per_step.sum()
        std_cost  = torch.tensor(0.0,
                                 dtype=cost_per_step.dtype,
                                 device=cost_per_step.device)
        return mean_cost, std_cost
