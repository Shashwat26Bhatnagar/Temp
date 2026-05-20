"""
UR5 Variant C cost — reverse KL with time-varying target trajectory.

    L = sum_t  w_t * KL( p_GFN(t) || q_t )  +  alpha * sum_t  slack_t

where
    q_t       = diagonal Gaussian fitted to particle moments at step t
    p_GFN(t)  = N( mu_p(t), diag(sigma^2) )
                with mu_p(t) sliding along the reference trajectory
                (the "diffusion->physical time" conversion).
                sigma is constant -- same vector the Phase-1 GFN was
                trained with.
    w_t       = (t / N_h)^2  (quadratic time weighting)
    slack_t   = soft chance-constraint slack on joint positions

KL direction is the REVERSE KL (Variant C):
    KL(p || q) = 0.5 * [ tr(Sigma_q^-1 Sigma_p)
                       + (mu_p - mu_q)^T Sigma_q^-1 (mu_p - mu_q)
                       - d
                       + log |Sigma_q| / |Sigma_p| ]
"""

import numpy as np
import torch

from policy_learning.ur5_gfn_prior import UR5GFNPrior
from policy_learning.ur5_chance_constraint import (
    ur5_joint_total_slack,
    UR5_Q_MIN_DEFAULT,
    UR5_Q_MAX_DEFAULT,
)
from policy_learning.kl_cost import (
    gaussian_moments_from_particles,
    reverse_kl_gaussian_diag,
)


class UR5_VariantC_Cost:
    """
    UR5 cost driven by the reverse Gaussian KL between the time-varying
    GFN target and the particle distribution, plus probabilistic safety.

    Args:
        checkpoint_path: path to ur5_denoising_theta_*.pt
        q_ref:           [N_traj+1, 6] joint-position reference trajectory
        dq_ref:          [N_traj+1, 6] joint-velocity reference trajectory
        T_control:       trajectory duration (s)
        alpha:           weight on chance-constraint slack
        epsilon:         allowed violation probability (e.g. 0.10)
        weighting:       'quadratic' (default) | 'linear' | 'none'
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

        # Per-call logging buffers
        self.last_divergence_per_step = None
        self.last_slack_per_step      = None
        self.last_weights             = None
        self.last_mu_p_traj           = None    # for plotting

        print(f"[UR5_VariantC_Cost] reverse KL + chance constraints | "
              f"alpha={alpha} epsilon={epsilon} weighting='{weighting}'")
        print(f"[UR5_VariantC_Cost] q_min={list(q_min)}")
        print(f"[UR5_VariantC_Cost] q_max={list(q_max)}")

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

        # Sanitise runaway particles (early-trial GP extrapolation can produce
        # +/- inf states which would make moment fitting NaN). Clamp positions
        # to a wide safe band and velocities to a sane maximum.
        states_sanitized = torch.nan_to_num(states_sequence,
                                            nan=0.0, posinf=1e3, neginf=-1e3)
        states_sanitized = torch.clamp(states_sanitized, min=-50.0, max=50.0)

        # ============================================================
        # 1) Per-step REVERSE KL with TIME-VARYING target mean
        # ============================================================
        mu_q, Sigma_q_diag = gaussian_moments_from_particles(states_sanitized)
        # mu_q: [N_h+1, 12]   Sigma_q_diag: [N_h+1, 12]

        # Diffusion-time -> physical-time mapping:
        # mu_p(k) = [q_ref(k/N_h * T_control), dq_ref(k/N_h * T_control)]
        mu_p_traj = self.gfn_prior.get_mu_at_steps(N_h, dtype=dtype, device=device)
        # mu_p_traj: [N_h+1, 12]

        Sigma_p_diag = self.gfn_prior.Sigma_p_diag.to(dtype=dtype, device=device)
        # Sigma_p_diag: [12]  (broadcast across time)

        divergence_per_step = reverse_kl_gaussian_diag(
            mu_p_traj, Sigma_p_diag, mu_q, Sigma_q_diag)  # [N_h+1]

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

        # Final NaN/Inf guard so the optimiser sees a finite signal
        cost_per_step = torch.nan_to_num(cost_per_step,
                                         nan=1e8, posinf=1e8, neginf=0.0)
        cost_per_step = torch.clamp(cost_per_step, max=1e8)

        # Logging
        self.last_divergence_per_step = divergence_per_step.detach()
        self.last_slack_per_step      = slack_per_step.detach()
        self.last_weights             = weights.detach()
        self.last_mu_p_traj           = mu_p_traj.detach()

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
