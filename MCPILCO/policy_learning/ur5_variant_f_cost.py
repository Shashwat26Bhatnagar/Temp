"""
UR5 Variant F cost -- REVERSE KL on fitted Gaussian moments,
with the same tight target sigma as Variants D (W2) and E (per-particle
forward KL). This provides a clean apples-to-apples comparison.

    L = sum_t  w_t * KL( p_GFN(t) || q_t )  +  alpha * sum_t  slack_t

where
    q_t       = diagonal Gaussian fitted to particle moments at step t
    p_GFN(t)  = N( mu_p(t), diag(sigma_p^2) )  with mu_p(t) sliding along
                the reference trajectory (diffusion -> physical time)
    w_t       = (t / N_h)^2
    slack_t   = soft chance-constraint slack at step t

For diagonal Gaussians the reverse KL has closed form

    KL(p || q) = 0.5 * sum_d [
                    sigma_{p,d}^2 / sigma_{q,d}^2
                  + (mu_{q,d} - mu_{p,d})^2 / sigma_{q,d}^2
                  - 1
                  + log( sigma_{q,d}^2 / sigma_{p,d}^2 ) ]

Variant F differs from the existing UR5 Variant C only in target sigma:
  C: sigma_p from GFN training  (0.10 positions, 0.50 velocities)
  F: tight sigma_p              (0.05 positions, 0.20 velocities)
                                  -- same as D, E

Why we run F at all
-------------------
Variants D (W2) and E (forward KL) both failed differently on UR5 tracking.
F closes the experimental design so we can attribute UR5 behaviour to the
KL DIRECTION rather than to the target tightness. If F also fails (likely
given Variant C's UR5 failure), it confirms that distributional reverse
KL is fundamentally mis-aligned with trajectory tracking, regardless of
target tightness.
"""

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


class UR5_VariantF_Cost:
    """
    UR5 cost driven by reverse Gaussian KL between time-varying target
    and the particle-fitted Gaussian, plus probabilistic safety.

    Args:
        checkpoint_path: path to ur5_denoising_theta_*.pt
        q_ref:           [N_traj+1, 6] joint-position reference trajectory
        dq_ref:          [N_traj+1, 6] joint-velocity reference trajectory
        T_control:       trajectory duration (s)
        alpha:           weight on chance-constraint slack
        epsilon:         allowed violation probability
        weighting:       'quadratic' | 'linear' | 'none'
        sigma_p_q:       per-position-dim sigma in the target (rad).
                         Default 0.05 (matches D, E).
        sigma_p_dq:      per-velocity-dim sigma in the target (rad/s).
                         Default 0.20 (matches D, E).
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
                 sigma_p_q=0.05,
                 sigma_p_dq=0.20,
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

        # ---- Sigma override (matches Variants D and E for fair comparison)
        sigma = self.gfn_prior.sigma.clone()
        if sigma_p_q is not None:
            sigma[:6] = sigma_p_q
        if sigma_p_dq is not None:
            sigma[6:] = sigma_p_dq
        self.sigma_p = sigma                                  # [12]
        self.Sigma_p_diag = self.sigma_p ** 2                 # [12]

        # Logging buffers
        self.last_divergence_per_step = None
        self.last_slack_per_step      = None
        self.last_weights             = None
        self.last_mu_p_traj           = None

        print(f"[UR5_VariantF_Cost] REVERSE KL + chance constraints "
              f"(tight sigma matching Variants D, E) | "
              f"alpha={alpha} epsilon={epsilon} weighting='{weighting}'")
        print(f"[UR5_VariantF_Cost] sigma_p (positions, rad):  "
              f"{self.sigma_p[:6].tolist()}")
        print(f"[UR5_VariantF_Cost] sigma_p (velocities, rad/s): "
              f"{self.sigma_p[6:].tolist()}")
        print(f"[UR5_VariantF_Cost] q_min={list(q_min)}")
        print(f"[UR5_VariantF_Cost] q_max={list(q_max)}")

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

        # Sanitise runaway particles (reverse KL has sigma_q in denominator
        # so extreme variances are dangerous; the 1e-8 floor inside
        # gaussian_moments_from_particles handles complete collapse, but
        # we still clamp magnitude).
        states_sanitized = torch.nan_to_num(states_sequence,
                                            nan=0.0, posinf=1e3, neginf=-1e3)
        states_sanitized = torch.clamp(states_sanitized, min=-50.0, max=50.0)

        # ============================================================
        # 1) Per-step REVERSE KL with TIME-VARYING target mean
        # ============================================================
        mu_q, Sigma_q_diag = gaussian_moments_from_particles(states_sanitized)
        # mu_q:        [N_h+1, 12]
        # Sigma_q_diag:[N_h+1, 12]

        mu_p_traj = self.gfn_prior.get_mu_at_steps(
            N_h, dtype=dtype, device=device)                  # [N_h+1, 12]
        Sigma_p_diag = self.Sigma_p_diag.to(dtype=dtype,
                                            device=device)    # [12]

        # KL(p_GFN || q)  -- note arg order vs Variant E's log_density
        divergence_per_step = reverse_kl_gaussian_diag(
            mu_p_traj, Sigma_p_diag, mu_q, Sigma_q_diag)      # [N_h+1]

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

        # Safety floor (reverse KL with sigma_q -> 0 can produce inf)
        cost_per_step = torch.nan_to_num(cost_per_step,
                                         nan=1e8, posinf=1e8, neginf=0.0)
        cost_per_step = torch.clamp(cost_per_step, max=1e8)

        # Logging (detached)
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
