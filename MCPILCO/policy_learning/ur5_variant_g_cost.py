"""
UR5 Variant G cost -- GOAL-REACHING (not trajectory tracking).

Aligns the task structure with what the Phase-1 GFN actually learned:
  * GFN diffusion process: starts at zeros (12-D origin), ends at the
    trained terminal target N(q_ref(4s), diag(sigma_p^2)).
  * MC-PILCO trajectory (in this variant): starts at zeros (matching the
    GFN's initial state), drives to q_ref(4s) (the GFN's terminal target).

This is the UR5 analog of CartPole swing-up: a single-point regulation
task, not a pointwise trajectory tracking task.  Earlier UR5 variants
(C, D, E, F) tried to track q_ref(t) at every step -- a structurally
different problem from what the GFN was trained for.  Variant G fixes
this mismatch.

Cost:
    L = sum_t  w_t * KL( p_GFN || q_t )  +  alpha * sum_t  slack_t

where
    p_GFN     = N( mu_p_final, diag(sigma_p^2) )     -- CONSTANT in time
                (this is the GFN's actual terminal target)
    q_t       = Gaussian fitted to particle moments at step t
    w_t       = (t / N_h)^2  (quadratic, emphasises endpoint)
    slack_t   = soft chance-constraint slack at step t

Note: sigma_p is left at GFN-trained values (0.10 / 0.50), not
artificially tightened. Reverse KL plus matched sigma_p is exactly the
combination that worked for CartPole Variant C.
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


class UR5_VariantG_Cost:
    """
    Goal-reaching UR5 cost: reverse KL between particles and the FIXED
    GFN terminal target, with chance constraints on joint positions.

    Args:
        checkpoint_path: path to ur5_denoising_theta_*.pt
        q_ref:           [N_traj+1, 6] reference trajectory (used only to
                         identify the terminal target q_ref(4s))
        dq_ref:          [N_traj+1, 6] reference velocities
        T_control:       trajectory duration (s)
        alpha:           weight on chance-constraint slack
        epsilon:         allowed violation probability
        weighting:       'quadratic' | 'linear' | 'none'
        sigma_p_q:       override target sigma for positions
                         (None -> use GFN-trained value 0.10)
        sigma_p_dq:      override target sigma for velocities
                         (None -> use GFN-trained value 0.50)
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
                 sigma_p_q=None,         # default = GFN-trained 0.10
                 sigma_p_dq=None,        # default = GFN-trained 0.50
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

        # ---- Sigma override (default: keep GFN-trained values) ----
        sigma = self.gfn_prior.sigma.clone()
        if sigma_p_q is not None:
            sigma[:6] = sigma_p_q
        if sigma_p_dq is not None:
            sigma[6:] = sigma_p_dq
        self.sigma_p = sigma
        self.Sigma_p_diag = self.sigma_p ** 2

        # ---- CONSTANT target = GFN's terminal config q_ref(4s) ----
        # This is the key difference from C/D/E/F: no time-varying mu_p.
        # The target is the same at every step -- but quadratic weighting
        # makes it matter most at the endpoint.
        self.mu_p_final = self.gfn_prior.mu_p_final.detach().clone()  # [12]

        # Logging buffers
        self.last_divergence_per_step = None
        self.last_slack_per_step      = None
        self.last_weights             = None
        self.last_mu_p                = None

        print(f"[UR5_VariantG_Cost] GOAL-REACHING + reverse KL + chance "
              f"constraints | alpha={alpha} epsilon={epsilon} "
              f"weighting='{weighting}'")
        print(f"[UR5_VariantG_Cost] FIXED target mu_p = q_ref(4s) = "
              f"{self.mu_p_final[:6].tolist()}  (positions)")
        print(f"[UR5_VariantG_Cost] FIXED target dq_ref(4s) = "
              f"{self.mu_p_final[6:].tolist()}  (velocities)")
        print(f"[UR5_VariantG_Cost] sigma_p (positions): "
              f"{self.sigma_p[:6].tolist()}")
        print(f"[UR5_VariantG_Cost] sigma_p (velocities): "
              f"{self.sigma_p[6:].tolist()}")

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

        # Sanitise runaway particles
        states_sanitized = torch.nan_to_num(states_sequence,
                                            nan=0.0, posinf=1e3, neginf=-1e3)
        states_sanitized = torch.clamp(states_sanitized, min=-50.0, max=50.0)

        # ============================================================
        # 1) Per-step REVERSE KL with CONSTANT target = q_ref(4s)
        # ============================================================
        mu_q, Sigma_q_diag = gaussian_moments_from_particles(states_sanitized)
        # mu_q:        [N_h+1, 12]
        # Sigma_q_diag:[N_h+1, 12]

        # Constant target across all timesteps -- broadcast to [N_h+1, 12]
        mu_p_const = self.mu_p_final.to(dtype=dtype, device=device)        # [12]
        mu_p_traj  = mu_p_const.unsqueeze(0).expand(N_h_plus_1, -1)        # [N_h+1, 12]
        Sigma_p_diag = self.Sigma_p_diag.to(dtype=dtype, device=device)    # [12]

        divergence_per_step = reverse_kl_gaussian_diag(
            mu_p_traj, Sigma_p_diag, mu_q, Sigma_q_diag)                   # [N_h+1]

        # ============================================================
        # 2) Quadratic time weighting w_t = (t/N_h)^2  (emphasise endpoint)
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

        # Safety floor
        cost_per_step = torch.nan_to_num(cost_per_step,
                                         nan=1e8, posinf=1e8, neginf=0.0)
        cost_per_step = torch.clamp(cost_per_step, max=1e8)

        # Logging (detached)
        self.last_divergence_per_step = divergence_per_step.detach()
        self.last_slack_per_step      = slack_per_step.detach()
        self.last_weights             = weights.detach()
        self.last_mu_p                = mu_p_const.detach()

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
