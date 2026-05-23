"""
UR5 Variant H cost -- LOCAL per-step Mahalanobis with time-varying target.

Structural fix for the BPTT credit-assignment failure of Variant G
----------------------------------------------------------------
Variant G uses:
    L = sum_k (k/N_h)^2 * KL( p_GFN_terminal || q_k )
with a CONSTANT terminal target. Quadratic weighting concentrates the
gradient at the endpoint, so >60% of dL/dtheta must backprop through
~150 GP rollout steps. The signal vanishes / becomes too noisy, and
the optimizer settles in a "fallen arm" local minimum where the GP
has dense data but the target is unreachable.

Variant H replaces the cost with:
    L = sum_k w_k * (1/P) sum_i (1/2) * || x_k^i - mu_ref(t_k) ||^2_{Sigma^-1}
      + alpha * sum_k slack_k
      + beta  * sum_k (1/P) sum_i || u_k^i / u_max ||^2
where:
    mu_ref(t_k) = time-varying IK waypoint at step k  (locally reachable)
    Sigma       = diag(sigma_p^2),   sigma_p from the GFN training
    w_k         = 1 (uniform)  by default;  'linear' or 'quadratic' available
    slack_k     = soft chance-constraint slack at step k
    beta        = action regularisation weight (default 0; set >0 to penalise
                  large torques and prevent free-fall / bang-bang behaviour)
    u_max       = per-joint torque limits used to normalise the action penalty
                  so joints with different scales (150 N·m vs 28 N·m) are
                  penalised equally in proportion

Why this works (and why no sigma_q in denominator):
  * Each step k has a target that is reachable from step k-1 in one
    physically plausible step. The Mahalanobis gradient at step k
    pushes particles toward mu_ref(t_k), an achievable nearby pose.
  * The gradient signal is well-defined at EVERY step, not just the
    endpoint. Even when BPTT through many steps attenuates the
    contribution from far-future steps, the local late-policy
    parameters still receive a strong gradient from steps close to k.
  * Uniform time weighting (default) distributes the optimizer's
    attention evenly along the trajectory, breaking Variant G's
    end-loading.
  * No sigma_q in any denominator -> no reverse-KL trace-term blowup
    when particles concentrate (the failure mode of Variants F/G).

For truly BPTT-free training, combine this cost with the MC_PILCO_Local
subclass (policy_learning/mc_pilco_local.py), which detaches state at
configurable intervals during the rollout. Setting bptt_truncate=1
gives a pure 1-step-lookahead policy gradient with NO backprop through
the GP chain at all.
"""

import torch

from policy_learning.ur5_gfn_prior import UR5GFNPrior
from policy_learning.ur5_chance_constraint import (
    ur5_joint_total_slack,
    UR5_Q_MIN_DEFAULT,
    UR5_Q_MAX_DEFAULT,
)


class UR5_VariantH_Cost:
    """
    Local per-step Mahalanobis tracking cost.

    Args:
        checkpoint_path: path to ur5_denoising_theta_*.pt (used only for
                         sigma_p and for terminal MMD reference samples).
        q_ref:           [N_traj+1, 6] reference positions
        dq_ref:          [N_traj+1, 6] reference velocities
        T_control:       trajectory duration (s)
        alpha:           weight on chance-constraint slack
        epsilon:         allowed violation probability
        weighting:       'uniform' (default) | 'linear' | 'quadratic'
        sigma_p_q:       per-position-dim sigma in the Mahalanobis metric.
                         If None, uses the GFN-trained value (0.10 rad).
                         Tighten (e.g., 0.05) for stricter tracking.
        sigma_p_dq:      per-velocity-dim sigma. If None, GFN-trained 0.50.
        beta:            weight on action regularisation (default 0.0 = off).
                         Penalises (u/u_max)^2 per joint per step so the
                         optimizer prefers smaller torques. Prevents free-fall
                         and bang-bang control.
        u_max:           per-joint torque limits [6] for normalising the
                         action penalty. If None, no normalisation (raw u^2).
        q_min, q_max:    UR5 joint limits (rad)
    """

    def __init__(self,
                 checkpoint_path,
                 q_ref,
                 dq_ref,
                 T_control,
                 alpha=5.0,
                 epsilon=0.10,
                 weighting='uniform',
                 sigma_p_q=None,
                 sigma_p_dq=None,
                 beta=0.0,
                 u_max=None,
                 q_min=UR5_Q_MIN_DEFAULT,
                 q_max=UR5_Q_MAX_DEFAULT,
                 num_ref_samples=512,
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

        # Action normalisation: if u_max is provided, the action penalty is
        # sum_j (u_j / u_max_j)^2 so each joint contributes ~1 at full torque.
        if u_max is not None:
            self.u_max = torch.as_tensor(u_max, dtype=dtype, device=device)
        else:
            self.u_max = None

        self.gfn_prior = UR5GFNPrior(
            checkpoint_path=checkpoint_path,
            q_ref=q_ref,
            dq_ref=dq_ref,
            T_control=T_control,
            num_ref_samples=num_ref_samples,
            dtype=dtype, device=device,
        )

        # Sigma in the Mahalanobis metric -- default to GFN-trained values.
        sigma = self.gfn_prior.sigma.clone()
        if sigma_p_q is not None:
            sigma[:6] = sigma_p_q
        if sigma_p_dq is not None:
            sigma[6:] = sigma_p_dq
        self.sigma_p        = sigma                                # [12]
        self.precision_diag = 1.0 / (self.sigma_p ** 2)            # [12]

        # Logging buffers
        self.last_divergence_per_step = None
        self.last_action_per_step     = None
        self.last_slack_per_step      = None
        self.last_weights             = None
        self.last_mu_p_traj           = None

        print(f"[UR5_VariantH_Cost] LOCAL per-step Mahalanobis "
              f"(time-varying target) | alpha={alpha} beta={beta} "
              f"epsilon={epsilon} weighting='{weighting}'")
        print(f"[UR5_VariantH_Cost] sigma_p (positions, rad):  "
              f"{self.sigma_p[:6].tolist()}")
        print(f"[UR5_VariantH_Cost] sigma_p (velocities, rad/s): "
              f"{self.sigma_p[6:].tolist()}")
        if self.beta > 0:
            print(f"[UR5_VariantH_Cost] ACTION PENALTY active: "
                  f"beta={self.beta}, "
                  f"u_max={self.u_max.tolist() if self.u_max is not None else 'None (raw u^2)'}")

    # ------------------------------------------------------------------ #
    def cost_function(self, states_sequence, inputs_sequence, trial_index):
        """
        Args:
            states_sequence: [N_h+1, num_particles, 12]
            inputs_sequence: [N_h+1, num_particles, 6]
            trial_index:     int

        Returns:
            cost_per_step: [N_h+1]
        """
        N_h_plus_1, P, D = states_sequence.shape
        assert D == 12, f"Expected state_dim=12, got {D}"
        N_h = N_h_plus_1 - 1

        dtype  = states_sequence.dtype
        device = states_sequence.device

        # Sanitise extreme particles (no division by sigma_q here so a soft
        # clip suffices; we avoid NaN/inf in pathological GP rollouts).
        states_sanitized = torch.nan_to_num(states_sequence,
                                            nan=0.0,
                                            posinf=1e3, neginf=-1e3)
        states_sanitized = torch.clamp(states_sanitized, min=-50.0, max=50.0)

        # =========================================================
        # 1) Per-step time-varying target  mu_ref(t_k)
        # =========================================================
        mu_p_traj = self.gfn_prior.get_mu_at_steps(
            N_h, dtype=dtype, device=device)                       # [N_h+1,12]
        precision = self.precision_diag.to(dtype=dtype,
                                           device=device)          # [12]

        # =========================================================
        # 2) Per-particle Mahalanobis to mu_ref(t_k) at EACH step
        #    diff: [N_h+1, P, 12];  per_particle: [N_h+1, P]
        # =========================================================
        diff = states_sanitized - mu_p_traj.unsqueeze(1)
        per_particle = 0.5 * (diff ** 2 * precision).sum(dim=-1)
        divergence_per_step = per_particle.mean(dim=1)             # [N_h+1]

        # =========================================================
        # 3) Time weighting -- DEFAULT IS UNIFORM
        # =========================================================
        t = torch.arange(N_h_plus_1, dtype=dtype, device=device) / N_h
        if self.weighting == 'uniform' or self.weighting == 'none':
            weights = torch.ones_like(t)
        elif self.weighting == 'linear':
            weights = t
        elif self.weighting == 'quadratic':
            weights = t ** 2
        weighted_div_per_step = weights * divergence_per_step

        # =========================================================
        # 4) Chance-constraint slack on joint positions
        # =========================================================
        slack_per_step, _ = ur5_joint_total_slack(
            states_sanitized,
            q_min=self.q_min, q_max=self.q_max,
            epsilon=self.epsilon,
        )

        # =========================================================
        # 5) Action regularisation: beta * (1/P) sum_i ||u_k^i / u_max||^2
        #    Penalises large torques so the optimizer prefers gentle
        #    control over bang-bang. Normalised by u_max so all joints
        #    contribute equally at full torque (~1 per joint).
        # =========================================================
        if self.beta > 0 and inputs_sequence is not None:
            u = inputs_sequence                                    # [N_h+1, P, 6]
            if self.u_max is not None:
                u_normalised = u / self.u_max.to(dtype=dtype, device=device)
            else:
                u_normalised = u
            action_per_step = (u_normalised ** 2).sum(dim=-1).mean(dim=1)  # [N_h+1]
        else:
            action_per_step = torch.zeros(N_h_plus_1,
                                          dtype=dtype, device=device)

        # =========================================================
        # 6) Total cost per step
        # =========================================================
        cost_per_step = (weighted_div_per_step
                         + self.alpha * slack_per_step
                         + self.beta  * action_per_step)
        cost_per_step = torch.nan_to_num(cost_per_step,
                                         nan=1e8, posinf=1e8, neginf=0.0)
        cost_per_step = torch.clamp(cost_per_step, max=1e8)

        # Logging (detached)
        self.last_divergence_per_step = divergence_per_step.detach()
        self.last_action_per_step     = action_per_step.detach()
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
