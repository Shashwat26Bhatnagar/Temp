"""
UR5 Variant J cost -- Per-particle GP-vs-GFN reverse KL with time-varying
target.

    L = sum_t  w_t * (1/M) sum_m  KL(p_GFN(t) || GP_m(t))
      + alpha * sum_t  slack_t
      + beta  * sum_t  (1/M) sum_m || u_t^m / u_max ||^2

where
    GP_m(t)  = N(mu_GP_m(t), diag(sigma^2_GP_m(t)))
               Per-particle GP prediction Gaussian. EXACT -- each GP
               posterior is a true Gaussian, not a moment-matched surrogate.
    p_GFN(t) = N(mu_ref(t), diag(sigma_p^2))
               Time-varying GFN target sliding along the IK reference
               trajectory.
    w_t      = time weighting ('uniform' default)
    slack_t  = soft chance-constraint slack on UR5 joint limits
    beta     = optional action regularisation weight

Differences from earlier UR5 variants
--------------------------------------
Variant C/F/G: collapse all particles into ONE moment-matched Gaussian,
               then closed-form KL between two Gaussians.
               -> loses mixture structure, sign-correction trick.

Variant E:     per-particle cross-entropy  -log p_GFN(x_i).
               -> uses particle POSITIONS, discards GP variance info.

Variant H:     per-particle Mahalanobis to mu_ref.
               -> uses particle positions, discards GP variance.
               Works well but treats variance information as noise.

Variant J:     per-particle KL using the EXACT GP Gaussian per particle.
               -> no moment matching, no information loss. Both sides of
               the KL are true Gaussians. Uses GP variance as a
               precision-weighted confidence signal.

Requires MC_PILCO_GPStats
--------------------------
Stores gp_means_seq [N_h+1, M, 12] and gp_vars_seq [N_h+1, M, 12] on
this cost object before each call.
"""

import torch

from policy_learning.ur5_gfn_prior import UR5GFNPrior
from policy_learning.ur5_chance_constraint import (
    ur5_joint_total_slack,
    UR5_Q_MIN_DEFAULT,
    UR5_Q_MAX_DEFAULT,
)
from policy_learning.kl_cost import reverse_kl_gaussian_diag


class UR5_VariantJ_Cost:
    """
    Per-particle GP-vs-GFN reverse KL for UR5 trajectory tracking.

    Attributes set externally by MC_PILCO_GPStats.apply_policy:
        gp_means_seq : [N_h+1, M, 12]  per-particle GP state prediction mean
        gp_vars_seq  : [N_h+1, M, 12]  per-particle GP state prediction var
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

        # Sigma for the GFN target -- default to GFN-trained values
        sigma = self.gfn_prior.sigma.clone()
        if sigma_p_q is not None:
            sigma[:6] = sigma_p_q
        if sigma_p_dq is not None:
            sigma[6:] = sigma_p_dq
        self.sigma_p        = sigma                           # [12]
        self.Sigma_p_diag   = sigma ** 2                      # [12]

        # These will be set by MC_PILCO_GPStats
        self.gp_means_seq = None
        self.gp_vars_seq = None

        # Logging buffers
        self.last_kl_per_step      = None
        self.last_action_per_step  = None
        self.last_slack_per_step   = None
        self.last_weights          = None
        self.last_mu_p_traj        = None

        print(f"[UR5_VariantJ_Cost] PER-PARTICLE GP-vs-GFN reverse KL "
              f"(time-varying target)")
        print(f"[UR5_VariantJ_Cost] alpha={alpha} beta={beta} "
              f"epsilon={epsilon} weighting='{weighting}'")
        print(f"[UR5_VariantJ_Cost] sigma_p (pos):  {self.sigma_p[:6].tolist()}")
        print(f"[UR5_VariantJ_Cost] sigma_p (vel):  {self.sigma_p[6:].tolist()}")

    # ------------------------------------------------------------------ #
    def cost_function(self, states_sequence, inputs_sequence, trial_index):
        """
        Args:
            states_sequence: [N_h+1, M, 12]  (used for chance constraints)
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

        # ============================================================
        # 1) Time-varying target mean from GFN prior
        # ============================================================
        mu_p_traj = self.gfn_prior.get_mu_at_steps(
            N_h, dtype=dtype, device=device)                  # [N_h+1, 12]
        Sigma_p = self.Sigma_p_diag.to(dtype=dtype, device=device)  # [12]

        # ============================================================
        # 2) Per-particle reverse KL using EXACT GP Gaussians
        # ============================================================
        assert self.gp_means_seq is not None, (
            "gp_means_seq not set! Use MC_PILCO_GPStats.")
        assert self.gp_vars_seq is not None, (
            "gp_vars_seq not set! Use MC_PILCO_GPStats.")

        gp_means = self.gp_means_seq.to(dtype=dtype, device=device)
        gp_vars = self.gp_vars_seq.to(dtype=dtype, device=device)

        # Sanitise GP stats
        gp_means = torch.nan_to_num(gp_means, nan=0.0, posinf=1e3, neginf=-1e3)
        gp_means = torch.clamp(gp_means, min=-50.0, max=50.0)
        gp_vars = torch.nan_to_num(gp_vars, nan=1e-6, posinf=1e6, neginf=1e-12)
        gp_vars = torch.clamp(gp_vars, min=1e-8, max=1e6)

        kl_per_step = torch.zeros(N_h_plus_1, dtype=dtype, device=device)
        for t in range(N_h_plus_1):
            # KL(p_GFN(t) || GP_m(t)) for each particle m
            kl_per_particle = reverse_kl_gaussian_diag(
                mu_p_traj[t],   # [12]   target mean at this step
                Sigma_p,        # [12]   target variance (constant)
                gp_means[t],    # [M, 12] per-particle GP mean
                gp_vars[t],     # [M, 12] per-particle GP variance
            )  # [M]
            kl_per_particle = torch.clamp(kl_per_particle, min=0.0, max=1e6)
            kl_per_step[t] = kl_per_particle.mean()

        # ============================================================
        # 3) Time weighting
        # ============================================================
        t_frac = torch.arange(N_h_plus_1, dtype=dtype, device=device) / max(N_h, 1)
        if self.weighting == 'quadratic':
            weights = t_frac ** 2
        elif self.weighting == 'linear':
            weights = t_frac
        else:  # 'uniform' or 'none'
            weights = torch.ones_like(t_frac)

        weighted_kl = weights * kl_per_step

        # ============================================================
        # 4) Chance-constraint slack on joint positions
        # ============================================================
        slack_per_step, _ = ur5_joint_total_slack(
            states_sequence,
            q_min=self.q_min, q_max=self.q_max,
            epsilon=self.epsilon,
        )

        # ============================================================
        # 5) Optional action regularisation
        # ============================================================
        if self.beta > 0 and inputs_sequence is not None:
            if self.u_max is not None:
                u_norm = inputs_sequence / self.u_max.unsqueeze(0).unsqueeze(0)
            else:
                u_norm = inputs_sequence
            action_per_step = (u_norm ** 2).sum(dim=-1).mean(dim=1)
        else:
            action_per_step = torch.zeros(N_h_plus_1, dtype=dtype, device=device)

        # ============================================================
        # 6) Total cost per step
        # ============================================================
        cost_per_step = (weighted_kl
                         + self.alpha * slack_per_step
                         + self.beta * action_per_step)

        # Logging (detached)
        self.last_kl_per_step     = kl_per_step.detach()
        self.last_action_per_step = action_per_step.detach()
        self.last_slack_per_step  = slack_per_step.detach()
        self.last_weights         = weights.detach()
        self.last_mu_p_traj       = mu_p_traj.detach()

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
