"""
UR5 Variant I cost -- LOCAL per-step Jensen-Shannon Divergence (JSD)
with time-varying target.

Same architecture as Variant H (local per-step, time-varying target,
uniform weighting, optional BPTT truncation, initial state at zeros)
but replaces the Mahalanobis distance with Jensen-Shannon Divergence.

Variant H cost at step k:
    L_k = (1/P) sum_i  0.5 * || x_k^i - mu_ref(t_k) ||^2_{Sigma^-1}

Variant I replaces this with:
    L_k = JSD( q_k || p_k )
where
    q_k = N(mu_q_k, diag(var_q_k))  fitted from particles at step k
    p_k = N(mu_ref(t_k), diag(sigma_p^2))  reference from GFN

Jensen-Shannon Divergence:
    JSD(q || p) = 0.5 * KL(q || M) + 0.5 * KL(p || M)
    M = 0.5 * (q + p)   (mixture distribution)

Since M is a Gaussian MIXTURE (not a Gaussian), JSD has no closed form
for Gaussians. We estimate it via Monte Carlo:
    * KL(q || M) is estimated using the rollout particles (samples from q).
    * KL(p || M) is estimated using fresh samples drawn from p.
Both terms evaluate the mixture density M via logsumexp, which is
numerically stable.

Properties that make JSD attractive vs KL:
  * Bounded: 0 <= JSD <= ln(2) ~ 0.693 -- no unbounded blow-up.
  * Symmetric: treats p and q equally. KL(q||p) mode-seeks (collapses
    variance); KL(p||q) mode-covers (inflates variance). JSD balances
    both tendencies.
  * sqrt(JSD) is a proper metric (triangle inequality).
  * No sigma_q in any denominator alone -- avoids the reverse-KL
    trace-term blow-up (sigma_p^2 / sigma_q^2 -> inf when particles
    concentrate). In JSD, sigma_q appears inside logsumexp(log_q, log_p),
    which is stable.
  * Differentiable through particles via the fitted Gaussian moments
    (mu_q, var_q depend on particles which depend on policy params theta).
"""

import math

import torch

from policy_learning.ur5_gfn_prior import UR5GFNPrior
from policy_learning.ur5_chance_constraint import (
    ur5_joint_total_slack,
    UR5_Q_MIN_DEFAULT,
    UR5_Q_MAX_DEFAULT,
)


# ---------------------------------------------------------------------------
# Helper: full (normalised) log-probability of a diagonal Gaussian.
# ---------------------------------------------------------------------------
def _diag_gaussian_log_prob(x, mu, var):
    """
    Log-probability of x under N(mu, diag(var)).

    Args:
        x:   [..., D]
        mu:  [..., D]  (broadcastable with x)
        var: [..., D]  (broadcastable with x, must be > 0)

    Returns:
        log_prob: [...]  (scalar per leading index)
    """
    D = x.shape[-1]
    log_norm = -0.5 * D * math.log(2 * math.pi) - 0.5 * torch.log(var).sum(dim=-1)
    mahal = -0.5 * ((x - mu) ** 2 / var).sum(dim=-1)
    return log_norm + mahal


class UR5_VariantI_Cost:
    """
    Local per-step Jensen-Shannon Divergence tracking cost.

    Identical interface and structure to UR5_VariantH_Cost -- the only
    difference is the divergence measure used at each timestep.

    Args:
        checkpoint_path: path to ur5_denoising_theta_*.pt
        q_ref:           [N_traj+1, 6] reference positions
        dq_ref:          [N_traj+1, 6] reference velocities
        T_control:       trajectory duration (s)
        alpha:           weight on chance-constraint slack
        epsilon:         allowed violation probability
        weighting:       'uniform' (default) | 'linear' | 'quadratic'
        sigma_p_q:       per-position-dim sigma (default: GFN-trained 0.10)
        sigma_p_dq:      per-velocity-dim sigma (default: GFN-trained 0.50)
        n_p_samples:     number of samples from p for estimating KL(p||M).
                         More samples = less variance in gradient, but slower.
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
                 n_p_samples=50,
                 q_min=UR5_Q_MIN_DEFAULT,
                 q_max=UR5_Q_MAX_DEFAULT,
                 num_ref_samples=512,
                 dtype=torch.float64,
                 device=torch.device('cpu')):
        assert weighting in ('uniform', 'linear', 'quadratic', 'none'), \
            f"Unknown weighting '{weighting}'."

        self.alpha       = alpha
        self.epsilon     = epsilon
        self.weighting   = weighting
        self.n_p_samples = n_p_samples
        self.q_min       = q_min
        self.q_max       = q_max
        self.dtype       = dtype
        self.device      = device

        self.gfn_prior = UR5GFNPrior(
            checkpoint_path=checkpoint_path,
            q_ref=q_ref,
            dq_ref=dq_ref,
            T_control=T_control,
            num_ref_samples=num_ref_samples,
            dtype=dtype, device=device,
        )

        # Sigma for the reference Gaussian p = N(mu_ref, diag(sigma_p^2)).
        sigma = self.gfn_prior.sigma.clone()
        if sigma_p_q is not None:
            sigma[:6] = sigma_p_q
        if sigma_p_dq is not None:
            sigma[6:] = sigma_p_dq
        self.sigma_p    = sigma                     # [12]
        self.sigma_p_sq = self.sigma_p ** 2         # [12]

        # Logging buffers
        self.last_jsd_per_step   = None
        self.last_kl_q_m         = None
        self.last_kl_p_m         = None
        self.last_slack_per_step = None
        self.last_weights        = None
        self.last_mu_p_traj      = None

        print(f"[UR5_VariantI_Cost] LOCAL per-step JSD "
              f"(time-varying target) | alpha={alpha} epsilon={epsilon} "
              f"weighting='{weighting}' n_p_samples={n_p_samples}")
        print(f"[UR5_VariantI_Cost] sigma_p (positions, rad):  "
              f"{self.sigma_p[:6].tolist()}")
        print(f"[UR5_VariantI_Cost] sigma_p (velocities, rad/s): "
              f"{self.sigma_p[6:].tolist()}")

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

        # Sanitise extreme particles
        states_sanitized = torch.nan_to_num(states_sequence,
                                            nan=0.0,
                                            posinf=1e3, neginf=-1e3)
        states_sanitized = torch.clamp(states_sanitized, min=-50.0, max=50.0)

        # =========================================================
        # 1) Per-step time-varying target mu_ref(t_k) and p-variance
        # =========================================================
        mu_p_traj = self.gfn_prior.get_mu_at_steps(
            N_h, dtype=dtype, device=device)                  # [N_h+1, 12]
        var_p = self.sigma_p_sq.to(dtype=dtype, device=device)  # [12]

        # =========================================================
        # 2) Fit diagonal Gaussian q_k to particles at each step
        # =========================================================
        mu_q  = states_sanitized.mean(dim=1)                   # [N_h+1, 12]
        var_q = states_sanitized.var(dim=1) + 1e-8             # [N_h+1, 12]

        # =========================================================
        # 3) JSD(q_k || p_k) via Monte Carlo
        #
        #    JSD = 0.5 * KL(q || M) + 0.5 * KL(p || M)
        #    M   = 0.5 * q + 0.5 * p   (mixture)
        #
        #    KL(q || M) = E_q[ log q(x) - log M(x) ]
        #               ~ (1/P) sum_i [ log q(x_i) - log M(x_i) ]
        #
        #    KL(p || M) = E_p[ log p(x) - log M(x) ]
        #               ~ (1/S) sum_j [ log p(z_j) - log M(z_j) ]
        #    where z_j ~ p_k
        # =========================================================

        # --- Term 1: KL(q || M) from rollout particles ---
        # log q(x_i):  [N_h+1, P]
        log_q_at_x = _diag_gaussian_log_prob(
            states_sanitized,                      # [N_h+1, P, 12]
            mu_q.unsqueeze(1),                     # [N_h+1, 1, 12]
            var_q.unsqueeze(1),                    # [N_h+1, 1, 12]
        )
        # log p(x_i):  [N_h+1, P]
        log_p_at_x = _diag_gaussian_log_prob(
            states_sanitized,                      # [N_h+1, P, 12]
            mu_p_traj.unsqueeze(1),                # [N_h+1, 1, 12]
            var_p.unsqueeze(0).unsqueeze(0)         # [1, 1, 12]
              .expand(N_h_plus_1, P, D),
        )
        # log M(x_i) = log(0.5*q(x_i) + 0.5*p(x_i))
        #            = logsumexp([log_q, log_p], dim=-1) - ln(2)
        log_m_at_x = (torch.logsumexp(
            torch.stack([log_q_at_x, log_p_at_x], dim=-1),
            dim=-1) - math.log(2))                             # [N_h+1, P]

        kl_q_m = (log_q_at_x - log_m_at_x).mean(dim=1)       # [N_h+1]

        # --- Term 2: KL(p || M) from p-samples ---
        S = self.n_p_samples
        # z_j ~ N(mu_p, diag(sigma_p^2))
        noise = torch.randn(N_h_plus_1, S, D, dtype=dtype, device=device)
        z = mu_p_traj.unsqueeze(1) + self.sigma_p.to(
            dtype=dtype, device=device) * noise                # [N_h+1, S, 12]

        # log p(z_j):  [N_h+1, S]
        log_p_at_z = _diag_gaussian_log_prob(
            z,
            mu_p_traj.unsqueeze(1),
            var_p.unsqueeze(0).unsqueeze(0).expand(N_h_plus_1, S, D),
        )
        # log q(z_j):  [N_h+1, S]
        log_q_at_z = _diag_gaussian_log_prob(
            z,
            mu_q.unsqueeze(1),
            var_q.unsqueeze(1).expand(N_h_plus_1, S, D),
        )
        # log M(z_j):  [N_h+1, S]
        log_m_at_z = (torch.logsumexp(
            torch.stack([log_q_at_z, log_p_at_z], dim=-1),
            dim=-1) - math.log(2))

        kl_p_m = (log_p_at_z - log_m_at_z).mean(dim=1)       # [N_h+1]

        # --- JSD per step ---
        jsd_per_step = 0.5 * kl_q_m + 0.5 * kl_p_m           # [N_h+1]

        # Clamp to [0, ln2] -- numerical noise can push slightly negative
        jsd_per_step = torch.clamp(jsd_per_step, min=0.0, max=math.log(2) + 0.01)

        # =========================================================
        # 4) Time weighting -- DEFAULT IS UNIFORM
        # =========================================================
        t = torch.arange(N_h_plus_1, dtype=dtype, device=device) / N_h
        if self.weighting == 'uniform' or self.weighting == 'none':
            weights = torch.ones_like(t)
        elif self.weighting == 'linear':
            weights = t
        elif self.weighting == 'quadratic':
            weights = t ** 2
        weighted_jsd_per_step = weights * jsd_per_step

        # =========================================================
        # 5) Chance-constraint slack on joint positions
        # =========================================================
        slack_per_step, _ = ur5_joint_total_slack(
            states_sanitized,
            q_min=self.q_min, q_max=self.q_max,
            epsilon=self.epsilon,
        )

        # =========================================================
        # 6) Total cost per step
        # =========================================================
        cost_per_step = weighted_jsd_per_step + self.alpha * slack_per_step
        cost_per_step = torch.nan_to_num(cost_per_step,
                                         nan=1e8, posinf=1e8, neginf=0.0)
        cost_per_step = torch.clamp(cost_per_step, max=1e8)

        # Logging (detached)
        self.last_jsd_per_step   = jsd_per_step.detach()
        self.last_kl_q_m         = kl_q_m.detach()
        self.last_kl_p_m         = kl_p_m.detach()
        self.last_slack_per_step = slack_per_step.detach()
        self.last_weights        = weights.detach()
        self.last_mu_p_traj      = mu_p_traj.detach()

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
