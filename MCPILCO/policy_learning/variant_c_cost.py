import torch

from policy_learning.gfn_prior import GFNPrior
from policy_learning.chance_constraint import cartpole_total_slack
from policy_learning.kl_cost import (
    gaussian_moments_from_particles,
    reverse_kl_gaussian_diag,
)


class VariantC_Cost:
    """
    Variant C: MC-PILCO cost driven by the REVERSE Gaussian KL between the
    GFN target and the particle distribution, plus probabilistic safety.

        L = sum_t  w_t * KL(p_GFN || q_t)  +  alpha * sum_t  slack_t

    where
        q_t     = diagonal Gaussian fitted to particle moments at step t
        p_GFN   = frozen Phase-1 GFN target (analytical Gaussian)
        w_t     = (t / N_h)^2  (quadratic time weighting)
        slack_t = soft chance-constraint slack at step t

    Forward vs. reverse KL — why this variant exists
    ------------------------------------------------
    Variant B minimises KL(q || p)  (forward KL, mode-seeking).
      * sigma_p (fixed) appears in the denominator -> numerically stable.
      * Pushes q toward p; if p is multi-modal, q collapses to one mode.

    Variant C minimises KL(p || q)  (reverse KL, mass-covering).
      * sigma_q (particle variance) appears in the denominator ->
        if particles collapse, KL blows up.  This is intentional: the
        gradient penalises particle clouds that become too narrow to
        cover the target, which yields more exploratory policies.
      * For a single-mode unimodal target (Phase-1 Gaussian) reverse KL
        does NOT change the optimum (both KLs are minimised when q = p),
        but the gradient geometry differs and can lead the optimiser
        along a different path.

    Numerical safeguards
    --------------------
      * gaussian_moments_from_particles() floors the diagonal variance
        with +1e-8 before any division -> avoids inf/NaN when particles
        nearly collapse early in training.
      * Cost mode is forced to 'kl' here; cross-entropy is undefined for
        reverse direction without an explicit p sampler at every step.

    Differentiability
    -----------------
    The reparameterisation trick in MC-PILCO's GP rollout makes the
    per-step particle moments (mu_q[t], Sigma_q[t]) differentiable in the
    policy parameters, so reverse_kl_gaussian_diag(mu_p, Sigma_p_diag,
    mu_q[t], Sigma_q_diag[t]) yields a usable gradient.
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

        # Per-call logging buffers (detached, scalar-friendly)
        self.last_divergence_per_step = None
        self.last_slack_per_step = None
        self.last_weights = None

        print(f"[VariantC_Cost] cost_mode = 'reverse_kl', "
              f"alpha = {alpha}, epsilon = {epsilon}, weighting = '{weighting}'")
        print(f"[VariantC_Cost] position_bound = {position_bound} m, "
              f"angle_bound = {angle_bound} rad "
              f"({angle_bound * 180 / 3.14159:.2f} deg around theta=pi)")

    # ------------------------------------------------------------------ #
    # Cost API expected by MC-PILCO                                      #
    # ------------------------------------------------------------------ #
    def cost_function(self, states_sequence, inputs_sequence, trial_index):
        """
        Args:
            states_sequence: [N_h+1, num_particles, 4]
            inputs_sequence: [N_h+1, num_particles, 1]  (unused)
            trial_index:     int (unused, MC-PILCO interface)

        Returns:
            cost_per_step: [N_h+1]   per-timestep cost (sum gives total)
        """
        N_h_plus_1, P, D = states_sequence.shape
        assert D == 4, f"Expected state_dim=4, got {D}"
        N_h = N_h_plus_1 - 1

        # ============================================================
        # 1) Per-step REVERSE KL from particles' Gaussian fit to target
        # ============================================================
        mu_q, Sigma_q_diag = gaussian_moments_from_particles(states_sequence)

        mu_p = self.gfn_prior.mu_p.to(dtype=states_sequence.dtype,
                                      device=states_sequence.device)
        Sigma_p_diag = self.gfn_prior.Sigma_p_diag.to(
            dtype=states_sequence.dtype, device=states_sequence.device)

        # KL(p || q) — note argument order vs. Variant B
        divergence_per_step = reverse_kl_gaussian_diag(
            mu_p, Sigma_p_diag, mu_q, Sigma_q_diag)               # [N_h+1]

        # ============================================================
        # 2) Quadratic time weighting (emphasise end of horizon)
        # ============================================================
        t = torch.arange(N_h_plus_1,
                         dtype=states_sequence.dtype,
                         device=states_sequence.device) / N_h
        if self.weighting == 'quadratic':
            weights = t ** 2
        elif self.weighting == 'linear':
            weights = t
        else:  # 'none' or 'uniform'
            weights = torch.ones_like(t)

        weighted_div_per_step = weights * divergence_per_step    # [N_h+1]

        # ============================================================
        # 3) Soft chance-constraint slack (relaxed bounds + epsilon)
        # ============================================================
        slack_per_step, _ = cartpole_total_slack(
            states_sequence,
            position_bound=self.position_bound,
            angle_bound=self.angle_bound,
            epsilon=self.epsilon,
        )                                                         # [N_h+1]

        # ============================================================
        # 4) Total cost per step
        # ============================================================
        cost_per_step = weighted_div_per_step + self.alpha * slack_per_step

        # Logging (detached)
        self.last_divergence_per_step = divergence_per_step.detach()
        self.last_slack_per_step = slack_per_step.detach()
        self.last_weights = weights.detach()

        return cost_per_step

    def __call__(self, states_sequence, inputs_sequence, trial_index):
        """MC-PILCO calls the cost object as a function returning
        (mean_cost, std_cost). std is zero because particle averaging
        already happened inside cost_function via moment fitting."""
        cost_per_step = self.cost_function(states_sequence,
                                           inputs_sequence,
                                           trial_index)
        mean_cost = cost_per_step.sum()
        std_cost = torch.tensor(0.0,
                                dtype=cost_per_step.dtype,
                                device=cost_per_step.device)
        return mean_cost, std_cost
