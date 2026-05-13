import torch
from policy_learning.gfn_prior import GFNPrior
from policy_learning.kl_cost import gaussian_moments_from_particles, reverse_kl_gaussian_diag
from policy_learning.chance_constraint import cartpole_total_slack


class VariantB_Cost:
    """
    Forward KL toward GFN prior + soft chance constraints.

    Implements:
        L = KL_total(q_GP || p_GFN) + alpha * slack_total

    The interface matches Cart_pole_cost: an instance has a
    cost_function(states, inputs, trial_index) method that returns
    a per-timestep cost tensor [num_instants].
    """

    def __init__(self,
                 alpha=10.0,
                 epsilon=0.05,
                 weighting='quadratic',
                 position_bound=2.4,
                 angle_bound=0.2094,
                 dtype=torch.float64,
                 device=torch.device('cpu')):
        self.alpha = alpha
        self.epsilon = epsilon
        self.weighting = weighting
        self.position_bound = position_bound
        self.angle_bound = angle_bound
        self.dtype = dtype
        self.device = device

        self.gfn_prior = GFNPrior(dtype=dtype, device=device)

        # Logging
        self.last_kl_per_step = None
        self.last_slack_per_step = None

    def cost_function(self, states_sequence, inputs_sequence, trial_index):
        """
        Args:
            states_sequence: [num_instants, num_particles, 4]
            inputs_sequence: [num_instants, num_particles, 1]
            trial_index: int (unused but required by MC-PILCO interface)

        Returns:
            cost_per_step: [num_instants]
        """
        # Forward KL: KL(q_GP || p_GFN)
        # This means MC-PILCO's particle distribution is minimized
        # toward the GFN prior's target Gaussian.
        from policy_learning.kl_cost import gaussian_moments_from_particles

        # Get GP particle moments
        mu_q, Sigma_q_diag = gaussian_moments_from_particles(states_sequence)
        # mu_q: [num_instants, 4], Sigma_q_diag: [num_instants, 4]

        # Get GFN target (time-independent for MVP)
        mu_p = self.gfn_prior.mu_p                    # [4]
        Sigma_p_diag = self.gfn_prior.Sigma_p_diag    # [4]

        # Compute forward KL per timestep
        from policy_learning.kl_cost import reverse_kl_gaussian_diag
        # Note: we call it "reverse_kl" but swap the arguments
        # reverse_kl(mu_p, Sp, mu_q, Sq) = KL(p||q)
        # So reverse_kl(mu_q, Sq, mu_p, Sp) = KL(q||p) — forward KL
        N_h_plus_1 = states_sequence.shape[0]
        N_h = N_h_plus_1 - 1

        kl_steps = []
        for k in range(N_h_plus_1):
            kl_k = reverse_kl_gaussian_diag(
                mu_q[k], Sigma_q_diag[k],  # q = MC-PILCO particles
                mu_p, Sigma_p_diag          # p = GFN prior (fixed target)
            )
            kl_steps.append(kl_k)
        kl_per_step = torch.stack(kl_steps)   # [N_h+1]

        # Time weighting (quadratic ramp emphasizes end of horizon)
        t = torch.arange(N_h_plus_1, dtype=states_sequence.dtype,
                         device=states_sequence.device) / N_h
        if self.weighting == 'quadratic':
            weights = t ** 2
        elif self.weighting == 'linear':
            weights = t
        else:
            weights = torch.ones_like(t)

        kl_total = (weights * kl_per_step).sum()

        # Chance constraint slack term
        slack_per_step, slack_total = cartpole_total_slack(
            states_sequence,
            position_bound=self.position_bound,
            angle_bound=self.angle_bound,
            epsilon=self.epsilon
        )

        # Per-step combined cost (matches Cart_pole_cost output shape)
        cost_per_step = kl_per_step + self.alpha * slack_per_step

        # Store for logging
        self.last_kl_per_step = kl_per_step.detach()
        self.last_slack_per_step = slack_per_step.detach()

        return cost_per_step

    def __call__(self, states_sequence, inputs_sequence, trial_index):
        """
        MC-PILCO calls cost_function(states, inputs, trial_index)
        directly on the instance. This makes the object callable
        and returns the (mean_cost, std_cost) tuple that
        MC-PILCO's reinforce_policy expects.
        """
        cost_per_step = self.cost_function(
            states_sequence, inputs_sequence, trial_index
        )
        # MC-PILCO expects (mean, std). Our cost_per_step is
        # already aggregated over particles, so std is 0.
        mean_cost = cost_per_step.sum()
        std_cost = torch.tensor(0.0, dtype=cost_per_step.dtype, device=cost_per_step.device)
        return mean_cost, std_cost
