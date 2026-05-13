import torch
import math


def standard_normal_inverse_cdf(p):
    """Inverse standard-normal CDF (Phi^-1). Uses torch.erfinv."""
    return math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)


def chance_constraint_slack(mu, Sigma_diag, h, b, epsilon):
    """
    Compute the soft slack for a single linear chance constraint:
        Pr( h^T x <= b ) >= 1 - epsilon
    Reformulated (eq. 9) as:
        slack = ReLU( h^T mu - b + Phi^-1(1 - epsilon) * sqrt(h^T Sigma h) )

    Args:
        mu:         [..., state_dim]     particle mean per timestep
        Sigma_diag: [..., state_dim]     particle variance diagonal
        h:          [state_dim]          linear coefficient
        b:          scalar               bound
        epsilon:    scalar in (0, 1)     violation probability (e.g. 0.05)

    Returns:
        slack: [...]   (>=0, zero when constraint comfortably satisfied)
    """
    # Phi^-1(1 - epsilon) is a fixed scalar for given epsilon
    eps_t = torch.tensor(1.0 - epsilon, dtype=mu.dtype, device=mu.device)
    phi_inv = standard_normal_inverse_cdf(eps_t)

    # Mean and variance of h^T x
    mean_lin = (h * mu).sum(dim=-1)                        # h^T mu
    var_lin = (h ** 2 * Sigma_diag).sum(dim=-1)            # h^T diag(Sigma) h
    std_lin = torch.sqrt(var_lin + 1e-12)

    raw = mean_lin - b + phi_inv * std_lin
    return torch.relu(raw)


def cartpole_total_slack(states_sequence,
                        position_bound=2.4,
                        angle_bound=0.2094,
                        epsilon=0.05):
    """
    Sum of all 4 chance-constraint slacks across all timesteps for CartPole.

    State layout: [position, velocity, angle, angular_velocity]

    Args:
        states_sequence: [num_instants, num_particles, state_dim]

    Returns:
        slack_per_step: [num_instants]   — for logging
        slack_total:    scalar           — sum over time and constraints
    """
    from policy_learning.kl_cost import gaussian_moments_from_particles
    mu, Sigma_diag = gaussian_moments_from_particles(states_sequence)
    # mu, Sigma_diag both: [num_instants, state_dim]

    state_dim = 4
    device = mu.device
    dtype = mu.dtype

    # 4 constraints: +x, -x, +theta, -theta
    h_pos_pos   = torch.tensor([ 1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device)
    h_pos_neg   = torch.tensor([-1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device)
    h_angle_pos = torch.tensor([ 0.0, 0.0, 1.0, 0.0], dtype=dtype, device=device)
    h_angle_neg = torch.tensor([ 0.0, 0.0,-1.0, 0.0], dtype=dtype, device=device)

    s1 = chance_constraint_slack(mu, Sigma_diag, h_pos_pos,   position_bound, epsilon)
    s2 = chance_constraint_slack(mu, Sigma_diag, h_pos_neg,   position_bound, epsilon)
    s3 = chance_constraint_slack(mu, Sigma_diag, h_angle_pos, angle_bound,    epsilon)
    s4 = chance_constraint_slack(mu, Sigma_diag, h_angle_neg, angle_bound,    epsilon)

    slack_per_step = s1 + s2 + s3 + s4                # [num_instants]
    slack_total = slack_per_step.sum()
    return slack_per_step, slack_total
