"""
Chance constraints for UR5 trajectory tracking.

For each joint i in {0..5}, two constraints:
    Pr( q_i <=  q_max[i] ) >= 1 - epsilon
    Pr( q_i >= -q_min[i] ) >= 1 - epsilon

Total: 12 linear constraints (2 per joint x 6 joints).

Re-uses chance_constraint_slack() from chance_constraint.py (Gaussian CDF
reformulation, eq. 9 of the chance-constrained PILCO paper).
"""

import torch

from policy_learning.kl_cost import gaussian_moments_from_particles
from policy_learning.chance_constraint import chance_constraint_slack


# Default UR5 joint limits (rad). Matches the XML model_jnt_range.
UR5_Q_MIN_DEFAULT = (-3.14159, -3.14159, -3.14159, -3.14159, -3.14159, -3.14159)
UR5_Q_MAX_DEFAULT = ( 3.14159,  3.14159,  3.14159,  3.14159,  3.14159,  3.14159)


def ur5_joint_total_slack(states_sequence,
                          q_min=UR5_Q_MIN_DEFAULT,
                          q_max=UR5_Q_MAX_DEFAULT,
                          epsilon=0.10):
    """
    Sum of all 12 joint-position chance-constraint slacks across all timesteps.

    State layout (12-D): [q1..q6, dq1..dq6]
    Position constraints: q_min[i] <= q_i <= q_max[i] for i in 0..5

    Args:
        states_sequence: [num_instants, num_particles, 12]
        q_min, q_max:    sequences of length 6 (rad)
        epsilon:         allowed violation probability (e.g. 0.10)

    Returns:
        slack_per_step: [num_instants]
        slack_total:    scalar
    """
    mu, Sigma_diag = gaussian_moments_from_particles(states_sequence)
    # mu, Sigma_diag both: [num_instants, 12]

    device = mu.device
    dtype  = mu.dtype

    slack_per_step = torch.zeros(mu.shape[0], dtype=dtype, device=device)

    for i in range(6):
        # Upper bound:  q_i <=  q_max[i]
        h_up = torch.zeros(12, dtype=dtype, device=device)
        h_up[i] = 1.0
        slack_per_step = slack_per_step + chance_constraint_slack(
            mu, Sigma_diag, h_up, float(q_max[i]), epsilon)

        # Lower bound:  -q_i <= -q_min[i]   (i.e., q_i >= q_min[i])
        h_lo = torch.zeros(12, dtype=dtype, device=device)
        h_lo[i] = -1.0
        slack_per_step = slack_per_step + chance_constraint_slack(
            mu, Sigma_diag, h_lo, float(-q_min[i]), epsilon)

    slack_total = slack_per_step.sum()
    return slack_per_step, slack_total
