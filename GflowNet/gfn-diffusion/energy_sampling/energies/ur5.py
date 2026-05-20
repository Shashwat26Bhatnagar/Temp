import torch


class UR5Energy:
    """
    Target distribution N(mu, diag(sigma^2)) over UR5 states
    x = [q1..q6, dq1..dq6]   (12-D)

    The target mean mu_final is the end-state of one full circular
    end-effector trajectory (radius 0.15 m, centre (-0.6, 0, 0.4),
    duration 4 s) starting from the UR5 'ready' pose.  After a full
    circle the arm is back at the start, so

        mu_final = [q_ref(4s), dq_ref(4s)]

    Numerical values come from
        MCPILCO/simulation_class/ur5_trajectory.generate_circle_trajectory()
        with center=(-0.6, 0, 0.4), radius=0.15, duration=4, dt=0.02
    and are cached here as constants so this file does not depend on
    MuJoCo at training time.  Regenerate with the trajectory script if
    the circle geometry changes.
    """

    def __init__(self, device):
        self.device = device
        self.data_ndim = 12

        # --- Target mean: end of one closed-loop circular trajectory ----
        # First 6 values: joint positions q_ref(4s)
        # Last  6 values: joint velocities dq_ref(4s)
        q_final = torch.tensor(
            [0.1884, -1.9495,  2.0472, -1.7108, -1.6774,  0.0000],
            device=device,
        )
        dq_final = torch.tensor(
            [-0.5045, -0.0036,  0.0162,  0.0000,  0.0921,  0.0000],
            device=device,
        )
        self.mu = torch.cat([q_final, dq_final])              # [12]

        # --- Target std: tight positions, looser velocities -------------
        # Convention follows CartPoleEnergy: position dims small std,
        # velocity dims larger std.
        sigma_q  = torch.full((6,), 0.10, device=device)      # 0.1 rad
        sigma_dq = torch.full((6,), 0.50, device=device)      # 0.5 rad/s
        self.sigma = torch.cat([sigma_q, sigma_dq])           # [12]

    def log_reward(self, x, condition=None):
        # log R(x) = -0.5 * (x - mu)^T diag(sigma^-2) (x - mu)
        return -0.5 * ((x - self.mu) ** 2 / self.sigma ** 2).sum(dim=-1)

    def sample(self, n):
        raise NotImplementedError("UR5Energy has no ground truth sampler")
