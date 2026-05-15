import math

import torch

class CartPoleEnergy:
    """
    Target distribution N(mu, diag(sigma^2)) over CartPole states
    [pos, vel, angle, ang_vel] using MC-PILCO's convention where
    theta = pi is the unstable upright equilibrium (theta = 0 is
    the stable hanging-down equilibrium, see
    MCPILCO/simulation_class/ode_systems.py).
    """

    def __init__(self, device):
        self.device = device
        self.data_ndim = 4
        self.sigma = torch.tensor([0.5, 0.5, 0.1, 0.1], device=device)
        self.mu = torch.tensor([0.0, 0.0, math.pi, 0.0], device=device)

    def log_reward(self, x, condition=None):
        # log R(x) = -0.5 * (x - mu)^T diag(sigma^-2) (x - mu)
        return -0.5 * ((x - self.mu) ** 2 / self.sigma ** 2).sum(dim=-1)

    def sample(self, n):
        raise NotImplementedError("CartPoleEnergy has no ground truth sampler")
