import torch

class CartPoleEnergy:
    def __init__(self, device):
        self.device = device
        self.data_ndim = 4
        self.sigma = torch.tensor([0.5, 0.5, 0.1, 0.1], device=device)

    def log_reward(self, x, condition=None):
        # x: [batch, 4] — CartPole state [pos, vel, angle, ang_vel]
        # R(x1) = exp(-E(x)), E(x) = 0.5 * x^T Sigma^-1 x
        # log R(x1) = -E(x)
        return -0.5 * (x ** 2 / self.sigma ** 2).sum(dim=-1)

    def sample(self, n):
        raise NotImplementedError("CartPoleEnergy has no ground truth sampler")
