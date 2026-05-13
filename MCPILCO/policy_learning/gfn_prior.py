import torch


class GFNPrior:
    """
    Frozen Gaussian prior over CartPole state space.

    For the MVP this is the time-independent target distribution that
    the Phase 1 GFlowNet was trained to match:
        p_GFN(x) = N(0, Sigma_target)

    Sigma_target = diag(sigma^2) where sigma = [0.5, 0.5, 0.1, 0.1]
    corresponding to [position, velocity, angle, angular_velocity].
    """

    def __init__(self, dtype=torch.float64, device=torch.device('cpu')):
        self.dtype = dtype
        self.device = device
        self.state_dim = 4

        # Target sigma values per dimension
        self.sigma = torch.tensor([0.5, 0.5, 0.1, 0.1],
                                  dtype=dtype, device=device)

        # Target mean (upright equilibrium in GFN frame)
        self.mu_p = torch.zeros(self.state_dim, dtype=dtype, device=device)

        # Target covariance (diagonal)
        self.Sigma_p_diag = self.sigma ** 2   # [state_dim]
        self.Sigma_p = torch.diag(self.Sigma_p_diag)   # [state_dim, state_dim]

        # Precompute logdet for KL formula
        self.logdet_Sigma_p = torch.log(self.Sigma_p_diag).sum()

    def get_target_at_step(self, k, N_h):
        """
        Time-rescaled query of the prior.
        For the MVP, the prior is time-independent so this just returns
        (mu_p, Sigma_p) regardless of k. The method exists so we can
        later swap in a time-conditioned GFN without changing callers.

        Args:
            k: physical timestep index in {0, ..., N_h}
            N_h: total horizon steps

        Returns:
            mu_p:    [state_dim]
            Sigma_p: [state_dim, state_dim] diagonal
        """
        t_diffusion = k / N_h   # ∈ [0, 1]  — currently unused
        return self.mu_p, self.Sigma_p
