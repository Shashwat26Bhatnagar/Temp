import math
import pathlib
import sys

import torch


def _add_gfn_to_path():
    """Make the Phase-1 GflowNet package importable.

    Search order:
      1. GFN_PATH env var
      2. Sibling layout:  <repo_root>/GflowNet/gfn-diffusion/energy_sampling
      3. Documents layout: C:/Users/Shashwat/Documents/GflowNet/...
    """
    import os

    candidates = []
    env = os.environ.get('GFN_PATH')
    if env:
        candidates.append(pathlib.Path(env))

    here = pathlib.Path(__file__).resolve()
    # gfn_prior.py -> policy_learning -> MCPILCO -> repo_root
    repo_root = here.parent.parent.parent
    candidates.append(repo_root / 'GflowNet' / 'gfn-diffusion' / 'energy_sampling')

    candidates.append(pathlib.Path.home() / 'Documents' / 'GflowNet'
                      / 'gfn-diffusion' / 'energy_sampling')

    for p in candidates:
        if (p / 'models' / 'gfn.py').exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return p

    raise FileNotFoundError(
        "Could not locate the GflowNet energy_sampling package. "
        "Set the GFN_PATH environment variable to the directory that "
        "contains models/gfn.py (typically .../GflowNet/gfn-diffusion/energy_sampling)."
    )


_GFN_DIR = _add_gfn_to_path()

from models.gfn import GFN as _GFNNet


def _build_gfn(device):
    """Reconstruct the GFN architecture used to train cartpole_denoising_theta_final.pt.

    These hyperparameters MUST match the ones recorded in the checkpoint's
    'args' dict (batch_size=512, t_scale=5.0, learn_pb=True, learned_variance=True,
    clipping=True, etc.).
    """
    return _GFNNet(
        dim=4, s_emb_dim=64, hidden_dim=64,
        harmonics_dim=64, t_dim=64, log_var_range=4.0,
        t_scale=5.0, learned_variance=True, partial_energy=False,
        clipping=True, lgv_clip=1e2, gfn_clip=1e4,
        pb_scale_range=0.1, learn_pb=True, device=device,
        langevin_scaling_per_dimension=False,
    )


class GFNPrior:
    """
    Frozen prior derived from the Phase-1 trained GFlowNet.

    The Phase-1 GFN was trained to sample from
        p(x) propto exp( -0.5 * (x - mu)^T diag(sigma^-2) (x - mu) )
    with
        mu    = [0, 0, pi, 0]   (upright in MC-PILCO frame)
        sigma = [0.5, 0.5, 0.1, 0.1]

    This class:
      * Loads the trained checkpoint and freezes its parameters.
      * Pre-samples a frozen reference set from the trained model (used for
        MMD logging and as the empirical target the policy is matched to).
      * Provides log_density(x), the (unnormalised) log p_target(x) used
        as a per-particle cross-entropy signal in VariantB_Cost.

    log_density uses the analytical energy that the GFN was trained on
    (mathematically identical to the trained model's target up to log Z when
    Phase-1 has converged — verified at init via MMD).
    """

    def __init__(self,
                 checkpoint_path,
                 num_ref_samples=512,
                 dtype=torch.float64,
                 device=torch.device('cpu'),
                 verify_mmd=True):
        self.dtype = dtype
        self.device = device
        self.state_dim = 4

        # Target parameters — these MUST match the Phase-1 CartPoleEnergy
        self.sigma = torch.tensor([0.5, 0.5, 0.1, 0.1],
                                  dtype=dtype, device=device)
        self.mu_p = torch.tensor([0.0, 0.0, math.pi, 0.0],
                                 dtype=dtype, device=device)
        self.Sigma_p_diag = self.sigma ** 2
        self.Sigma_p = torch.diag(self.Sigma_p_diag)

        # Load trained network and freeze
        ckpt_path = pathlib.Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"GFN checkpoint not found: {ckpt_path}. "
                "Run the Phase-1 training first (see GflowNet/.../train.py)."
            )
        self.gfn_model = _build_gfn(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        self.gfn_model.load_state_dict(state_dict)
        self.gfn_model.eval()
        for p in self.gfn_model.parameters():
            p.requires_grad_(False)

        # Pre-sample frozen reference set from the trained network.
        # These are used (a) for MMD validation at init and (b) as the
        # empirical target the policy can be matched against.
        self.ref_samples = self._draw(num_ref_samples).detach()

        if verify_mmd:
            self._mmd_init = self._mmd_against_target(self.ref_samples)
            print(f"[GFNPrior] Loaded {ckpt_path.name}; "
                  f"MMD(GFN samples vs analytical N(mu,Sigma)) = "
                  f"{self._mmd_init:.6f}")

    def _draw(self, n):
        """Internal: draw n samples from the trained GFN (no grad)."""
        # The GFN's log_reward callback must be float32 (matches the
        # network's parameters and the energy class's tensors).
        import math as _math
        mu32 = torch.tensor([0.0, 0.0, _math.pi, 0.0],
                            dtype=torch.float32, device=self.device)
        sigma32 = torch.tensor([0.5, 0.5, 0.1, 0.1],
                               dtype=torch.float32, device=self.device)

        def log_reward(x, condition=None):
            return -0.5 * ((x - mu32) ** 2 / sigma32 ** 2).sum(dim=-1)

        with torch.no_grad():
            init_state = torch.zeros(n, self.state_dim,
                                     dtype=torch.float32,
                                     device=self.device)
            states, _, _, _ = self.gfn_model.get_trajectory_fwd(
                init_state, None, log_reward)
            terminal = states[:, -1, :].detach()
        return terminal.to(dtype=self.dtype)

    def sample(self, n):
        """Draw fresh samples from the trained network."""
        return self._draw(n)

    def log_density(self, x):
        """
        Unnormalised log p_target(x).

        log p_target(x) = -0.5 * (x - mu)^T diag(sigma^-2) (x - mu)
                          + const

        We omit the additive constant (the normaliser) because it does
        not affect optimisation.

        Args:
            x: tensor of shape [..., 4], any dtype.
        Returns:
            log_density: tensor of shape [...] in x's dtype.
        """
        mu = self.mu_p.to(dtype=x.dtype, device=x.device)
        sigma = self.sigma.to(dtype=x.dtype, device=x.device)
        return -0.5 * ((x - mu) ** 2 / sigma ** 2).sum(dim=-1)

    def _mmd_against_target(self, samples, n_target=None, kernel_sigma=1.0):
        """Sanity check: MMD between trained-GFN samples and an analytical
        N(mu, diag(sigma^2)) drawn fresh."""
        if n_target is None:
            n_target = samples.shape[0]
        gt = self.mu_p + torch.randn(n_target, self.state_dim,
                                     dtype=self.dtype,
                                     device=self.device) * self.sigma
        xx = torch.cdist(samples, samples, p=2.0) ** 2
        yy = torch.cdist(gt, gt, p=2.0) ** 2
        xy = torch.cdist(samples, gt, p=2.0) ** 2
        s2 = 2.0 * (kernel_sigma ** 2)
        kxx = torch.exp(-xx / s2).mean()
        kyy = torch.exp(-yy / s2).mean()
        kxy = torch.exp(-xy / s2).mean()
        return float((kxx + kyy - 2 * kxy).item())

    def get_target_at_step(self, k, N_h):
        """Time-rescaled query — kept for backwards compatibility. The
        prior is currently time-independent (frozen terminal target)."""
        return self.mu_p, self.Sigma_p
