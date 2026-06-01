"""
UR5 GFN prior for Phase-2 MC-PILCO.

Differences from `gfn_prior.py` (CartPole):
  * 12-D state x = [q1..q6, dq1..dq6]
  * Loads ur5_denoising_theta_*.pt with the GFN built at dim=12
  * Provides TIME-VARYING target via get_target_at_step(k, N_h):
        physical_time = (k / N_h) * T_control
        mu_p(t)       = [q_ref(physical_time), dq_ref(physical_time)]
        Sigma_p       = diag(sigma^2)  (constant; only the mean slides)

This is the "diffusion-time -> physical-time" conversion:
    diffusion progress   = k / N_h  in [0, 1]
    physical progress    = t / T_control in [0, 1]
both axes share the same normalised [0, 1] interval, so the policy at
step k is asked to match the reference trajectory at the matching point.
"""

import math
import pathlib
import sys

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Make the GflowNet package importable (same logic as gfn_prior.py)
# ---------------------------------------------------------------------------
def _add_gfn_to_path():
    import os
    candidates = []
    env = os.environ.get('GFN_PATH')
    if env:
        candidates.append(pathlib.Path(env))

    here = pathlib.Path(__file__).resolve()
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
        "Set the GFN_PATH environment variable.")


_GFN_DIR = _add_gfn_to_path()
from models.gfn import GFN as _GFNNet


def _build_gfn(device, dim=12):
    """Reconstruct the GFN architecture used to train ur5_denoising_theta_*.pt.

    Hyperparameters must match the training command:
        --T 100 --t_scale 5.0 --learned_variance --clipping
        --lgv_clip 100 --gfn_clip 10000 --learn_pb
        --pb_scale_range 0.1 --log_var_range 4.0
    """
    return _GFNNet(
        dim=dim, s_emb_dim=64, hidden_dim=64,
        harmonics_dim=64, t_dim=64, log_var_range=4.0,
        t_scale=5.0, learned_variance=True, partial_energy=False,
        clipping=True, lgv_clip=1e2, gfn_clip=1e4,
        pb_scale_range=0.1, learn_pb=True, device=device,
        langevin_scaling_per_dimension=False,
    )


class UR5GFNPrior:
    """
    Frozen 12-D Phase-1 prior with a time-varying mean for trajectory tracking.

    Constants (matching Phase-1 UR5Energy):
        Terminal target  mu_final = [q_ref(T_control), dq_ref(T_control)]
        Std              sigma    = [0.1 x 6 (positions), 0.5 x 6 (velocities)]
    """

    def __init__(self,
                 checkpoint_path,
                 q_ref,         # [N_traj+1, 6]   joint positions over [0, T_control]
                 dq_ref,        # [N_traj+1, 6]   joint velocities over [0, T_control]
                 T_control,     # scalar (s)
                 num_ref_samples=512,
                 dtype=torch.float64,
                 device=torch.device('cpu'),
                 verify_mmd=True):
        self.dtype     = dtype
        self.device    = device
        self.state_dim = 12
        self.T_control = float(T_control)

        # Per-dim std: tight for positions (0.1 rad) and looser for velocities
        # (0.5 rad/s) -- must match energies/ur5.py
        sigma_q  = torch.full((6,), 0.10, dtype=dtype, device=device)
        sigma_dq = torch.full((6,), 0.50, dtype=dtype, device=device)
        self.sigma        = torch.cat([sigma_q, sigma_dq])           # [12]
        self.Sigma_p_diag = self.sigma ** 2                          # [12]
        self.Sigma_p      = torch.diag(self.Sigma_p_diag)            # [12,12]

        # Terminal target -- used for verify_mmd and as default get-target
        q_final  = torch.as_tensor(q_ref[-1],  dtype=dtype, device=device)   # [6]
        dq_final = torch.as_tensor(dq_ref[-1], dtype=dtype, device=device)   # [6]
        self.mu_p_final = torch.cat([q_final, dq_final])             # [12]
        # For backwards compatibility with CartPole interface
        self.mu_p = self.mu_p_final

        # Cache full trajectory for time-varying queries
        self.q_ref_traj  = torch.as_tensor(q_ref,  dtype=dtype, device=device)  # [N+1, 6]
        self.dq_ref_traj = torch.as_tensor(dq_ref, dtype=dtype, device=device)  # [N+1, 6]
        self.N_traj      = self.q_ref_traj.shape[0] - 1
        self.dt_traj     = self.T_control / self.N_traj

        # ----- Load the Phase-1 GFN -----
        ckpt_path = pathlib.Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"UR5 GFN checkpoint not found: {ckpt_path}. "
                "Run Phase-1 training first.")

        self.gfn_model = _build_gfn(device, dim=self.state_dim)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        self.gfn_model.load_state_dict(state_dict)
        self.gfn_model.eval()
        for p in self.gfn_model.parameters():
            p.requires_grad_(False)

        # Pre-sample a frozen reference set (used for verify_mmd)
        self.ref_samples = self._draw(num_ref_samples).detach()

        if verify_mmd:
            self._mmd_init = self._mmd_against_target(self.ref_samples)
            print(f"[UR5GFNPrior] Loaded {ckpt_path.name}; "
                  f"MMD(GFN samples vs analytical N(mu_final, Sigma)) = "
                  f"{self._mmd_init:.6f}")
        print(f"[UR5GFNPrior] T_control = {self.T_control:.3f} s, "
              f"N_traj = {self.N_traj} reference waypoints, "
              f"dt_traj = {self.dt_traj:.4f} s")

    # ------------------------------------------------------------------ #
    # Internal: draw from GFN (analytical energy on terminal target)     #
    # ------------------------------------------------------------------ #
    def _draw(self, n):
        mu32    = self.mu_p_final.to(dtype=torch.float32).detach()
        sigma32 = self.sigma.to(dtype=torch.float32).detach()

        def log_reward(x, condition=None):
            return -0.5 * ((x - mu32) ** 2 / sigma32 ** 2).sum(dim=-1)

        with torch.no_grad():
            init_state = torch.zeros(n, self.state_dim,
                                     dtype=torch.float32, device=self.device)
            states, _, _, _ = self.gfn_model.get_trajectory_fwd(
                init_state, None, log_reward)
            terminal = states[:, -1, :].detach()
        return terminal.to(dtype=self.dtype)

    def sample(self, n):
        return self._draw(n)

    def get_diffusion_marginals(self, n_samples=512):
        """
        Sample full GFN diffusion trajectories and return per-diffusion-step
        empirical Gaussian marginals (mean + diagonal variance), WITHOUT
        discarding intermediate denoising steps. Used by UR5 Variant K.

        CAVEAT for UR5: the Phase-1 GFN was trained only on the TERMINAL
        target N(mu_p_final, Sigma_p) (the closed-circle endpoint). Its
        diffusion path therefore goes zeros -> terminal pose, it does NOT
        trace the q_ref(t) circle. So Variant K on UR5 is effectively a
        GOAL-REACHING curriculum (zeros -> terminal pose with growing
        tolerance), not circle tracking. For pointwise circle tracking use
        Variant H/J (which slide mu along q_ref).

        Returns:
            mu_per_step:  [T_gfn+1, 12]
            var_per_step: [T_gfn+1, 12]  (floored at 1e-6)
        """
        mu32 = self.mu_p_final.to(dtype=torch.float32).detach()
        sigma32 = self.sigma.to(dtype=torch.float32).detach()

        def log_reward(x, condition=None):
            return -0.5 * ((x - mu32) ** 2 / sigma32 ** 2).sum(dim=-1)

        with torch.no_grad():
            init_state = torch.zeros(n_samples, self.state_dim,
                                     dtype=torch.float32, device=self.device)
            states, _, _, _ = self.gfn_model.get_trajectory_fwd(
                init_state, None, log_reward)
            # states: [n_samples, T_gfn+1, 12]
        mu_per_step = states.mean(dim=0).to(dtype=self.dtype)
        var_per_step = states.var(dim=0).to(dtype=self.dtype)
        var_per_step = torch.clamp(var_per_step, min=1e-6)
        return mu_per_step, var_per_step

    # ------------------------------------------------------------------ #
    # Time-varying target -- the diffusion->physical time conversion     #
    # ------------------------------------------------------------------ #
    def get_mu_at_steps(self, N_h, dtype=None, device=None):
        """
        Return the time-varying target mean for ALL N_h+1 control steps.

        For each k in {0, 1, ..., N_h}:
            physical_time = (k / N_h) * T_control
            mu_p[k]       = [q_ref(physical_time), dq_ref(physical_time)]

        Linear interpolation is used between recorded waypoints.

        Returns:
            mu_p_traj: [N_h+1, 12]   time-varying target mean
        """
        dtype  = dtype  or self.dtype
        device = device or self.device

        # diffusion-time grid normalised to [0, 1]
        u = torch.arange(N_h + 1, dtype=dtype, device=device) / float(N_h)
        # map to trajectory index in float
        idx_float = u * self.N_traj
        idx_lo = torch.clamp(idx_float.floor().long(),
                             0, self.N_traj)
        idx_hi = torch.clamp(idx_lo + 1, 0, self.N_traj)
        frac   = (idx_float - idx_lo.to(dtype)).unsqueeze(-1)        # [N_h+1, 1]

        q_lo  = self.q_ref_traj.to(dtype=dtype, device=device)[idx_lo]   # [N_h+1, 6]
        q_hi  = self.q_ref_traj.to(dtype=dtype, device=device)[idx_hi]   # [N_h+1, 6]
        dq_lo = self.dq_ref_traj.to(dtype=dtype, device=device)[idx_lo]  # [N_h+1, 6]
        dq_hi = self.dq_ref_traj.to(dtype=dtype, device=device)[idx_hi]  # [N_h+1, 6]

        q_t  = q_lo  + frac * (q_hi  - q_lo)                           # [N_h+1, 6]
        dq_t = dq_lo + frac * (dq_hi - dq_lo)                          # [N_h+1, 6]
        mu_p_traj = torch.cat([q_t, dq_t], dim=-1)                     # [N_h+1, 12]
        return mu_p_traj

    def get_target_at_step(self, k, N_h):
        """Backward-compatible single-step query."""
        mu_traj = self.get_mu_at_steps(N_h)
        return mu_traj[k], self.Sigma_p

    # ------------------------------------------------------------------ #
    # log p_target(x) used by cross-entropy modes (terminal-only target) #
    # ------------------------------------------------------------------ #
    def log_density(self, x):
        mu = self.mu_p_final.to(dtype=x.dtype, device=x.device)
        sigma = self.sigma.to(dtype=x.dtype, device=x.device)
        return -0.5 * ((x - mu) ** 2 / sigma ** 2).sum(dim=-1)

    # ------------------------------------------------------------------ #
    # MMD against analytical target (terminal config only)               #
    # ------------------------------------------------------------------ #
    def _mmd_against_target(self, samples, n_target=None, kernel_sigma=1.0):
        if n_target is None:
            n_target = samples.shape[0]
        gt = self.mu_p_final + torch.randn(
            n_target, self.state_dim,
            dtype=self.dtype, device=self.device) * self.sigma
        xx = torch.cdist(samples, samples, p=2.0) ** 2
        yy = torch.cdist(gt, gt, p=2.0) ** 2
        xy = torch.cdist(samples, gt, p=2.0) ** 2
        s2 = 2.0 * kernel_sigma ** 2
        return float((torch.exp(-xx / s2).mean()
                    + torch.exp(-yy / s2).mean()
                    - 2 * torch.exp(-xy / s2).mean()).item())
