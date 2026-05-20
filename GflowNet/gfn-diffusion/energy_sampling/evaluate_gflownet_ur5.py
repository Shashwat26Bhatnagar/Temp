"""
Phase-1 GFN evaluation for UR5 — mirrors evaluate_gflownet.py (CartPole)
but adapted to the 12-D UR5 state x = [q1..q6, dq1..dq6].

Generates:
  - ur5_phase_portraits.png    -- 6 hex-bin panels (one per joint, q vs dq)
  - ur5_marginals.png          -- 12 histograms vs analytical Gaussian
  - ur5_timeseries_variance.png -- per-dim variance vs target across diffusion time
Plus prints MMD between GFN samples and analytical N(mu, diag(sigma^2)).

Usage:
    python evaluate_gflownet_ur5.py
    python evaluate_gflownet_ur5.py -checkpoint ur5_denoising_theta_step3000.pt
"""

import argparse
import pathlib
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from models.gfn import GFN
from energies.ur5 import UR5Energy


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
p = argparse.ArgumentParser("evaluate_gflownet_ur5")
p.add_argument("-checkpoint", default="ur5_denoising_theta_final.pt",
               help="Path to a UR5 GFN checkpoint .pt file")
p.add_argument("-out_dir",   default="ur5_eval_plots",
               help="Directory to save plots")
p.add_argument("-n_samples", type=int, default=2000,
               help="Number of GFN samples for evaluation")
p.add_argument("-seed",      type=int, default=42)
args = p.parse_args()


# -----------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device   = torch.device("cpu")
out_dir  = pathlib.Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

ckpt_path = pathlib.Path(args.checkpoint)
if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

print(f"Loading checkpoint: {ckpt_path}")
ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
state = ckpt.get("model_state_dict", ckpt)

# Build GFN architecture matching training config
gfn_model = GFN(
    dim=12, s_emb_dim=64, hidden_dim=64,
    harmonics_dim=64, t_dim=64, log_var_range=4.0,
    t_scale=5.0, learned_variance=True, partial_energy=False,
    clipping=True, lgv_clip=1e2, gfn_clip=1e4,
    pb_scale_range=0.1, learn_pb=True, device=device,
    langevin_scaling_per_dimension=False,
).to(device)

gfn_model.load_state_dict(state)
gfn_model.eval()
for prm in gfn_model.parameters():
    prm.requires_grad_(False)

energy   = UR5Energy(device)
mu_p     = energy.mu.detach().cpu().numpy()       # [12]
sigma_p  = energy.sigma.detach().cpu().numpy()    # [12]


# -----------------------------------------------------------------------
# Sample from trained GFN
# -----------------------------------------------------------------------
print(f"Drawing {args.n_samples} samples from GFN ...")
with torch.no_grad():
    init = torch.zeros(args.n_samples, 12, device=device)
    states, _, _, _ = gfn_model.get_trajectory_fwd(init, None, energy.log_reward)
    # states: [n, T+1, 12]
terminal_np = states[:, -1, :].cpu().numpy()      # [n, 12]
states_np   = states.cpu().numpy()                # [n, T+1, 12]


# -----------------------------------------------------------------------
# MMD against analytical target
# -----------------------------------------------------------------------

def mmd(x, y, sigma=1.0):
    xx = torch.cdist(x, x, p=2.0) ** 2
    yy = torch.cdist(y, y, p=2.0) ** 2
    xy = torch.cdist(x, y, p=2.0) ** 2
    s2 = 2.0 * sigma ** 2
    return float((torch.exp(-xx / s2).mean()
                + torch.exp(-yy / s2).mean()
                - 2 * torch.exp(-xy / s2).mean()).item())


gt = (torch.tensor(mu_p) + torch.randn(args.n_samples, 12) *
      torch.tensor(sigma_p))
mmd_val = mmd(torch.tensor(terminal_np), gt, sigma=1.0)
print(f"MMD (GFN samples vs analytical N(mu, diag(sigma^2))): {mmd_val:.6f}")
print(f"  (lower is better; ~1e-3 indicates good convergence)")


# -----------------------------------------------------------------------
# Plot 1: per-joint phase portraits (q vs dq for each of 6 joints)
# -----------------------------------------------------------------------
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow",
               "wrist_1",      "wrist_2",       "wrist_3"]

fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
axes = axes.flatten()
fig.suptitle("UR5 GFN samples — per-joint phase portraits (q vs dq)\n"
             "Red contours: analytical N(mu, diag(sigma^2))",
             fontsize=12, fontweight="bold")

for j in range(6):
    ax = axes[j]
    q_samp  = terminal_np[:, j]
    dq_samp = terminal_np[:, 6 + j]

    hb = ax.hexbin(q_samp, dq_samp, gridsize=30, cmap="Blues")
    fig.colorbar(hb, ax=ax, label="count")

    # Analytical contour
    pj = np.linspace(mu_p[j] - 4 * sigma_p[j],
                     mu_p[j] + 4 * sigma_p[j], 60)
    pk = np.linspace(mu_p[6 + j] - 4 * sigma_p[6 + j],
                     mu_p[6 + j] + 4 * sigma_p[6 + j], 60)
    XX, YY = np.meshgrid(pj, pk)
    E = 0.5 * (((XX - mu_p[j])     / sigma_p[j])     ** 2
             + ((YY - mu_p[6 + j]) / sigma_p[6 + j]) ** 2)
    ax.contour(XX, YY, E, levels=6, colors="red", alpha=0.6, linewidths=0.8)

    ax.scatter([mu_p[j]], [mu_p[6 + j]], c="red", s=40, marker="x", zorder=5)
    ax.set_xlabel(f"q[{j}] (rad)")
    ax.set_ylabel(f"dq[{j}] (rad/s)")
    ax.set_title(f"{JOINT_NAMES[j]}  (mu={mu_p[j]:.2f}, {mu_p[6+j]:.2f})",
                 fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / "ur5_phase_portraits.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_dir / 'ur5_phase_portraits.png'}")


# -----------------------------------------------------------------------
# Plot 2: 12 marginal histograms vs analytical Gaussian
# -----------------------------------------------------------------------
fig, axes = plt.subplots(4, 3, figsize=(14, 12))
axes = axes.flatten()
fig.suptitle("UR5 GFN marginals (12-D) vs analytical N(mu, sigma^2)",
             fontsize=12, fontweight="bold")

DIM_LABELS = [f"q[{i}] {JOINT_NAMES[i]}"     for i in range(6)] \
           + [f"dq[{i}] {JOINT_NAMES[i]}"    for i in range(6)]

for k in range(12):
    ax = axes[k]
    samp = terminal_np[:, k]
    ax.hist(samp, bins=40, density=True, alpha=0.6, color="steelblue",
            label="GFN")

    # Analytical pdf overlay
    xx = np.linspace(mu_p[k] - 4 * sigma_p[k],
                     mu_p[k] + 4 * sigma_p[k], 200)
    pdf = (1.0 / (sigma_p[k] * np.sqrt(2 * np.pi))) * \
          np.exp(-0.5 * ((xx - mu_p[k]) / sigma_p[k]) ** 2)
    ax.plot(xx, pdf, color="red", lw=1.5, label="target")

    ax.axvline(mu_p[k], color="red", ls="--", lw=0.8)
    ax.set_title(f"{DIM_LABELS[k]}\nmu={mu_p[k]:.3f}, sigma={sigma_p[k]:.3f}",
                 fontsize=8)
    ax.grid(True, alpha=0.3)
    if k == 0:
        ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(out_dir / "ur5_marginals.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_dir / 'ur5_marginals.png'}")


# -----------------------------------------------------------------------
# Plot 3: variance over diffusion time (per dim) vs analytical target
# -----------------------------------------------------------------------
T_plus1   = states_np.shape[1]
timesteps = np.linspace(0, 1, T_plus1)
var_traj  = states_np.var(axis=0)              # [T+1, 12]

fig, axes = plt.subplots(4, 3, figsize=(14, 11), sharex=True)
axes = axes.flatten()
fig.suptitle("UR5 GFN per-dimension variance across diffusion time\n"
             "(red = analytical target variance sigma^2)",
             fontsize=12, fontweight="bold")

for k in range(12):
    ax = axes[k]
    ax.plot(timesteps, var_traj[:, k], color="steelblue", lw=1.4,
            label="empirical")
    ax.axhline(sigma_p[k] ** 2, color="red", ls="--", lw=1.0,
               label=f"target {sigma_p[k]**2:.4f}")
    ax.set_title(f"{DIM_LABELS[k]}", fontsize=8)
    ax.grid(True, alpha=0.3)
    if k == 0:
        ax.legend(fontsize=7)
    if k >= 9:
        ax.set_xlabel("t (normalised)")

plt.tight_layout()
plt.savefig(out_dir / "ur5_timeseries_variance.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_dir / 'ur5_timeseries_variance.png'}")


# -----------------------------------------------------------------------
# Numerical summary
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("Per-dimension mean / std comparison")
print("=" * 70)
print(f"{'Dim':<25} {'target_mu':>10} {'gfn_mu':>10} "
      f"{'target_sigma':>14} {'gfn_sigma':>12}")
print("-" * 75)
for k in range(12):
    print(f"{DIM_LABELS[k]:<25} {mu_p[k]:>10.4f} {terminal_np[:,k].mean():>10.4f} "
          f"{sigma_p[k]:>14.4f} {terminal_np[:,k].std():>12.4f}")

print()
print(f"Overall MMD: {mmd_val:.6f}")
print(f"All plots saved to: {out_dir.resolve()}")
