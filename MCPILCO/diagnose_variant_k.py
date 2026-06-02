"""
Diagnostic: GFN diffusion trajectory vs cartpole physical trajectory.

Shows why Variant K struggles: the GFN's conditional transition at time t
is defined on the DENOISING path from zeros to target, which is NOT the
physical cartpole swing-up path. The policy is being asked to track a
smooth monotone denoising curve, but cartpole needs energy pumping
(theta must go NEGATIVE before it can reach pi).

Usage (only needs a GFN checkpoint, no cartpole log required):
    python diagnose_variant_k.py

Optional — overlay real cartpole rollouts if a log is available:
    python diagnose_variant_k.py -log results_variant_c_5/1/log.pkl
    python diagnose_variant_k.py -log results_variant_k_reverse/1/log.pkl
"""

import argparse
import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser("diagnose_variant_k")
p.add_argument("-log", default=None,
               help="Optional cartpole log.pkl to overlay real rollouts.")
p.add_argument("-checkpoint", default=None,
               help="GFN checkpoint .pt (auto-located if omitted).")
p.add_argument("-save", default="diagnose_variant_k_plots",
               help="Output directory for plots.")
p.add_argument("-n_gfn", type=int, default=50,
               help="Number of GFN diffusion trajectories to sample.")
args = p.parse_args()

save_dir = pathlib.Path(args.save)
save_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Locate checkpoint
# ---------------------------------------------------------------------------
if args.checkpoint is None:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    args.checkpoint = str(
        repo_root / "GflowNet" / "gfn-diffusion" / "energy_sampling"
        / "cartpole_denoising_theta_final.pt")

if not pathlib.Path(args.checkpoint).exists():
    print(f"[ERROR] GFN checkpoint not found: {args.checkpoint}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Sample full GFN diffusion trajectories (keep ALL 101 steps)
# ---------------------------------------------------------------------------
print(f"Loading GFN from {args.checkpoint} ...")
import torch

repo_root = pathlib.Path(__file__).resolve().parent.parent
gfn_dir = repo_root / "GflowNet" / "gfn-diffusion" / "energy_sampling"
if str(gfn_dir) not in sys.path:
    sys.path.insert(0, str(gfn_dir))

from models.gfn import GFN as _GFN
device = torch.device("cpu")
gfn = _GFN(
    dim=4, s_emb_dim=64, hidden_dim=64,
    harmonics_dim=64, t_dim=64, log_var_range=4.0,
    t_scale=5.0, learned_variance=True, partial_energy=False,
    clipping=True, lgv_clip=1e2, gfn_clip=1e4,
    pb_scale_range=0.1, learn_pb=True, device=device,
    langevin_scaling_per_dimension=False,
).to(device)

ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
gfn.load_state_dict(ckpt.get("model_state_dict", ckpt))
gfn.eval()
for prm in gfn.parameters():
    prm.requires_grad_(False)

PI = math.pi
mu32    = torch.tensor([0.0, 0.0, PI, 0.0],         dtype=torch.float32)
sigma32 = torch.tensor([0.5, 0.5, 0.1, 0.1],        dtype=torch.float32)

def log_reward(x, condition=None):
    return -0.5 * ((x - mu32) ** 2 / sigma32 ** 2).sum(dim=-1)

print(f"Sampling {args.n_gfn} GFN diffusion trajectories ...")
with torch.no_grad():
    init = torch.zeros(args.n_gfn, 4, dtype=torch.float32)
    states, _, _, _ = gfn.get_trajectory_fwd(init, None, log_reward)
# states: [n_gfn, T_gfn+1, 4]   where T_gfn = gfn.trajectory_length = 100
gfn_traj = states.cpu().numpy()          # [N, 101, 4]
T_gfn    = gfn_traj.shape[1] - 1        # = 100
gfn_time = np.linspace(0.0, 1.0, T_gfn + 1)   # normalised [0,1]

# Per-step marginal mean and std of theta (dim 2)
gfn_theta_mean = gfn_traj[:, :, 2].mean(axis=0)   # [101]
gfn_theta_std  = gfn_traj[:, :, 2].std(axis=0)    # [101]

# GFN live conditional drift from zeros at several t values
print("Computing GFN live conditional drift at key times ...")
gfn_drift_t   = []
gfn_drift_mu  = []
gfn_drift_sig = []
for t_frac in np.linspace(0.0, 0.99, 20):
    z = torch.zeros(1, 4, dtype=torch.float32)
    def _dummy(x, condition=None):
        return torch.zeros(x.shape[0])
    with torch.no_grad():
        pfs, _ = gfn.predict_next_state(z, float(t_frac), _dummy)
        pf_mean, pf_logvar = gfn.split_params(pfs)
        mu_out  = z + gfn.dt * pf_mean     # GFN's "where do I push the state"
        sig_out = (gfn.dt * torch.exp(pf_logvar)).sqrt()
    gfn_drift_t.append(t_frac)
    gfn_drift_mu.append(mu_out[0, 2].item())   # theta component
    gfn_drift_sig.append(sig_out[0, 2].item())

gfn_drift_t   = np.array(gfn_drift_t)
gfn_drift_mu  = np.array(gfn_drift_mu)
gfn_drift_sig = np.array(gfn_drift_sig)

# ---------------------------------------------------------------------------
# Load real cartpole rollouts if available
# ---------------------------------------------------------------------------
real_trajs = []
variant_name = "unknown"
T_sampling = 0.05
if args.log:
    import pickle
    try:
        with open(args.log, "rb") as f:
            log = pickle.load(f)
        hist = log.get("state_samples_history", [])
        # skip exploration (index 0), take policy rollouts
        for i in range(1, len(hist)):
            real_trajs.append(np.asarray(hist[i]))
        variant_name = pathlib.Path(args.log).parts[-3]
        print(f"Loaded {len(real_trajs)} real policy rollouts from {args.log}")
    except Exception as e:
        print(f"[WARN] Could not load log: {e}")

# ---------------------------------------------------------------------------
# PLOT 1: GFN diffusion marginal theta(t) vs physical time axis
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("GFN Diffusion Path vs Cartpole Physical Path\n"
             "Both x-axes normalised to [0,1] for fair comparison",
             fontsize=13, fontweight="bold")

ax = axes[0]
for i in range(min(args.n_gfn, 20)):
    ax.plot(gfn_time, gfn_traj[i, :, 2],
            color="gold", alpha=0.25, lw=0.8)
ax.fill_between(gfn_time,
                gfn_theta_mean - 2*gfn_theta_std,
                gfn_theta_mean + 2*gfn_theta_std,
                alpha=0.25, color="orange", label="GFN ±2σ")
ax.plot(gfn_time, gfn_theta_mean,
        color="darkorange", lw=2.0, label="GFN mean θ")
ax.axhline(PI,  color="green", ls="--", lw=1.5, label="target θ=π")
ax.axhline(0.0, color="gray",  ls=":",  lw=1.0, alpha=0.6)
ax.set_xlabel("Normalised diffusion time  t = step / 100")
ax.set_ylabel("θ (rad)")
ax.set_title("GFN diffusion path (denoising, zeros → π)\n"
             "SMOOTH & MONOTONE — θ increases steadily")
ax.legend(fontsize=8)
ax.set_ylim(-0.5, PI + 0.5)
ax.grid(True, alpha=0.3)

ax = axes[1]
if real_trajs:
    for i, traj in enumerate(real_trajs):
        T_traj = traj.shape[0]
        t_norm = np.linspace(0.0, 1.0, T_traj)
        lbl = f"Trial {i+1}" if i < 6 else None
        ax.plot(t_norm, traj[:, 2], alpha=0.75, lw=1.5, label=lbl)
    ax.axhline(PI,  color="green", ls="--", lw=1.5, label="target θ=π")
    ax.axhline(0.0, color="gray",  ls=":",  lw=1.0, alpha=0.6)
    ax.axhline(-PI, color="gray",  ls=":",  lw=0.8, alpha=0.4)
    ax.set_title(f"Real cartpole rollouts ({variant_name})\n"
                 "NOT MONOTONE — must go negative (energy pumping) to reach π")
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, "No log provided.\nPass -log <path>/log.pkl\nto see real rollouts.",
            ha="center", va="center", transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))
    ax.set_title("Real cartpole rollouts (not loaded)")
ax.set_xlabel("Normalised physical time  t = step / N_h")
ax.set_ylabel("θ (rad)")
ax.set_ylim(-PI - 0.3, PI + 0.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = save_dir / "01_gfn_vs_cartpole_path.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved -> {out}")

# ---------------------------------------------------------------------------
# PLOT 2: GFN live conditional drift from zeros -- what does the network "want"?
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("GFN Live Conditional: what does predict_next_state(zeros, t) output?\n"
             "This is the 'target' Variant K pushes each particle toward at time t",
             fontsize=12, fontweight="bold")

ax = axes[0]
ax.plot(gfn_drift_t, gfn_drift_mu, "o-", color="darkorange", lw=2, ms=5,
        label="GFN μ_θ (next state mean from zeros)")
ax.fill_between(gfn_drift_t,
                gfn_drift_mu - 2*gfn_drift_sig,
                gfn_drift_mu + 2*gfn_drift_sig,
                alpha=0.2, color="orange", label="±2σ")
ax.axhline(PI,  color="green", ls="--", lw=1.5, label="target π")
ax.axhline(0.0, color="gray",  ls=":",  lw=1.0)
ax.set_xlabel("Diffusion time t = k/N_h")
ax.set_ylabel("GFN predicted μ_θ from zeros")
ax.set_title("GFN conditional drift (θ dimension)\n"
             "What 'next θ' does GFN predict from the zero state?")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(gfn_drift_t, gfn_drift_sig, "s-", color="purple", lw=2, ms=5,
        label="GFN σ_θ (uncertainty)")
ax.set_xlabel("Diffusion time t = k/N_h")
ax.set_ylabel("GFN predicted σ_θ from zeros")
ax.set_title("GFN conditional uncertainty (θ dimension)\n"
             "Wide early (forgiving), tight late (precise)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = save_dir / "02_gfn_live_conditional_drift.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  saved -> {out}")

# ---------------------------------------------------------------------------
# PLOT 3: The MISMATCH — overlay GFN marginal mean vs real rollouts theta
# ---------------------------------------------------------------------------
if real_trajs:
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("THE MISMATCH: GFN diffusion path vs real cartpole trajectories\n"
                 "GFN path is monotone; cartpole MUST dip below 0 to build energy",
                 fontsize=12, fontweight="bold")

    ax.fill_between(gfn_time,
                    gfn_theta_mean - 2*gfn_theta_std,
                    gfn_theta_mean + 2*gfn_theta_std,
                    alpha=0.25, color="orange", label="GFN marginal ±2σ")
    ax.plot(gfn_time, gfn_theta_mean,
            color="darkorange", lw=2.5, label="GFN mean path (target at step k)")

    for i, traj in enumerate(real_trajs):
        T_traj = traj.shape[0]
        t_norm = np.linspace(0.0, 1.0, T_traj)
        ax.plot(t_norm, traj[:, 2], alpha=0.7, lw=1.5,
                label=f"Trial {i+1}" if i < 6 else None)

    ax.axhline(PI,   color="green", ls="--", lw=1.5, label="target π")
    ax.axhline(0.0,  color="gray",  ls=":",  lw=1.0, alpha=0.6)
    ax.axhline(-0.5, color="red",   ls=":",  lw=0.8, alpha=0.5,
               label="typical energy-pumping dip")
    ax.set_xlabel("Normalised time [0,1]")
    ax.set_ylabel("θ (rad)")
    ax.set_ylim(-PI - 0.3, PI + 0.5)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Shade the "GFN says be here, cartpole can't be" zone
    ax.fill_between(gfn_time[:40],
                    np.zeros(40),
                    gfn_theta_mean[:40],
                    alpha=0.08, color="red",
                    label="Zone where GFN pulls θ up, cartpole MUST pull down")

    plt.tight_layout()
    out = save_dir / "03_mismatch_overlay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out}")

# ---------------------------------------------------------------------------
# PRINT DIAGNOSIS
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("DIAGNOSIS: Why Variant K peaks at Trial 2 then degrades")
print("=" * 70)
print(f"""
GFN diffusion path (what Variant K's KL target is at each step k):
  - Starts: θ ≈ {gfn_theta_mean[0]:.3f} rad  (zeros)
  - Mid:    θ ≈ {gfn_theta_mean[50]:.3f} rad  (t=0.5)
  - End:    θ ≈ {gfn_theta_mean[-1]:.3f} rad  (target ≈ π)
  - Shape:  MONOTONE INCREASING — GFN smoothly denoises toward π

Physical cartpole swing-up requires:
  - θ must go NEGATIVE (pump energy) before rising to π
  - This contradicts the GFN path: at t=0.1 the GFN wants θ≈{gfn_theta_mean[10]:.2f},
    but the policy NEEDS θ to be going DOWN to build momentum

Variant K's self-defeating loop (Trial 2→3 collapse):
  1. Trial 2 works by luck/exploration → GP gets data near θ=π
  2. GP becomes CONFIDENT near π → σ²_GP shrinks there
  3. Reverse KL trace term σ²_GFN/σ²_GP explodes near π
  4. Gradient pushes policy AWAY from the θ≈π region
  5. Trial 3: θ rises but stops short — policy forbidden from its own solution

Root causes (ranked):
  [1] GFN denoising path ≠ physical swing-up trajectory
      The GFN was trained as a goal-conditioned denoiser, not as a
      physical trajectory model. Its intermediate distributions at t=0.3
      say "be at θ≈{gfn_theta_mean[30]:.2f}" but cartpole needs to be going backward.
  [2] Reverse KL with GP variance in denominator — confidence-at-success
      causes gradient to eject the policy from the winning region.
  [3] State weighting via softmax(log gp(s_k)) uses the SAME GP that
      has the confidence problem — particles near π get high weight
      AND high KL penalty simultaneously, amplifying the ejection.

The comparison plots are saved to: {save_dir}/
""")
