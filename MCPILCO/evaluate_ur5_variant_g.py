"""
Evaluation script for MC-PILCO UR5 Variant G (goal-reaching, starts at zeros).

Usage:
    python evaluate_ur5_variant_g.py                         # uses results_ur5_variant_g_5/1/log.pkl
    python evaluate_ur5_variant_g.py -log .../log.pkl
    python evaluate_ur5_variant_g.py -no_anim                # skip GIF (faster)

Generates (all in <save_dir>/):
    01_cost_curves.png        -- per-trial policy-optimisation curve(s)
    02_joint_positions.png    -- 6 panels: q_i over time, with TARGET (constant)
                                 and INITIAL state (zeros) marked
    03_joint_velocities.png   -- same for velocities
    04_end_effector_3d.png    -- arm's end-effector path vs (start point at zeros,
                                 target point at q_ref(4s))
    05_terminal_error.png     -- distance to target as trial progresses
    06_arm_animation.gif      -- stick-figure 3D playback

This is a GOAL-REACHING task analogous to CartPole swing-up:
    Start: zeros (arm straight up vertically)
    Goal:  q_ref(4s) (the GFN's trained terminal target)
Success = particle distribution converges to a tight Gaussian around the
target by the end of the horizon.
"""

import argparse
import math
import pathlib
import pickle as pkl
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
# Note: do NOT explicitly `from mpl_toolkits.mplot3d import Axes3D`.
# On systems with mixed matplotlib installs (e.g. pip-user + apt system)
# the explicit import can fail with ModuleNotFoundError on matplotlib.tri.
# Modern matplotlib auto-registers the '3d' projection on first use.

import mujoco

# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
p = argparse.ArgumentParser("evaluate_ur5_variant_g")
p.add_argument("-log", default="results_ur5_variant_g_5/1/log.pkl")
p.add_argument("-save", default="results_ur5_variant_g_5/1/plots")
p.add_argument("-no_anim", action="store_true",
               help="Skip the GIF animation (faster).")
p.add_argument("-anim_step", type=int, default=4,
               help="Render every Nth timestep in the animation.")
p.add_argument("-gfn_checkpoint", type=str, default=None,
               help="Path to the trained GFN .pt (used to overlay GFN sample "
                    "trajectories on the joint plots). Defaults to the "
                    "ur5_denoising_theta_step25000.pt sibling location.")
p.add_argument("-n_gfn_samples", type=int, default=20,
               help="Number of GFN trajectories to overlay on the plots.")
args = p.parse_args()

save_dir = pathlib.Path(args.save)
save_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading: {args.log}")
with open(args.log, "rb") as f:
    log = pkl.load(f)

# Regenerate reference trajectory (faster + dodges pickle issue with closures
# in config_log.pkl). Parameters MUST match the training script.
print("Regenerating reference trajectory ...")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from simulation_class.ur5_trajectory import generate_circle_trajectory
T_sampling = 0.02
T_control  = 4.0
_, q_ref, dq_ref, ee_ref = generate_circle_trajectory(
    center=(-0.6, 0.0, 0.4), radius=0.15,
    duration=T_control, dt=T_sampling)
q_ref  = np.asarray(q_ref)
dq_ref = np.asarray(dq_ref)
ee_ref = np.asarray(ee_ref)
N_traj = q_ref.shape[0] - 1
t_axis = np.arange(q_ref.shape[0]) * T_sampling

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow",
               "wrist_1", "wrist_2", "wrist_3"]


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def savefig(fig, name):
    path = save_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


# -----------------------------------------------------------------------
# Sample GFN trajectories so we can overlay them on the joint plots.
# Returns array of shape [n_samples, T_gfn+1, 12] and the corresponding
# time axis in physical seconds (linearly mapped from diffusion time).
# -----------------------------------------------------------------------
def sample_gfn_trajectories(n_samples=20, gfn_ckpt_path=None,
                            T_control_seconds=4.0):
    import torch

    # Resolve checkpoint path
    if gfn_ckpt_path is None:
        repo_root = pathlib.Path(__file__).resolve().parent.parent
        gfn_ckpt_path = str(
            repo_root / 'GflowNet' / 'gfn-diffusion' / 'energy_sampling'
            / 'ur5_denoising_theta_step25000.pt')

    if not pathlib.Path(gfn_ckpt_path).exists():
        print(f"  [WARN] GFN checkpoint not found at {gfn_ckpt_path}; "
              f"skipping GFN trace overlay.")
        return None, None

    # Make the GflowNet package importable
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    gfn_dir = repo_root / 'GflowNet' / 'gfn-diffusion' / 'energy_sampling'
    if (gfn_dir / 'models' / 'gfn.py').exists():
        if str(gfn_dir) not in sys.path:
            sys.path.insert(0, str(gfn_dir))
    else:
        print(f"  [WARN] GflowNet package not at {gfn_dir}; "
              f"skipping GFN trace overlay.")
        return None, None

    from models.gfn import GFN

    device = torch.device('cpu')

    # Build the GFN architecture (must match training)
    gfn_model = GFN(
        dim=12, s_emb_dim=64, hidden_dim=64,
        harmonics_dim=64, t_dim=64, log_var_range=4.0,
        t_scale=5.0, learned_variance=True, partial_energy=False,
        clipping=True, lgv_clip=1e2, gfn_clip=1e4,
        pb_scale_range=0.1, learn_pb=True, device=device,
        langevin_scaling_per_dimension=False,
    ).to(device)

    ckpt = torch.load(gfn_ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    gfn_model.load_state_dict(state)
    gfn_model.eval()
    for prm in gfn_model.parameters():
        prm.requires_grad_(False)

    # Analytical UR5 target energy (must match energies/ur5.py)
    mu_t = torch.tensor(ckpt.get('mu',
            [0.1884, -1.9495, 2.0472, -1.7108, -1.6774, 0.0,
             -0.5045, -0.0036, 0.0162, 0.0, 0.0921, 0.0]),
                       dtype=torch.float32, device=device)
    sigma_t = torch.tensor(ckpt.get('sigma',
            [0.10] * 6 + [0.50] * 6),
                          dtype=torch.float32, device=device)

    def log_reward(x, condition=None):
        return -0.5 * ((x - mu_t) ** 2 / sigma_t ** 2).sum(dim=-1)

    with torch.no_grad():
        init_state = torch.zeros(n_samples, 12,
                                 dtype=torch.float32, device=device)
        states, _, _, _ = gfn_model.get_trajectory_fwd(
            init_state, None, log_reward)
    trajs = states.cpu().numpy()                # [n_samples, T_gfn+1, 12]

    # Map diffusion time -> physical time axis [0, T_control_seconds]
    T_gfn_plus_1 = trajs.shape[1]
    gfn_time_axis = np.linspace(0.0, T_control_seconds, T_gfn_plus_1)

    return trajs, gfn_time_axis, mu_t.cpu().numpy(), sigma_t.cpu().numpy()


# -----------------------------------------------------------------------
# Pre-sample GFN trajectories ONCE -- used in plots 2 and 3
# -----------------------------------------------------------------------
print(f"\nSampling {args.n_gfn_samples} GFN trajectories for overlay ...")
gfn_trajs, gfn_time_axis, gfn_mu_t, gfn_sigma_t = sample_gfn_trajectories(
    n_samples=args.n_gfn_samples,
    gfn_ckpt_path=args.gfn_checkpoint,
    T_control_seconds=T_control,
)
if gfn_trajs is not None:
    print(f"  GFN trajectories shape: {gfn_trajs.shape}  "
          f"(samples x diffusion-steps x state-dim)")

# ---------------------------------------------------------------------------
# Start / End alignment check for Variant G
#   Variant G is *goal-reaching*: starts at zeros, drives toward q_ref(4s).
#   The GFN is trained to map zeros -> mu_t. By construction these should
#   coincide: zero-start aligned by definition, terminal aligned because
#   mu_t was set equal to q_ref[-1] during GFN training.
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("START / END ALIGNMENT CHECK   (Variant G vs GFN diffusion)")
print("=" * 78)
print(f"Variant G initial state (zeros)            :  "
      f"[{', '.join(f'{0.0:+.4f}' for _ in range(6))}]")
print(f"Variant G terminal target q_ref[-1] (joints):  "
      f"[{', '.join(f'{x:+.4f}' for x in q_ref[-1])}]")
if gfn_mu_t is not None:
    print(f"GFN initial (zeros)                        :  "
          f"[{', '.join(f'{0.0:+.4f}' for _ in range(6))}]")
    print(f"GFN terminal target mu_t (joints)          :  "
          f"[{', '.join(f'{x:+.4f}' for x in gfn_mu_t[:6])}]")
    d_start = float(np.linalg.norm(np.zeros(6) - 0.0))   # both zero
    d_end   = float(np.linalg.norm(q_ref[-1] - gfn_mu_t[:6]))
    print(f"\n||VariantG_start (zeros) - GFN_start (zeros)|| = "
          f"{d_start:.4f} rad ({'aligned (both zero)' if d_start < 0.05 else 'MISMATCH'})")
    print(f"||VariantG_target q_ref[-1] - GFN_mu_t      || = "
          f"{d_end:.4f} rad "
          f"({'aligned (GFN target == circle endpoint)' if d_end < 0.05 else 'MISMATCH'})")
print("=" * 78)


# -----------------------------------------------------------------------
# 1) Cost curves
# -----------------------------------------------------------------------
print("\n[1] Cost curves ...")
n_trials = len(log["cost_trial_list"])

n_cols = min(n_trials, 5)
n_rows = (n_trials + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(4.5 * n_cols, 3.8 * n_rows),
                         sharey=False)
axes = np.atleast_1d(axes).reshape(-1)
fig.suptitle("UR5 Variant G - policy optimisation cost per trial",
             fontsize=12, fontweight="bold")

for i in range(n_trials):
    c = np.asarray(log["cost_trial_list"][i])
    ax = axes[i]
    ax.plot(c, color="crimson", lw=1)
    ax.axhline(c.min(), color="black", ls="--", lw=0.7, alpha=0.6)
    ax.set_title(f"Trial {i+1}\nstart={c[0]:.0f}, end={c[-1]:.0f}, "
                 f"min={c.min():.0f}", fontsize=9)
    ax.set_xlabel("Opt step")
    ax.set_ylabel("Cost (KL + slack)")
    ax.grid(True, alpha=0.3)
for j in range(n_trials, len(axes)):
    axes[j].axis("off")
plt.tight_layout()
savefig(fig, "01_cost_curves.png")


# -----------------------------------------------------------------------
# 2) Joint position trajectories vs reference
# -----------------------------------------------------------------------
print("[2] Joint positions ...")
states_hist = log["state_samples_history"]
n_hist = len(states_hist)

# For Variant G: target is the CONSTANT q_ref(4s), shown as a horizontal line.
q_target = q_ref[-1]                       # constant terminal target
q_start  = np.zeros(6)                      # zeros (initial state for Variant G)

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
axes = axes.flatten()
fig.suptitle(f"UR5 Variant G - joint positions (goal-reaching)  "
             f"trial {n_hist - 1} = policy rollout",
             fontsize=12, fontweight="bold")

for j in range(6):
    ax = axes[j]
    # -- GFN sampled trajectories (the 'where GFN goes' overlay) --
    if gfn_trajs is not None:
        for k in range(gfn_trajs.shape[0]):
            ax.plot(gfn_time_axis, gfn_trajs[k, :, j],
                    color="orange", alpha=0.18, lw=0.7)
        # Mean of GFN samples for clarity
        gfn_mean = gfn_trajs[:, :, j].mean(axis=0)
        ax.plot(gfn_time_axis, gfn_mean,
                color="darkorange", lw=1.4, alpha=0.9,
                label=f"GFN mean (N={gfn_trajs.shape[0]} samples)")
    # Constant target line for goal-reaching
    ax.axhline(q_target[j], color="green", ls="--", lw=1.4,
               label=f"target q_ref(4s) = {q_target[j]:.2f}")
    # Initial state marker
    ax.axhline(q_start[j], color="blue", ls=":", lw=1.0, alpha=0.5,
               label="initial = 0")
    # Exploration trial (random torques)
    if n_hist > 0:
        ax.plot(t_axis[:len(states_hist[0])], states_hist[0][:, j],
                color="gray", lw=1.0, alpha=0.55, label="exploration")
    # Latest policy rollout
    if n_hist > 1:
        ax.plot(t_axis[:len(states_hist[1])], states_hist[1][:, j],
                color="crimson", lw=1.6, label=f"policy trial {n_hist-1}")
    ax.set_title(f"q[{j}]  {JOINT_NAMES[j]}", fontsize=9)
    ax.set_ylabel("rad")
    if j >= 3:
        ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    if j == 0:
        ax.legend(fontsize=6, loc="best", ncol=1)

plt.tight_layout()
savefig(fig, "02_joint_positions.png")


# -----------------------------------------------------------------------
# 3) Joint velocity trajectories vs reference
# -----------------------------------------------------------------------
print("[3] Joint velocities ...")
dq_target = dq_ref[-1]   # constant target velocity at the end

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
axes = axes.flatten()
fig.suptitle("UR5 Variant G - joint velocities (goal-reaching)",
             fontsize=12, fontweight="bold")

for j in range(6):
    ax = axes[j]
    # -- GFN sampled trajectories (dimension 6+j = velocity j) --
    if gfn_trajs is not None:
        for k in range(gfn_trajs.shape[0]):
            ax.plot(gfn_time_axis, gfn_trajs[k, :, 6 + j],
                    color="orange", alpha=0.18, lw=0.7)
        gfn_mean = gfn_trajs[:, :, 6 + j].mean(axis=0)
        ax.plot(gfn_time_axis, gfn_mean,
                color="darkorange", lw=1.4, alpha=0.9,
                label=f"GFN mean (N={gfn_trajs.shape[0]} samples)")
    ax.axhline(dq_target[j], color="green", ls="--", lw=1.4,
               label=f"target dq_ref(4s) = {dq_target[j]:.2f}")
    ax.axhline(0.0, color="blue", ls=":", lw=1.0, alpha=0.5,
               label="initial = 0")
    if n_hist > 0:
        ax.plot(t_axis[:len(states_hist[0])], states_hist[0][:, 6 + j],
                color="gray", lw=1.0, alpha=0.55, label="exploration")
    if n_hist > 1:
        ax.plot(t_axis[:len(states_hist[1])], states_hist[1][:, 6 + j],
                color="crimson", lw=1.6, label=f"policy trial {n_hist-1}")
    ax.set_title(f"dq[{j}]  {JOINT_NAMES[j]}", fontsize=9)
    ax.set_ylabel("rad/s")
    if j >= 3:
        ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    if j == 0:
        ax.legend(fontsize=6, loc="best")
plt.tight_layout()
savefig(fig, "03_joint_velocities.png")


# -----------------------------------------------------------------------
# 4) End-effector 3D trajectory via MuJoCo forward kinematics
# -----------------------------------------------------------------------
print("[4] End-effector 3D trajectory ...")
xml_path = pathlib.Path(__file__).parent / "simulation_class" / "ur5_model.xml"
mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
mj_data  = mujoco.MjData(mj_model)
ee_site  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")

# Body chain to draw the arm stick-figure (base -> shoulder -> ... -> ee_body)
BODY_NAMES = ["base", "shoulder", "upper_arm", "forearm",
              "wrist1", "wrist2", "wrist3", "ee_body"]
body_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in BODY_NAMES]


def forward_kinematics(q):
    mj_data.qpos[:] = q
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)
    ee_pos = mj_data.site_xpos[ee_site].copy()
    body_pos = np.array([mj_data.xpos[bid].copy() for bid in body_ids])  # [B, 3]
    return ee_pos, body_pos


# Compute end-effector trajectories for ref + exploration + policy trial
def trajectory_ee(states):
    ee = np.zeros((len(states), 3))
    for i, s in enumerate(states):
        ee[i], _ = forward_kinematics(s[:6])
    return ee


ee_explore = trajectory_ee(states_hist[0]) if n_hist > 0 else None
ee_policy  = trajectory_ee(states_hist[1]) if n_hist > 1 else None

# Goal-reaching: compute EE position at START (zeros) and at TARGET (q_ref(4s))
ee_start, _  = forward_kinematics(np.zeros(6))     # arm straight up
ee_target, _ = forward_kinematics(q_ref[-1])       # circle endpoint pose

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")
# Reference circle in light green (for context, not the actual target)
ax.plot(ee_ref[:, 0], ee_ref[:, 1], ee_ref[:, 2],
        color="green", ls=":", lw=1, alpha=0.4, label="(circle for context)")
# Explicit start/target points
ax.scatter(*ee_start,  c="blue",  s=140, marker="o",
           label="start (arm at zeros)")
ax.scatter(*ee_target, c="green", s=140, marker="*",
           label="target (q_ref(4s))")
if ee_explore is not None:
    ax.plot(ee_explore[:, 0], ee_explore[:, 1], ee_explore[:, 2],
            color="gray", lw=1, alpha=0.6, label="exploration")
if ee_policy is not None:
    ax.plot(ee_policy[:, 0], ee_policy[:, 1], ee_policy[:, 2],
            color="crimson", lw=2, label=f"policy trial {n_hist-1}")
    ax.scatter(*ee_policy[-1], c="crimson", s=80, marker="x",
               label="policy endpoint")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("End-effector 3D trajectory (Variant G goal-reaching)\n"
             f"target reached if crimson line ends near green star")
ax.legend(fontsize=8)
plt.tight_layout()
savefig(fig, "04_end_effector_3d.png")


# -----------------------------------------------------------------------
# 5) Tracking error over time (joint + Cartesian)
# -----------------------------------------------------------------------
print("[5] Distance-to-target over time ...")
if n_hist > 1:
    pol_states = states_hist[1]
    L = pol_states.shape[0]
    # For Variant G: error is distance to CONSTANT target (not time-varying)
    q_err  = pol_states[:L, :6] - q_ref[-1]            # vs target q_ref(4s)
    dq_err = pol_states[:L, 6:] - dq_ref[-1]           # vs target dq_ref(4s)
    pos_err_rms_joint = np.sqrt((q_err ** 2).mean(axis=1))   # [L]
    vel_err_rms_joint = np.sqrt((dq_err ** 2).mean(axis=1))

    # Cartesian: distance from ee at time t to ee_target (constant)
    cart_err = np.linalg.norm(ee_policy[:L] - ee_target[None, :], axis=1) * 1000  # mm

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    ax = axes[0]
    ax.plot(t_axis[:L], pos_err_rms_joint, color="crimson")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS joint position error (rad)")
    ax.set_title(f"RMS joint position error (mean = {pos_err_rms_joint.mean():.3f} rad)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_axis[:L], vel_err_rms_joint, color="darkred")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS joint velocity error (rad/s)")
    ax.set_title(f"RMS joint velocity error (mean = {vel_err_rms_joint.mean():.3f} rad/s)")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t_axis[:L], cart_err, color="purple")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("End-effector error (mm)")
    ax.set_title(f"Cartesian error (mean = {cart_err.mean():.1f} mm)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig(fig, "05_tracking_error.png")


# -----------------------------------------------------------------------
# 6) Arm animation (stick figure of links)
# -----------------------------------------------------------------------
if not args.no_anim and n_hist > 1:
    print(f"[6] Building arm animation (every {args.anim_step} steps) ...")
    pol_states = states_hist[1]
    step = max(1, args.anim_step)
    frame_idx = np.arange(0, pol_states.shape[0], step)

    # Pre-compute body positions for every animation frame
    body_positions = []
    for i in frame_idx:
        _, bp = forward_kinematics(pol_states[i, :6])
        body_positions.append(bp)
    body_positions = np.array(body_positions)        # [F, B, 3]

    # Also pre-compute end-effector path up to each frame
    ee_path = ee_policy

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw the reference circle (static)
    ax.plot(ee_ref[:, 0], ee_ref[:, 1], ee_ref[:, 2],
            color="green", ls="--", lw=1.4, label="reference")
    # Faded full end-effector path (static)
    ax.plot(ee_path[:, 0], ee_path[:, 1], ee_path[:, 2],
            color="crimson", lw=0.6, alpha=0.25)

    # Initial arm line (will be updated)
    arm_line, = ax.plot([], [], [], "o-", color="steelblue", lw=2.5,
                        markersize=6, label="UR5 arm")
    # Trailing path
    trail_line, = ax.plot([], [], [], color="crimson", lw=1.8,
                          alpha=0.85, label="ee path so far")
    ee_dot,    = ax.plot([], [], [], "o", color="red", markersize=8)
    title_obj   = ax.set_title("UR5 swinging through reference circle  (t = 0.00 s)",
                               fontsize=11)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(loc="upper right", fontsize=8)

    # Lock axis ranges for the whole animation
    all_pts = np.concatenate([
        ee_ref,
        ee_policy,
        body_positions.reshape(-1, 3),
    ], axis=0)
    pad = 0.1
    ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
    ax.set_zlim(all_pts[:, 2].min() - pad, all_pts[:, 2].max() + pad)
    ax.view_init(elev=22, azim=-55)

    def init():
        arm_line.set_data([], [])
        arm_line.set_3d_properties([])
        trail_line.set_data([], [])
        trail_line.set_3d_properties([])
        ee_dot.set_data([], [])
        ee_dot.set_3d_properties([])
        return arm_line, trail_line, ee_dot, title_obj

    def update(fnum):
        bp = body_positions[fnum]
        arm_line.set_data(bp[:, 0], bp[:, 1])
        arm_line.set_3d_properties(bp[:, 2])
        # trail = ee path up to current real timestep
        t_idx = frame_idx[fnum]
        trail_line.set_data(ee_path[:t_idx + 1, 0], ee_path[:t_idx + 1, 1])
        trail_line.set_3d_properties(ee_path[:t_idx + 1, 2])
        ee_dot.set_data([bp[-1, 0]], [bp[-1, 1]])
        ee_dot.set_3d_properties([bp[-1, 2]])
        title_obj.set_text(f"UR5 swinging through reference circle  "
                           f"(t = {t_idx * T_sampling:.2f} s)")
        return arm_line, trail_line, ee_dot, title_obj

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=len(frame_idx), interval=60, blit=False)
    out_path = save_dir / "06_arm_animation.gif"
    anim.save(out_path, writer=PillowWriter(fps=15))
    plt.close(fig)
    print(f"  saved -> {out_path}")


# -----------------------------------------------------------------------
# Summary printout
# -----------------------------------------------------------------------
print("\n" + "=" * 78)
print("EVALUATION SUMMARY — MC-PILCO UR5 Variant G")
print("=" * 78)
print(f"Log file:                  {args.log}")
print(f"T_control / T_sampling:    {T_control}s / {T_sampling}s  -> N_h = {N_traj}")
print(f"Trials with cost history:  {n_trials}")
print(f"Real-system rollouts:      {n_hist - 1} policy + 1 exploration")
print()
print("--- COST per trial ---")
for i in range(n_trials):
    c = np.asarray(log["cost_trial_list"][i])
    print(f"  Trial {i+1:>2}: opt_steps={len(c):>4}  "
          f"start={c[0]:>10.1f}  end={c[-1]:>10.1f}  min={c.min():>10.1f}")
print()
print("--- TRACKING (policy trial vs reference) ---")
if n_hist > 1:
    pol = states_hist[1]
    L = min(pol.shape[0], q_ref.shape[0])
    print(f"  Mean RMS joint position error: {pos_err_rms_joint.mean():.4f} rad "
          f"({np.rad2deg(pos_err_rms_joint.mean()):.2f} deg)")
    print(f"  Mean RMS joint velocity error: {vel_err_rms_joint.mean():.4f} rad/s")
    print(f"  Mean end-effector error:       {cart_err.mean():.1f} mm  "
          f"(max {cart_err.max():.1f} mm)")
    print(f"  Final end-effector error:      {cart_err[-1]:.1f} mm")
print()
print(f"All plots saved to: {save_dir.resolve()}")
print("=" * 78)
