"""
Evaluation script for MC-PILCO UR5 Variant D (Wasserstein-2 cost) results.

Usage:
    python evaluate_ur5_variant_d.py                         # uses results_ur5_variant_d_5/1/log.pkl
    python evaluate_ur5_variant_d.py -log .../log.pkl
    python evaluate_ur5_variant_d.py -no_anim                # skip GIF (faster)

Generates (all in <save_dir>/):
    01_cost_curves.png        -- per-trial policy-optimisation curve(s)
    02_joint_positions.png    -- 6 panels: actual q_i vs reference q_ref_i
    03_joint_velocities.png   -- 6 panels: actual dq_i vs reference dq_ref_i
    04_end_effector_3d.png    -- end-effector path vs reference circle
    05_tracking_error.png     -- joint-space + Cartesian errors over time
    06_arm_animation.gif      -- stick-figure 3D playback of the trial
    summary table printed to stdout

Cost is the W2^2-based KL-free objective (||mu_q-mu_p||^2 + ||sigma_q-sigma_p||^2)
plus chance-constraint slack on joint position bounds.
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
# Auto-registered via add_subplot(projection='3d') -- no explicit import.

import mujoco

# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
p = argparse.ArgumentParser("evaluate_ur5_variant_d")
p.add_argument("-log", default="results_ur5_variant_d_5/1/log.pkl")
p.add_argument("-save", default="results_ur5_variant_d_5/1/plots")
p.add_argument("-no_anim", action="store_true",
               help="Skip the GIF animation (faster).")
p.add_argument("-anim_step", type=int, default=4,
               help="Render every Nth timestep in the animation.")
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
fig.suptitle("UR5 Variant D - policy optimisation cost per trial",
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

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
axes = axes.flatten()
fig.suptitle(f"UR5 joint positions vs reference  (trial {n_hist - 1} = policy rollout)",
             fontsize=12, fontweight="bold")

for j in range(6):
    ax = axes[j]
    ax.plot(t_axis, q_ref[:, j], color="green", ls="--", lw=1.4,
            label="reference q_ref")
    # Exploration trial (random torques)
    if n_hist > 0:
        ax.plot(t_axis[:len(states_hist[0])], states_hist[0][:, j],
                color="gray", lw=1.0, alpha=0.5, label="exploration")
    # Trial-1 policy rollout
    if n_hist > 1:
        ax.plot(t_axis[:len(states_hist[1])], states_hist[1][:, j],
                color="crimson", lw=1.5, label=f"policy trial {n_hist-1}")
    ax.set_title(f"q[{j}]  {JOINT_NAMES[j]}", fontsize=9)
    ax.set_ylabel("rad")
    if j >= 3:
        ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    if j == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
savefig(fig, "02_joint_positions.png")


# -----------------------------------------------------------------------
# 3) Joint velocity trajectories vs reference
# -----------------------------------------------------------------------
print("[3] Joint velocities ...")
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
axes = axes.flatten()
fig.suptitle("UR5 joint velocities vs reference", fontsize=12, fontweight="bold")

for j in range(6):
    ax = axes[j]
    ax.plot(t_axis, dq_ref[:, j], color="green", ls="--", lw=1.4,
            label="reference dq_ref")
    if n_hist > 0:
        ax.plot(t_axis[:len(states_hist[0])], states_hist[0][:, 6 + j],
                color="gray", lw=1.0, alpha=0.5, label="exploration")
    if n_hist > 1:
        ax.plot(t_axis[:len(states_hist[1])], states_hist[1][:, 6 + j],
                color="crimson", lw=1.5, label=f"policy trial {n_hist-1}")
    ax.set_title(f"dq[{j}]  {JOINT_NAMES[j]}", fontsize=9)
    ax.set_ylabel("rad/s")
    if j >= 3:
        ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    if j == 0:
        ax.legend(fontsize=8)
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

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot(ee_ref[:, 0], ee_ref[:, 1], ee_ref[:, 2],
        color="green", ls="--", lw=2, label="reference circle")
if ee_explore is not None:
    ax.plot(ee_explore[:, 0], ee_explore[:, 1], ee_explore[:, 2],
            color="gray", lw=1, alpha=0.6, label="exploration")
if ee_policy is not None:
    ax.plot(ee_policy[:, 0], ee_policy[:, 1], ee_policy[:, 2],
            color="crimson", lw=2, label=f"policy trial {n_hist-1}")
    # Mark start/end
    ax.scatter(*ee_policy[0],  c="black", s=60, marker="o", label="start")
    ax.scatter(*ee_policy[-1], c="black", s=60, marker="x", label="end")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("End-effector 3D trajectory")
ax.legend(fontsize=9)
plt.tight_layout()
savefig(fig, "04_end_effector_3d.png")


# -----------------------------------------------------------------------
# 5) Tracking error over time (joint + Cartesian)
# -----------------------------------------------------------------------
print("[5] Tracking error over time ...")
if n_hist > 1:
    pol_states = states_hist[1]
    L = min(pol_states.shape[0], q_ref.shape[0])
    q_err  = pol_states[:L, :6] - q_ref[:L]            # [L, 6]
    dq_err = pol_states[:L, 6:] - dq_ref[:L]           # [L, 6]
    pos_err_rms_joint = np.sqrt((q_err ** 2).mean(axis=1))   # [L]
    vel_err_rms_joint = np.sqrt((dq_err ** 2).mean(axis=1))

    cart_err = np.linalg.norm(ee_policy[:L] - ee_ref[:L], axis=1) * 1000  # mm

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
print("EVALUATION SUMMARY — MC-PILCO UR5 Variant D")
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
