"""
Evaluation script for MC-PILCO UR5 Variant H (local per-step Mahalanobis).

Designed to be SAFE to run after ANY completed trial, including:
  * After only exploration (no policy trials yet)
  * After the first policy trial
  * Mid-training (between trials)
  * Final (after all trials)

Usage:
    python evaluate_ur5_variant_h.py                          # default log dir
    python evaluate_ur5_variant_h.py -log <path>/log.pkl
    python evaluate_ur5_variant_h.py -no_anim                 # skip GIF
    python evaluate_ur5_variant_h.py -trial 2                 # plot a specific rollout

Outputs into <save_dir>/:
    01_cost_curves.png         (per-trial policy-optim cost; skipped if no trial done)
    02_joint_positions.png     6 panels q_i(t) vs time-varying target
    03_joint_velocities.png    same for velocities
    04_end_effector_3d.png     EE path vs reference circle
    05_tracking_error.png      RMS joint/EE error over time
    06_arm_animation.gif       3D stick-figure playback (unless -no_anim)

This is the trajectory-tracking analog of Variant G's evaluator. Targets
are TIME-VARYING here (q_ref(t)) rather than the constant q_ref(4s) used
for Variant G's goal-reaching.
"""

# --- fix Debian dist-packages precedence over user-site mpl_toolkits ---
import sys
_USER_SITE_CANDIDATES = [
    '/afs/inf.ed.ac.uk/user/s28/s2892016/.local/lib/python3.12/site-packages',
]
import site
_USER_SITE_CANDIDATES.append(site.getusersitepackages())
for _us in _USER_SITE_CANDIDATES:
    if _us in sys.path:
        sys.path.remove(_us)
sys.path.insert(0, _USER_SITE_CANDIDATES[0])
# Push any /usr/lib/.../dist-packages entries to the end
_dist  = [p for p in sys.path if p.startswith('/usr/lib/python3/dist-packages')]
_other = [p for p in sys.path if p not in _dist and p != _USER_SITE_CANDIDATES[0]]
sys.path = [_USER_SITE_CANDIDATES[0]] + _other + _dist
# Purge any already-cached old modules so the new path wins
for _m in list(sys.modules):
    if _m == 'matplotlib' or _m.startswith('matplotlib.') \
       or _m == 'mpl_toolkits' or _m.startswith('mpl_toolkits.'):
        del sys.modules[_m]
# ---------------------------------------------------------------------------

import argparse
import math
import pathlib
import pickle as pkl

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (forces '3d' projection registration)
from matplotlib.animation import FuncAnimation, PillowWriter

import mujoco

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser("evaluate_ur5_variant_h")
p.add_argument("-log", default="results_ur5_variant_h/1/log.pkl")
p.add_argument("-save", default=None,
               help="Save directory. Default = <log dir>/plots_<n_done_trials>.")
p.add_argument("-trial", type=int, default=-1,
               help="Which real-system rollout index to plot (default = last).")
p.add_argument("-no_anim", action="store_true")
p.add_argument("-anim_step", type=int, default=4)
p.add_argument("-gfn_checkpoint", type=str, default=None,
               help="GFN .pt for overlay (auto-located if omitted).")
p.add_argument("-n_gfn_samples", type=int, default=20)
args = p.parse_args()

log_path = pathlib.Path(args.log)
if not log_path.exists():
    raise SystemExit(f"[evaluate_h] log file not found: {log_path}\n"
                     f"  Did training run yet? Expected: {log_path.resolve()}")

print(f"Loading: {log_path}")
with open(log_path, "rb") as f:
    log = pkl.load(f)

n_trials = len(log.get("cost_trial_list", []))
n_hist   = len(log.get("state_samples_history", []))

# Resolve save directory
if args.save is None:
    save_dir = log_path.parent / f"plots_after_trial_{n_trials}"
else:
    save_dir = pathlib.Path(args.save)
save_dir.mkdir(parents=True, exist_ok=True)
print(f"Save dir: {save_dir}")
print(f"State of run: {n_hist} real rollouts, {n_trials} completed policy trials.")

# Reference trajectory
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


def savefig(fig, name):
    path = save_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


# ---------------------------------------------------------------------------
# Pick the rollout to plot
# ---------------------------------------------------------------------------
states_hist = log.get("state_samples_history", [])
if n_hist == 0:
    print("[evaluate_h] No real rollouts in log yet -- nothing to plot.")
    sys.exit(0)

trial_idx = args.trial if args.trial >= 0 else (n_hist - 1)
trial_idx = max(0, min(trial_idx, n_hist - 1))
print(f"Plotting rollout index {trial_idx} of {n_hist - 1}.")


# ---------------------------------------------------------------------------
# 1) Cost curves (only if at least one trial done)
# ---------------------------------------------------------------------------
if n_trials > 0:
    print("\n[1] Cost curves ...")
    n_cols = min(n_trials, 5)
    n_rows = (n_trials + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.8 * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)
    fig.suptitle("UR5 Variant H -- policy optimisation cost per trial",
                 fontsize=12, fontweight="bold")
    for i in range(n_trials):
        c  = np.asarray(log["cost_trial_list"][i])
        ax = axes[i]
        ax.plot(c, color="teal", lw=1)
        ax.axhline(c.min(), color="black", ls="--", lw=0.7, alpha=0.6)
        ax.set_title(f"Trial {i+1}\nstart={c[0]:.0f}, end={c[-1]:.0f}, "
                     f"min={c.min():.0f}", fontsize=9)
        ax.set_xlabel("Opt step")
        ax.set_ylabel("Cost (Mahal + slack)")
        ax.grid(True, alpha=0.3)
    for j in range(n_trials, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    savefig(fig, "01_cost_curves.png")
else:
    print("[1] Cost curves skipped (no completed policy trials).")


# ---------------------------------------------------------------------------
# 2) Joint position trajectories vs TIME-VARYING reference
# ---------------------------------------------------------------------------
print("[2] Joint positions ...")
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
axes = axes.flatten()
fig.suptitle(f"UR5 Variant H -- joint positions (rollout {trial_idx})",
             fontsize=12, fontweight="bold")
roll = states_hist[trial_idx]
for j in range(6):
    ax = axes[j]
    ax.plot(t_axis, q_ref[:, j], color="green", ls="--", lw=1.4,
            label="ref q_ref(t)")
    ax.plot(t_axis[:len(roll)], roll[:, j],
            color="teal", lw=1.6, label=f"rollout {trial_idx}")
    ax.axhline(q_ref[0, j], color="blue", ls=":", lw=0.8, alpha=0.5,
               label="start q_ref(0)")
    ax.set_title(f"q[{j}]  {JOINT_NAMES[j]}", fontsize=9)
    ax.set_ylabel("rad")
    if j >= 3:
        ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    if j == 0:
        ax.legend(fontsize=6)
plt.tight_layout()
savefig(fig, "02_joint_positions.png")


# ---------------------------------------------------------------------------
# 3) Joint velocity trajectories vs reference
# ---------------------------------------------------------------------------
print("[3] Joint velocities ...")
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
axes = axes.flatten()
fig.suptitle(f"UR5 Variant H -- joint velocities (rollout {trial_idx})",
             fontsize=12, fontweight="bold")
for j in range(6):
    ax = axes[j]
    ax.plot(t_axis, dq_ref[:, j], color="green", ls="--", lw=1.4,
            label="ref dq_ref(t)")
    ax.plot(t_axis[:len(roll)], roll[:, 6 + j],
            color="teal", lw=1.6, label=f"rollout {trial_idx}")
    ax.set_title(f"dq[{j}]  {JOINT_NAMES[j]}", fontsize=9)
    ax.set_ylabel("rad/s")
    if j >= 3:
        ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    if j == 0:
        ax.legend(fontsize=6)
plt.tight_layout()
savefig(fig, "03_joint_velocities.png")


# ---------------------------------------------------------------------------
# 4) End-effector 3D trajectory via MuJoCo forward kinematics
# ---------------------------------------------------------------------------
print("[4] End-effector 3D trajectory ...")
xml_path = pathlib.Path(__file__).parent / "simulation_class" / "ur5_model.xml"
mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
mj_data  = mujoco.MjData(mj_model)
ee_site  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")

BODY_NAMES = ["base", "shoulder", "upper_arm", "forearm",
              "wrist1", "wrist2", "wrist3", "ee_body"]
body_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in BODY_NAMES]


def forward_kinematics(q):
    mj_data.qpos[:] = q
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)
    ee_pos = mj_data.site_xpos[ee_site].copy()
    body_pos = np.array([mj_data.xpos[bid].copy() for bid in body_ids])
    return ee_pos, body_pos


def trajectory_ee(states):
    ee = np.zeros((len(states), 3))
    for i, s in enumerate(states):
        ee[i], _ = forward_kinematics(s[:6])
    return ee


ee_traj   = trajectory_ee(roll)
ee_start  = ee_ref[0]
ee_end    = ee_ref[-1]

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot(ee_ref[:, 0], ee_ref[:, 1], ee_ref[:, 2],
        color="green", ls="--", lw=1.5, label="reference circle")
ax.scatter(*ee_start, c="blue", s=140, marker="o", label="start q_ref(0)")
ax.scatter(*ee_end,   c="green", s=140, marker="*", label="end q_ref(4s)")
ax.plot(ee_traj[:, 0], ee_traj[:, 1], ee_traj[:, 2],
        color="teal", lw=2, label=f"rollout {trial_idx}")
ax.scatter(*ee_traj[-1], c="teal", s=80, marker="x", label="rollout endpoint")
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
ax.set_title(f"End-effector 3D trajectory (Variant H, rollout {trial_idx})")
ax.legend(fontsize=8)
plt.tight_layout()
savefig(fig, "04_end_effector_3d.png")


# ---------------------------------------------------------------------------
# 5) Tracking error over time (joint + Cartesian) -- TIME-VARYING reference
# ---------------------------------------------------------------------------
print("[5] Tracking error over time ...")
L = min(roll.shape[0], q_ref.shape[0])
q_err  = roll[:L, :6] - q_ref[:L]
dq_err = roll[:L, 6:] - dq_ref[:L]
pos_err_rms_joint = np.sqrt((q_err ** 2).mean(axis=1))
vel_err_rms_joint = np.sqrt((dq_err ** 2).mean(axis=1))
cart_err = np.linalg.norm(ee_traj[:L] - ee_ref[:L], axis=1) * 1000  # mm

fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
ax = axes[0]
ax.plot(t_axis[:L], pos_err_rms_joint, color="teal")
ax.set_xlabel("Time (s)"); ax.set_ylabel("RMS q error (rad)")
ax.set_title(f"RMS joint position error (mean = {pos_err_rms_joint.mean():.3f} rad)")
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(t_axis[:L], vel_err_rms_joint, color="darkcyan")
ax.set_xlabel("Time (s)"); ax.set_ylabel("RMS dq error (rad/s)")
ax.set_title(f"RMS joint velocity error (mean = {vel_err_rms_joint.mean():.3f} rad/s)")
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(t_axis[:L], cart_err, color="purple")
ax.set_xlabel("Time (s)"); ax.set_ylabel("EE error (mm)")
ax.set_title(f"Cartesian error (mean = {cart_err.mean():.1f} mm, "
             f"max = {cart_err.max():.1f} mm)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig(fig, "05_tracking_error.png")


# ---------------------------------------------------------------------------
# 6) Arm animation
# ---------------------------------------------------------------------------
if not args.no_anim:
    print(f"[6] Animation (every {args.anim_step} steps) ...")
    step = max(1, args.anim_step)
    frame_idx = np.arange(0, roll.shape[0], step)
    body_positions = []
    for i in frame_idx:
        _, bp = forward_kinematics(roll[i, :6])
        body_positions.append(bp)
    body_positions = np.array(body_positions)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ee_ref[:, 0], ee_ref[:, 1], ee_ref[:, 2],
            color="green", ls="--", lw=1.4, label="reference")
    ax.plot(ee_traj[:, 0], ee_traj[:, 1], ee_traj[:, 2],
            color="teal", lw=0.6, alpha=0.25)
    arm_line,   = ax.plot([], [], [], "o-", color="steelblue",
                          lw=2.5, markersize=6, label="UR5 arm")
    trail_line, = ax.plot([], [], [], color="teal", lw=1.8, alpha=0.85,
                          label="ee path")
    ee_dot,     = ax.plot([], [], [], "o", color="red", markersize=8)
    title_obj   = ax.set_title(f"UR5 Variant H rollout {trial_idx}  (t = 0.00 s)",
                               fontsize=11)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.legend(loc="upper right", fontsize=8)

    all_pts = np.concatenate([ee_ref, ee_traj, body_positions.reshape(-1, 3)],
                             axis=0)
    pad = 0.1
    ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
    ax.set_zlim(all_pts[:, 2].min() - pad, all_pts[:, 2].max() + pad)
    ax.view_init(elev=22, azim=-55)

    def init():
        arm_line.set_data([], []);  arm_line.set_3d_properties([])
        trail_line.set_data([], []); trail_line.set_3d_properties([])
        ee_dot.set_data([], []);    ee_dot.set_3d_properties([])
        return arm_line, trail_line, ee_dot, title_obj

    def update(fnum):
        bp = body_positions[fnum]
        arm_line.set_data(bp[:, 0], bp[:, 1])
        arm_line.set_3d_properties(bp[:, 2])
        t_idx = frame_idx[fnum]
        trail_line.set_data(ee_traj[:t_idx + 1, 0], ee_traj[:t_idx + 1, 1])
        trail_line.set_3d_properties(ee_traj[:t_idx + 1, 2])
        ee_dot.set_data([bp[-1, 0]], [bp[-1, 1]])
        ee_dot.set_3d_properties([bp[-1, 2]])
        title_obj.set_text(
            f"UR5 Variant H rollout {trial_idx}  (t = {t_idx * T_sampling:.2f} s)")
        return arm_line, trail_line, ee_dot, title_obj

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=len(frame_idx), interval=60, blit=False)
    out_path = save_dir / "06_arm_animation.gif"
    anim.save(out_path, writer=PillowWriter(fps=15))
    plt.close(fig)
    print(f"  saved -> {out_path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("EVALUATION SUMMARY -- MC-PILCO UR5 Variant H")
print("=" * 78)
print(f"Log file:       {log_path}")
print(f"Real rollouts:  {n_hist}  (exploration + {n_hist - 1} policy)")
print(f"Policy trials:  {n_trials} completed")
print(f"Plotted:        rollout {trial_idx}")
if n_trials > 0:
    for i in range(n_trials):
        c = np.asarray(log["cost_trial_list"][i])
        print(f"  Trial {i+1:>2}: opt_steps={len(c):>4}  "
              f"start={c[0]:>10.1f}  end={c[-1]:>10.1f}  min={c.min():>10.1f}")
print(f"Tracking (rollout {trial_idx}):")
print(f"  mean RMS q error  : {pos_err_rms_joint.mean():.4f} rad "
      f"({np.rad2deg(pos_err_rms_joint.mean()):.2f} deg)")
print(f"  mean RMS dq error : {vel_err_rms_joint.mean():.4f} rad/s")
print(f"  mean EE error     : {cart_err.mean():.1f} mm  "
      f"(max {cart_err.max():.1f} mm)")
print(f"Plots saved to: {save_dir.resolve()}")
print("=" * 78)
