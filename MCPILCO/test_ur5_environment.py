"""
UR5 environment smoke-test for MC-PILCO integration.

Runs:
  1. Load UR5 model
  2. Generate circular reference trajectory (IK)
  3. Roll out 4 seconds with random torques
  4. Plot joint angles, 3-D end-effector trajectory, tracking error
  5. Print readiness confirmation

Usage:
    python test_ur5_environment.py
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (registers 3-D projection)

from simulation_class.ur5_system     import UR5System
from simulation_class.ur5_trajectory import generate_circle_trajectory

# -----------------------------------------------------------------------
# 1. Load model
# -----------------------------------------------------------------------
print("=" * 60)
print("UR5 Environment Test for MC-PILCO")
print("=" * 60)

env = UR5System()
print(f"\n[1] Model loaded: {env}")
print(f"    Joint names : {env.joint_names()}")
print(f"    Torque lims : {env.tau_min} to {env.tau_max} Nm")

# -----------------------------------------------------------------------
# 2. Generate reference trajectory
# -----------------------------------------------------------------------
print("\n[2] Generating circular reference trajectory ...")
CENTER   = (-0.6, 0.0, 0.4)
RADIUS   = 0.15
DURATION = 4.0
DT       = 0.02

t_arr, q_ref, dq_ref, ee_ref = generate_circle_trajectory(
    center=CENTER, radius=RADIUS, duration=DURATION, dt=DT, env=env)

N = len(t_arr)
print(f"    Waypoints : {N}  (dt={DT}s, T={DURATION}s)")
print(f"    Circle centre : {CENTER}  radius : {RADIUS} m")

# -----------------------------------------------------------------------
# 3. Roll out with random torques
# -----------------------------------------------------------------------
print("\n[3] Rolling out with random torques for 4 s ...")
np.random.seed(42)

# Start near trajectory start
x0 = env.reset(q_init=q_ref[0].copy(), dq_init=np.zeros(6))

q_hist   = np.zeros((N, 6))
dq_hist  = np.zeros((N, 6))
ee_hist  = np.zeros((N, 3))
tau_hist = np.zeros((N, 6))

state = x0
for k in range(N):
    q_hist[k]  = state[:6]
    dq_hist[k] = state[6:]
    ee_hist[k] = env.get_end_effector_pos()

    # Random torque — small amplitude so arm doesn't fling wildly
    tau = np.random.uniform(-5, 5, size=6)
    tau_hist[k] = tau

    if k < N - 1:
        state = env.step(tau)

tracking_error = np.linalg.norm(ee_hist - ee_ref, axis=1)   # [N]
print(f"    Mean tracking error (random policy): {tracking_error.mean():.4f} m")
print(f"    Max  tracking error (random policy): {tracking_error.max():.4f} m")

# -----------------------------------------------------------------------
# 4. Plots
# -----------------------------------------------------------------------
print("\n[4] Generating plots ...")

SAVE_DIR = pathlib.Path("results_ur5_test")
SAVE_DIR.mkdir(exist_ok=True)

JOINT_LABELS = [
    "shoulder_pan", "shoulder_lift", "elbow",
    "wrist_1",      "wrist_2",       "wrist_3"
]

# -- Plot A: joint angles over time -----------------------------------
fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
axes = axes.flatten()
fig.suptitle("UR5 Joint Angles — Random Torques vs Reference", fontweight="bold")

for i in range(6):
    ax = axes[i]
    ax.plot(t_arr, np.degrees(q_hist[:, i]),  color="crimson",   lw=1.5, label="actual")
    ax.plot(t_arr, np.degrees(q_ref[:, i]),   color="steelblue", lw=1.5, ls="--", label="reference")
    ax.set_ylabel("deg")
    ax.set_title(JOINT_LABELS[i], fontsize=9)
    ax.grid(True, alpha=0.3)
    if i >= 4:
        ax.set_xlabel("Time (s)")
    if i == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
pA = SAVE_DIR / "A_joint_angles.png"
fig.savefig(pA, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"    Saved: {pA}")

# -- Plot B: 3-D end-effector trajectory ------------------------------
fig = plt.figure(figsize=(9, 7))
ax3 = fig.add_subplot(111, projection="3d")
ax3.plot(ee_ref[:,0],  ee_ref[:,1],  ee_ref[:,2],
         color="steelblue", lw=2, label="Reference circle")
ax3.plot(ee_hist[:,0], ee_hist[:,1], ee_hist[:,2],
         color="crimson",   lw=1.2, alpha=0.7, label="Actual (random torques)")
ax3.scatter(*ee_ref[0],  c="green", s=60, zorder=5, label="Start")
ax3.scatter(*ee_ref[-1], c="orange", s=60, zorder=5, label="End")
ax3.set_xlabel("X (m)")
ax3.set_ylabel("Y (m)")
ax3.set_zlabel("Z (m)")
ax3.set_title("3-D End-Effector Trajectory", fontweight="bold")
ax3.legend(fontsize=8)
fig.tight_layout()
pB = SAVE_DIR / "B_ee_trajectory_3d.png"
fig.savefig(pB, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"    Saved: {pB}")

# -- Plot C: Tracking error over time ---------------------------------
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t_arr, tracking_error * 100, color="darkorange", lw=1.8)
ax.fill_between(t_arr, 0, tracking_error * 100, alpha=0.2, color="darkorange")
ax.axhline(tracking_error.mean() * 100, color="red", ls="--", lw=1.2,
           label=f"Mean = {tracking_error.mean()*100:.1f} cm")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Tracking error (cm)")
ax.set_title("End-Effector Distance from Reference Trajectory\n"
             "(Random torques — this is the baseline BEFORE MC-PILCO training)",
             fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
pC = SAVE_DIR / "C_tracking_error.png"
fig.savefig(pC, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"    Saved: {pC}")

# -----------------------------------------------------------------------
# 5. Integration points for MC-PILCO
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("INTEGRATION GUIDE: MC-PILCO + UR5")
print("=" * 60)

print("""
[A] State / input dimensions
    state_dim    = 12   (q1..q6, dq1..dq6)
    input_dim    = 6    (tau1..tau6, Nm)
    T_sampling   = 0.02 s
    T_control    = 4.0  s   -> N_h = 200 timesteps

[B] Model learning
    num_gp       = 6    (one GP per JOINT ACCELERATION output,
                         matching 'Speed_Model_learning' style)
    gp_input_dim = 18   (q:6  +  dq:6  +  u:6)
    Use: f_model_learning = ML.Speed_Model_learning_RBF_MPK
         (no angle_state needed unless joints wrap)

[C] Cost function  (MC-PILCO paper eq. used for manipulator tasks)

    c(x_t) = 1 - exp( -||q_ref(t) - q(t)||^2 / (2 * length_scale^2) )

    Typical length_scale = 0.1 rad  (1 radian error -> cost ~0.39)
    The reference q_ref(t) must be precomputed and passed at each step.

    Implementation sketch:
        class UR5TrackingCost:
            def __init__(self, q_ref, length_scale=0.1):
                self.q_ref = q_ref          # [N_h+1, 6]
                self.ls2   = length_scale**2
            def __call__(self, states, inputs, trial_idx):
                # states: [N_h+1, P, 12]
                q_part  = states[:, :, :6]  # [N_h+1, P, 6]
                q_ref_t = torch.tensor(self.q_ref).unsqueeze(1)  # [N_h+1,1,6]
                err_sq  = ((q_part - q_ref_t)**2).sum(dim=-1)    # [N_h+1, P]
                cost_ps = (1 - torch.exp(-err_sq / (2*self.ls2))).mean(dim=1)
                return cost_ps, cost_ps.std()

[D] Policy
    f_control_policy = Policy.Sum_of_gaussians
    state_dim        = 12
    input_dim        = 6
    No angle wrapping needed (joint space, not end-effector space)

[E] Exploration
    u_max = 5.0 Nm   (low torques for safe exploration)
    Random exploration first trial, same as CartPole.
""")

print("UR5 environment setup complete. Ready for MC-PILCO integration.")
print(f"\nPlots saved to: {SAVE_DIR.resolve()}/")
print("  A_joint_angles.png      — joint angle trajectories")
print("  B_ee_trajectory_3d.png  — 3-D end-effector path")
print("  C_tracking_error.png    — distance from reference circle")
