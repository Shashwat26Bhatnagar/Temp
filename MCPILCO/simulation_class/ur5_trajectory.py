"""
Circular reference trajectory generator for UR5 MC-PILCO experiments.

Matches MC-PILCO paper Section VI-D setup:
  - Circle in the X-Y plane (constant Z height)
  - Numerical IK to convert Cartesian -> joint angles
  - Returns q_ref(t), dq_ref(t) sampled at dt = 0.02 s

Inverse kinematics: Jacobian pseudo-inverse with damped least-squares,
iterated until convergence (no external IK library required).
"""

import sys
import pathlib
import numpy as np
import mujoco

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from simulation_class.ur5_system import UR5System


# -----------------------------------------------------------------------
# Numerical IK via damped Jacobian pseudo-inverse
# -----------------------------------------------------------------------

def _ik_solve(env, target_pos, q_init=None, max_iter=500, tol=1e-4, damp=1e-3):
    """
    Solve IK for a 3-D end-effector position target using the
    Jacobian pseudo-inverse method.

    Args:
        env:        UR5System instance
        target_pos: [3] desired (x, y, z) in world frame
        q_init:     [6] starting joint angles (uses env's current state if None)
        max_iter:   maximum iterations
        tol:        position error tolerance (m)
        damp:       damping factor lambda for damped least-squares

    Returns:
        q_sol:   [6] joint angles (rad)
        success: bool
    """
    m = env.model
    d = env.data

    if q_init is not None:
        d.qpos[:] = q_init
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

    ee_id = env._ee_site_id
    jacp = np.zeros((3, m.nv))   # position Jacobian  [3, nv]
    jacr = np.zeros((3, m.nv))   # rotation Jacobian  [3, nv] (unused)

    for _ in range(max_iter):
        mujoco.mj_forward(m, d)
        pos_cur = d.site_xpos[ee_id].copy()
        err = target_pos - pos_cur
        if np.linalg.norm(err) < tol:
            break

        mujoco.mj_jacSite(m, d, jacp, jacr, ee_id)
        # Damped least-squares:  dq = J^T (J J^T + λ²I)^-1 err
        JJT = jacp @ jacp.T + damp ** 2 * np.eye(3)
        dq  = jacp.T @ np.linalg.solve(JJT, err)

        # Clip step size and apply
        dq   = np.clip(dq, -0.1, 0.1)
        q_new = d.qpos + dq
        # Clip to joint limits
        q_new = np.clip(q_new, m.jnt_range[:, 0], m.jnt_range[:, 1])
        d.qpos[:] = q_new

    mujoco.mj_forward(m, d)
    final_err = np.linalg.norm(target_pos - d.site_xpos[ee_id])
    success   = final_err < tol * 10   # relaxed acceptance
    return d.qpos.copy(), success


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def generate_circle_trajectory(
    center=(-0.6, 0.0, 0.4),
    radius=0.15,
    duration=4.0,
    dt=0.02,
    env=None,
):
    """
    Generate a circular end-effector trajectory in the X-Y plane and
    convert it to joint-space via numerical IK.

    Args:
        center:   (x, y, z) circle centre in world frame (m)
        radius:   circle radius (m)
        duration: trajectory duration (s)
        dt:       control time step (s)
        env:      UR5System instance (created internally if None)

    Returns:
        t_arr:   [N]     time stamps (s)
        q_ref:   [N, 6]  joint position reference (rad)
        dq_ref:  [N, 6]  joint velocity reference (rad/s, finite-difference)
        ee_ref:  [N, 3]  Cartesian end-effector reference (m)
    """
    if env is None:
        env = UR5System()

    cx, cy, cz = center
    N = int(round(duration / dt)) + 1
    t_arr  = np.linspace(0.0, duration, N)

    # Cartesian waypoints on the circle
    angles = 2.0 * np.pi * t_arr / duration   # 0 → 2π
    ee_ref = np.column_stack([
        cx + radius * np.cos(angles),
        cy + radius * np.sin(angles),
        np.full(N, cz),
    ])

    # Solve IK for each waypoint, seeding from the previous solution
    q_ref = np.zeros((N, 6))
    # Seed: solve IK for the first point from the home pose
    q_seed = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

    print(f"[ur5_trajectory] Solving IK for {N} waypoints ...")
    n_fail = 0
    for i, pos in enumerate(ee_ref):
        q_sol, ok = _ik_solve(env, pos, q_init=q_seed, tol=5e-4)
        if not ok:
            n_fail += 1
        q_ref[i]  = q_sol
        q_seed    = q_sol     # warm-start next step

    if n_fail > 0:
        print(f"  Warning: {n_fail}/{N} IK solutions did not fully converge "
              f"(position error < 5 mm accepted).")
    else:
        print(f"  All {N} IK solutions converged.")

    # Numerical differentiation for dq_ref
    dq_ref = np.gradient(q_ref, dt, axis=0)

    # Restore env state
    env.reset()

    return t_arr, q_ref, dq_ref, ee_ref
