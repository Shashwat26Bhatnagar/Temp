"""
UR5 6-DOF MuJoCo simulation environment for MC-PILCO.

State  x  = [q1..q6, dq1..dq6]  — 12-dimensional
Action u  = [tau1..tau6]         — 6 joint torques (Nm)
dt        = 0.02 s               — control frequency (50 Hz)

The MuJoCo physics step runs at 0.002 s internally;
each call to step() integrates 10 physics steps to reach 0.02 s.
"""

import os
import pathlib

import mujoco
import numpy as np


# Number of internal physics sub-steps per control step
_PHYSICS_DT = 0.002       # MuJoCo timestep (set in XML)
_CONTROL_DT = 0.02        # MC-PILCO timestep (50 Hz)
_N_SUBSTEPS  = int(round(_CONTROL_DT / _PHYSICS_DT))   # = 10

# Path to XML model — same directory as this file
_XML_PATH = pathlib.Path(__file__).parent / "ur5_model.xml"


class UR5System:
    """
    Thin wrapper around MuJoCo UR5 model, matching the interface
    used by MC-PILCO's simulation_class (see ode_systems.py style).

    State layout (12-D):
        x[0:6]  = joint positions  q  (rad)
        x[6:12] = joint velocities dq (rad/s)

    Action layout (6-D):
        u = [tau1, tau2, tau3, tau4, tau5, tau6]  (Nm)
    """

    STATE_DIM  = 12
    INPUT_DIM  = 6
    CONTROL_DT = _CONTROL_DT

    def __init__(self, xml_path=None):
        path = str(xml_path or _XML_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(f"UR5 XML not found: {path}")

        self.model = mujoco.MjModel.from_xml_path(path)
        self.data  = mujoco.MjData(self.model)

        # Cache end-effector site id
        self._ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector"
        )
        assert self._ee_site_id >= 0, "end_effector site not found in XML"

        # Joint limits [nq, 2]
        self.q_min = self.model.jnt_range[:, 0].copy()
        self.q_max = self.model.jnt_range[:, 1].copy()

        # Torque limits [nu]
        self.tau_min = self.model.actuator_ctrlrange[:, 0].copy()
        self.tau_max = self.model.actuator_ctrlrange[:, 1].copy()

        print(f"[UR5System] Loaded '{path}'")
        print(f"  state_dim={self.STATE_DIM}, input_dim={self.INPUT_DIM}, "
              f"control_dt={self.CONTROL_DT}s ({_N_SUBSTEPS} substeps)")

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self, q_init=None, dq_init=None):
        """
        Reset simulation to given initial joint angles and velocities.

        Args:
            q_init:  [6] joint positions (rad). Defaults to home pose.
            dq_init: [6] joint velocities (rad/s). Defaults to zeros.

        Returns:
            x: [12] initial state vector
        """
        mujoco.mj_resetData(self.model, self.data)

        if q_init is None:
            # UR5 "ready" pose: shoulder lifted, arm out
            q_init = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
        if dq_init is None:
            dq_init = np.zeros(6)

        self.data.qpos[:] = q_init
        self.data.qvel[:] = dq_init
        self.data.ctrl[:] = 0.0

        # Forward kinematics to populate site positions
        mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    def step(self, tau):
        """
        Apply torque command for one control step (0.02 s).

        Args:
            tau: [6] desired joint torques (Nm)

        Returns:
            x_next: [12] state after the step
        """
        tau = np.clip(tau, self.tau_min, self.tau_max)
        self.data.ctrl[:] = tau
        for _ in range(_N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        return self.get_state()

    def get_state(self):
        """Return current 12-D state [q, dq]."""
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

    def get_end_effector_pos(self):
        """
        Return (x, y, z) Cartesian position of the end-effector site
        in world frame (meters).
        """
        # site_xpos is populated after mj_step / mj_forward
        return self.data.site_xpos[self._ee_site_id].copy()

    # ------------------------------------------------------------------
    # MC-PILCO callable interface (matches ode_systems.py convention)
    # ------------------------------------------------------------------

    def __call__(self, state, t, u):
        """
        MC-PILCO calls f_sim(state, t, u) to advance the simulation.
        Here we ignore t (MuJoCo tracks time internally) and delegate
        to step().  Returns the next state.
        """
        return self.step(u)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def forward_kinematics(self, q):
        """
        Compute end-effector position for arbitrary joint angles q
        without advancing simulation time.

        Args:
            q: [6] joint positions (rad)

        Returns:
            pos: [3] end-effector (x, y, z) in world frame
        """
        old_q   = self.data.qpos.copy()
        old_dq  = self.data.qvel.copy()
        old_t   = self.data.time

        self.data.qpos[:] = q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        pos = self.data.site_xpos[self._ee_site_id].copy()

        # Restore
        self.data.qpos[:] = old_q
        self.data.qvel[:] = old_dq
        self.data.time    = old_t
        mujoco.mj_forward(self.model, self.data)
        return pos

    def joint_names(self):
        return [self.model.joint(i).name for i in range(self.model.njnt)]

    def __repr__(self):
        return (f"UR5System(state_dim={self.STATE_DIM}, "
                f"input_dim={self.INPUT_DIM}, dt={self.CONTROL_DT}s)")
