"""
Drop-in replacement for MC-PILCO's simulation_class.model.Model that uses
the MuJoCo UR5System instead of scipy.odeint.

MC-PILCO calls `self.system.rollout(s0, policy, T, dt, noise)`.
That signature is preserved here.
"""

import numpy as np

from simulation_class.ur5_system import UR5System


class UR5_Model:
    """
    Same interface as simulation_class.model.Model but advances the state
    via MuJoCo physics steps. Each control step integrates T_sampling
    worth of physics (multiple internal substeps -- handled by UR5System).
    """

    def __init__(self, env=None):
        self.env = env if env is not None else UR5System()

    def rollout(self, s0, policy, T, dt, noise):
        """
        Roll out the real (MuJoCo) UR5 dynamics for T seconds.

        Args:
            s0:     [12] initial state [q (6), dq (6)]
            policy: callable policy(state, t) -> tau [6]
            T:      total trajectory duration (s)
            dt:     control time step (s) -- must equal UR5System.CONTROL_DT (0.02)
            noise:  measurement noise std (broadcast over state dims)

        Returns:
            noisy_states: [N+1, 12]  states + Gaussian noise
            inputs:       [N+1, 6]   torques sent to the simulator
            states:       [N+1, 12]  clean states (no noise)
        """
        state_dim = len(s0)
        s0 = np.asarray(s0, dtype=np.float64).flatten()

        N = int(round(T / dt)) + 1
        states       = np.zeros((N, state_dim))
        noisy_states = np.zeros((N, state_dim))
        # We will discover num_inputs from the first policy call.

        # Reset MuJoCo to s0
        self.env.reset(q_init=s0[:6], dq_init=s0[6:])
        states[0, :] = self.env.get_state()
        noisy_states[0, :] = states[0, :] + np.random.randn(state_dim) * noise

        # Find input size from the first policy call
        u0 = np.array(policy(noisy_states[0, :], 0.0)).flatten()
        num_inputs = u0.size
        inputs = np.zeros((N, num_inputs))
        inputs[0, :] = u0

        for k in range(N - 1):
            t = k * dt
            u = np.array(policy(noisy_states[k, :], t)).flatten()
            inputs[k, :] = u
            next_state = self.env.step(u)
            states[k + 1, :]       = next_state
            noisy_states[k + 1, :] = next_state + np.random.randn(state_dim) * noise

        # Last action (for shape compatibility)
        inputs[-1, :] = np.array(policy(noisy_states[-1, :], (N - 1) * dt)).flatten()

        return noisy_states, inputs, states
