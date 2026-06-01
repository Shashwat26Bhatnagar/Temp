"""
MC_PILCO subclass that exposes per-particle GP prediction statistics
(mean and variance) to the cost function.

Standard MC_PILCO.apply_policy calls get_next_state() which returns
(next_states, delta_mean, delta_var) -- but the delta_mean and delta_var
are discarded.  This subclass captures them, reconstructs the full-state
GP prediction Gaussian for each particle, and stores the result on the
cost function object so that Variant-J-style per-particle KL costs can
use the exact GP Gaussians rather than a moment-matched aggregate.

Also supports optional BPTT truncation (same as MC_PILCO_Local).

Usage:
    PL_obj = MC_PILCO_GPStats(bptt_truncate=None, **MC_PILCO_init_dict)
    # cost_function.gp_means_seq and .gp_vars_seq are automatically set
    # after every apply_policy call.
"""

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform

import policy_learning.MC_PILCO as MC_PILCO_module


class MC_PILCO_GPStats(MC_PILCO_module.MC_PILCO):
    """
    MC_PILCO with per-particle GP statistics passed to the cost function.

    After each apply_policy call, the cost function object will have:
        cost_function.gp_means_seq : [N_h+1, M, D]  per-particle GP mean
        cost_function.gp_vars_seq  : [N_h+1, M, D]  per-particle GP variance (diagonal)

    where
        t=0:  initial-distribution parameters (same for all particles)
        t>=1: reconstructed from GP delta prediction + current particle state
    """

    def __init__(self, *args, bptt_truncate=None, **kwargs):
        super().__init__(*args, **kwargs)
        if bptt_truncate is not None:
            assert isinstance(bptt_truncate, int) and bptt_truncate >= 1, \
                f"bptt_truncate must be a positive int or None, got {bptt_truncate}"
        self.bptt_truncate = bptt_truncate
        if bptt_truncate is not None:
            print(f"[MC_PILCO_GPStats] BPTT truncation ACTIVE; "
                  f"detaching state every {bptt_truncate} step(s).")
        else:
            print(f"[MC_PILCO_GPStats] full BPTT (no truncation).")

    # ------------------------------------------------------------------ #
    # Convert GP delta outputs to full next-state mean & variance         #
    # ------------------------------------------------------------------ #
    def _gp_delta_to_full_state(self, current_state, delta_mean, delta_var):
        """
        Reconstruct the GP's Gaussian prediction for the *next full state*
        from the delta (or velocity-delta) GP outputs.

        For direct-delta models (cartpole):
            next_state ~ N(current + delta_mean, delta_var)

        For speed models (UR5):
            GP predicts velocity delta dv ~ N(delta_mean, delta_var).
            next_vel  = current_vel + dv       -> var = delta_var
            next_pos  = current_pos + T_s * current_vel + T_s/2 * dv
                                               -> var = (T_s/2)^2 * delta_var

        Returns:
            gp_mean: [M, D]  mean of the GP prediction for the full next state
            gp_var:  [M, D]  diagonal variance of that prediction
        """
        ml = self.model_learning

        # Sanitise delta inputs (GP can produce NaN/Inf when extrapolating)
        delta_mean = torch.nan_to_num(delta_mean, nan=0.0, posinf=1e4, neginf=-1e4)
        delta_var = torch.nan_to_num(delta_var, nan=1e-6, posinf=1e6, neginf=1e-12)
        delta_var = torch.clamp(delta_var, min=1e-12, max=1e6)

        if hasattr(ml, 'vel_indeces') and hasattr(ml, 'not_vel_indeces'):
            # Speed model (UR5): delta is over velocities only
            T_s = ml.T_sampling
            gp_mean = torch.zeros_like(current_state)
            gp_var = torch.zeros_like(current_state)

            # Velocities: v_{t+1} = v_t + delta
            gp_mean[:, ml.vel_indeces] = (
                current_state[:, ml.vel_indeces] + delta_mean
            )
            gp_var[:, ml.vel_indeces] = delta_var

            # Positions: q_{t+1} = q_t + T_s*v_t + (T_s/2)*delta
            gp_mean[:, ml.not_vel_indeces] = (
                current_state[:, ml.not_vel_indeces]
                + T_s * current_state[:, ml.vel_indeces]
                + (T_s / 2.0) * delta_mean
            )
            gp_var[:, ml.not_vel_indeces] = (T_s / 2.0) ** 2 * delta_var
        else:
            # Direct delta model (cartpole): next = current + delta
            gp_mean = current_state + delta_mean
            gp_var = delta_var

        return gp_mean, gp_var

    # ------------------------------------------------------------------ #
    # Override apply_policy to capture GP statistics                       #
    # ------------------------------------------------------------------ #
    def apply_policy(self,
                     particles_initial_state_mean,
                     particles_initial_state_var,
                     flg_particles_init_uniform,
                     particles_init_up_bound,
                     particles_init_low_bound,
                     flg_particles_init_multi_gauss,
                     num_particles,
                     T_control,
                     p_dropout=0.0):

        states_sequence_list = []
        inputs_sequence_list = []
        gp_means_list = []
        gp_vars_list = []

        # ---- Initial particle distribution (identical to parent) ----
        if flg_particles_init_uniform:
            ub = particles_init_up_bound.repeat(num_particles, 1)
            lb = particles_init_low_bound.repeat(num_particles, 1)
            state_distribution = Uniform(lb, ub)
            # Approximate Gaussian stats for uniform: mean=(ub+lb)/2, var=(ub-lb)^2/12
            _init_mean = (particles_init_up_bound + particles_init_low_bound) / 2.0
            _init_var = (particles_init_up_bound - particles_init_low_bound) ** 2 / 12.0
        elif flg_particles_init_multi_gauss:
            indices = torch.randint(0,
                                    particles_initial_state_mean.shape[0],
                                    [num_particles])
            init_mean = particles_initial_state_mean[indices, :]
            init_cov = torch.stack([
                torch.diag(particles_initial_state_var[i, :]) for i in indices])
            state_distribution = MultivariateNormal(loc=init_mean,
                                                    covariance_matrix=init_cov)
            # For multi-Gaussian: use per-particle component mean/var
            _init_mean = particles_initial_state_mean[indices, :]  # [M, D]
            _init_var = particles_initial_state_var[indices, :]    # [M, D]
        else:
            init_mean = particles_initial_state_mean.repeat(num_particles, 1)
            init_cov = torch.stack(
                [torch.diag(particles_initial_state_var)] * num_particles)
            state_distribution = MultivariateNormal(loc=init_mean,
                                                    covariance_matrix=init_cov)
            _init_mean = particles_initial_state_mean.unsqueeze(0).expand(
                num_particles, -1)
            _init_var = particles_initial_state_var.unsqueeze(0).expand(
                num_particles, -1)

        # t = 0: sample from initial distribution
        states_sequence_list.append(state_distribution.rsample())
        inputs_sequence_list.append(
            self.control_policy(states_sequence_list[0], t=0,
                                p_dropout=p_dropout)
        )

        # GP stats at t=0: initial distribution (no GP prediction yet)
        gp_means_list.append(_init_mean.clone())
        gp_vars_list.append(torch.clamp(_init_var.clone(), min=1e-12))

        K = self.bptt_truncate

        # t = 1 .. T_control-1
        for t in range(1, int(T_control)):
            prev_state = states_sequence_list[t - 1]
            prev_input = inputs_sequence_list[t - 1]

            # Optional BPTT truncation
            if K is not None and ((t - 1) % K == 0):
                prev_state = prev_state.detach()
                prev_input = self.control_policy(prev_state, t=t - 1,
                                                  p_dropout=p_dropout)

            # Get next state + GP delta prediction
            particles, delta_mean, delta_var = self.model_learning.get_next_state(
                current_state=prev_state,
                current_input=prev_input,
            )

            # Reconstruct full-state GP prediction from delta outputs
            gp_mean, gp_var = self._gp_delta_to_full_state(
                prev_state, delta_mean, delta_var
            )
            gp_means_list.append(gp_mean)
            gp_vars_list.append(gp_var)

            states_sequence_list.append(particles)
            inputs_sequence_list.append(
                self.control_policy(states_sequence_list[t], t=t,
                                    p_dropout=p_dropout)
            )

        # Stack into tensors: [N_h+1, M, D]
        gp_means_seq = torch.stack(gp_means_list)
        gp_vars_seq = torch.stack(gp_vars_list)

        # Store on cost function for Variant-J to read
        self.cost_function.gp_means_seq = gp_means_seq
        self.cost_function.gp_vars_seq = gp_vars_seq

        return (torch.stack(states_sequence_list),
                torch.stack(inputs_sequence_list))
