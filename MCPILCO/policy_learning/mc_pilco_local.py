"""
MC_PILCO subclass with optional BPTT truncation in the particle rollout.

Standard MC_PILCO.apply_policy builds the full rollout

    x_0 -> u_0 -> x_1 -> u_1 -> x_2 -> ... -> x_{N_h}

as a single PyTorch graph, so dL/dtheta backprops through every GP
transition and every policy evaluation. For long horizons (UR5 has N_h=200)
this is the dominant cause of vanishing / noisy gradients.

This subclass adds one knob:

    bptt_truncate : int or None
        None  -> full BPTT (identical behaviour to MC_PILCO)
        K (int >= 1) -> detach state every K steps, so each x_k depends
        on theta only through at most K policy evaluations.
        K = 1 -> pure 1-step lookahead: NO backprop through the GP chain.

Combine with UR5_VariantH_Cost (which provides a per-step LOCAL Mahalanobis
target) for fully local credit assignment. Each step's cost gradient flows
back through at most K policy evaluations, period.
"""

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform

import policy_learning.MC_PILCO as MC_PILCO_module


class MC_PILCO_Local(MC_PILCO_module.MC_PILCO):
    """
    MC_PILCO with periodic state detach during policy rollout.

    The rest of the API (reinforce, load_model_from_log, etc.) is unchanged.
    """

    def __init__(self, *args, bptt_truncate=None, **kwargs):
        """
        Args:
            bptt_truncate: None | int.
                None: same as MC_PILCO (full BPTT).
                K: detach state at every step t with (t-1) % K == 0
                   AND t > 0, capping gradient chain length at ~K.
                K=1 disables all gradient flow through the GP chain.
        """
        super().__init__(*args, **kwargs)
        if bptt_truncate is not None:
            assert isinstance(bptt_truncate, int) and bptt_truncate >= 1, \
                f"bptt_truncate must be a positive int or None, got {bptt_truncate}"
        self.bptt_truncate = bptt_truncate
        if bptt_truncate is not None:
            print(f"[MC_PILCO_Local] BPTT truncation ACTIVE; "
                  f"detaching state every {bptt_truncate} step(s) during rollout.")
        else:
            print(f"[MC_PILCO_Local] full BPTT (no truncation).")

    # ---------------------------------------------------------------- #
    # Override apply_policy with optional periodic state detach.
    # ---------------------------------------------------------------- #
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

        # ---- Initial particle distribution (identical to parent) ----
        if flg_particles_init_uniform:
            ub = particles_init_up_bound.repeat(num_particles, 1)
            lb = particles_init_low_bound.repeat(num_particles, 1)
            state_distribution = Uniform(lb, ub)
        elif flg_particles_init_multi_gauss:
            indices = torch.randint(0,
                                    particles_initial_state_mean.shape[0],
                                    [num_particles])
            init_mean = particles_initial_state_mean[indices, :]
            init_cov  = torch.stack([
                torch.diag(particles_initial_state_var[i, :]) for i in indices])
            state_distribution = MultivariateNormal(loc=init_mean,
                                                    covariance_matrix=init_cov)
        else:
            init_mean = particles_initial_state_mean.repeat(num_particles, 1)
            init_cov  = torch.stack(
                [torch.diag(particles_initial_state_var)] * num_particles)
            state_distribution = MultivariateNormal(loc=init_mean,
                                                    covariance_matrix=init_cov)

        # t = 0
        states_sequence_list.append(state_distribution.rsample())
        inputs_sequence_list.append(
            self.control_policy(states_sequence_list[0], t=0,
                                p_dropout=p_dropout)
        )

        K = self.bptt_truncate

        # t = 1 .. T_control-1
        for t in range(1, int(T_control)):
            # Choose whether to detach the previous step for the GP input.
            # Detaching the previous state breaks the BPTT chain.
            prev_state = states_sequence_list[t - 1]
            prev_input = inputs_sequence_list[t - 1]

            if K is not None and ((t - 1) % K == 0):
                # NB: we detach BOTH the state and the corresponding input;
                # the input was computed from the previous-step state via
                # the policy. We then re-emit a fresh policy evaluation on
                # the detached state so the local gradient through the
                # policy is preserved for the next K-step window.
                prev_state = prev_state.detach()
                prev_input = self.control_policy(prev_state, t=t - 1,
                                                 p_dropout=p_dropout)

            particles, _, _ = self.model_learning.get_next_state(
                current_state=prev_state,
                current_input=prev_input,
            )
            states_sequence_list.append(particles)
            inputs_sequence_list.append(
                self.control_policy(states_sequence_list[t], t=t,
                                    p_dropout=p_dropout)
            )

        return (torch.stack(states_sequence_list),
                torch.stack(inputs_sequence_list))
