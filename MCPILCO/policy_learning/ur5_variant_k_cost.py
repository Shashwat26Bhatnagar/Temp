"""
UR5 Variant K (overwritten): per-particle KL between the GFN's LIVE conditional
transition and the GP's transition, with truncated BPTT.

Same formulation as the cartpole Variant K -- the GFN is queried as the
conditional transition kernel P_F(s'|s,t): the network takes the particle's
current 12-D state and the (physical->diffusion-mapped) time and OUTPUTS a
Gaussian over the next state.

1) REVERSE KL:  KL( p_GFN(.|s) || q_GP(.|s,a) )   (GFN = p, GP = q)
2) PER PARTICLE
3) TRUNCATED BPTT (middle ground) via MC_PILCO_GPStats(bptt_truncate=K), K~10
4) LIVE GFN CONDITIONAL: mu_GFN = x_k + dt*pf_mean(x_k,t),
                         var_GFN = dt*exp(pf_logvar(x_k,t)),  t = k/N_h

Requires MC_PILCO_GPStats (sets gp_means_seq / gp_vars_seq).
"""

import math

import torch

from policy_learning.ur5_gfn_prior import UR5GFNPrior
from policy_learning.ur5_chance_constraint import (
    ur5_joint_total_slack,
    UR5_Q_MIN_DEFAULT,
    UR5_Q_MAX_DEFAULT,
)
from policy_learning.kl_cost import (
    reverse_kl_gaussian_diag,
    forward_kl_gaussian_diag,
)


class UR5_VariantK_Cost:
    """
    Per-particle KL vs the GFN's LIVE conditional transition (UR5).

    Attributes set externally by MC_PILCO_GPStats.apply_policy:
        gp_means_seq : [N_h+1, M, 12]
        gp_vars_seq  : [N_h+1, M, 12]
    """

    def __init__(self,
                 checkpoint_path,
                 q_ref,
                 dq_ref,
                 T_control,
                 alpha=5.0,
                 epsilon=0.10,
                 weighting='uniform',
                 kl_direction='reverse',
                 use_state_weight=True,
                 beta=0.0,
                 u_max=None,
                 q_min=UR5_Q_MIN_DEFAULT,
                 q_max=UR5_Q_MAX_DEFAULT,
                 num_ref_samples=512,
                 dtype=torch.float64,
                 device=torch.device('cpu')):
        assert weighting in ('uniform', 'linear', 'quadratic', 'none'), \
            f"Unknown weighting '{weighting}'."
        assert kl_direction in ('forward', 'reverse'), \
            f"Unknown kl_direction '{kl_direction}'."

        self.alpha        = alpha
        self.beta         = beta
        self.epsilon      = epsilon
        self.weighting    = weighting
        self.kl_direction = kl_direction
        self.use_state_weight = use_state_weight
        self.q_min        = q_min
        self.q_max        = q_max
        self.dtype        = dtype
        self.device       = device

        if u_max is not None:
            self.u_max = torch.as_tensor(u_max, dtype=dtype, device=device)
        else:
            self.u_max = None

        self.gfn_prior = UR5GFNPrior(
            checkpoint_path=checkpoint_path,
            q_ref=q_ref, dq_ref=dq_ref, T_control=T_control,
            num_ref_samples=num_ref_samples,
            dtype=dtype, device=device,
        )
        self.gfn = self.gfn_prior.gfn_model
        self.gfn_dt = float(self.gfn.dt)
        assert not self.gfn.langevin

        self.gp_means_seq = None
        self.gp_vars_seq = None

        self.last_kl_per_step     = None
        self.last_action_per_step = None
        self.last_slack_per_step  = None
        self.last_weights         = None

        print(f"[UR5_VariantK_Cost] LIVE conditional GFN transition "
              f"(state,t) -> N(next state)")
        print(f"[UR5_VariantK_Cost] kl_direction = {kl_direction.upper()} | "
              f"alpha={alpha} beta={beta} epsilon={epsilon} "
              f"weighting='{weighting}' gfn_dt={self.gfn_dt:.4f}")
        print(f"[UR5_VariantK_Cost] state-probability weighting = "
              f"{'ON' if use_state_weight else 'OFF'}")
        self._sanity_print()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _diag_gaussian_logprob(x, mu, var):
        """log N(x | mu, diag(var)) per particle. x,mu,var: [M,D] -> [M]."""
        var = torch.clamp(var, min=1e-12)
        return -0.5 * (((x - mu) ** 2 / var)
                       + torch.log(2.0 * math.pi * var)).sum(dim=-1)

    def _dummy_logr(self, x, condition=None):
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

    def _gfn_predict(self, state_k, t_scalar):
        dtype, device = state_k.dtype, state_k.device
        with torch.no_grad():
            s32 = state_k.detach().to(dtype=torch.float32, device=device)
            pfs, _ = self.gfn.predict_next_state(s32, float(t_scalar),
                                                 self._dummy_logr)
            pf_mean, pf_logvar = self.gfn.split_params(pfs)
            mu_gfn = s32 + self.gfn_dt * pf_mean
            var_gfn = self.gfn_dt * torch.exp(pf_logvar)
        return (mu_gfn.to(dtype=dtype, device=device),
                var_gfn.to(dtype=dtype, device=device))

    def _sanity_print(self):
        z = torch.zeros(1, 12, dtype=self.dtype, device=self.device)
        for t in (0.0, 0.5, 0.95):
            mu, var = self._gfn_predict(z, t)
            q = mu[0][:6].tolist()
            sd = [v ** 0.5 for v in var[0][:6].tolist()]
            print(f"[UR5_VariantK_Cost]   GFN(zeros, t={t:.2f}) q: "
                  f"mu=[{', '.join(f'{m:+.2f}' for m in q)}]  "
                  f"sigma=[{', '.join(f'{s:.2f}' for s in sd)}]")

    # ------------------------------------------------------------------ #
    def cost_function(self, states_sequence, inputs_sequence, trial_index):
        N_h_plus_1, P, D = states_sequence.shape
        assert D == 12, f"Expected state_dim=12, got {D}"
        N_h = N_h_plus_1 - 1

        dtype  = states_sequence.dtype
        device = states_sequence.device

        assert self.gp_means_seq is not None, "Use MC_PILCO_GPStats."
        gp_means = self.gp_means_seq.to(dtype=dtype, device=device)
        gp_means = torch.nan_to_num(gp_means, nan=0.0, posinf=1e3, neginf=-1e3)
        gp_means = torch.clamp(gp_means, min=-50.0, max=50.0)
        gp_vars = self.gp_vars_seq.to(dtype=dtype, device=device)
        gp_vars = torch.nan_to_num(gp_vars, nan=1e-6, posinf=1e6, neginf=1e-12)
        gp_vars = torch.clamp(gp_vars, min=1e-8, max=1e6)

        kl_per_step = torch.zeros(N_h_plus_1, dtype=dtype, device=device)
        for k in range(N_h):
            t_diff = k / max(N_h, 1)
            mu_gfn, var_gfn = self._gfn_predict(states_sequence[k], t_diff)
            var_gfn = torch.clamp(var_gfn, min=1e-8)

            gp_mu = gp_means[k + 1]
            gp_v = gp_vars[k + 1]

            if self.kl_direction == 'reverse':
                kl = reverse_kl_gaussian_diag(mu_gfn, var_gfn, gp_mu, gp_v)
            else:
                kl = forward_kl_gaussian_diag(gp_mu, gp_v, mu_gfn, var_gfn)
            kl = torch.clamp(kl, min=0.0, max=1e6)             # [M]

            # State-probability weighting: weight each particle's KL by the
            # GP density of its CURRENT state s_k (detached sample weights).
            if self.use_state_weight:
                log_w = self._diag_gaussian_logprob(
                    states_sequence[k].detach(),
                    gp_means[k].detach(), gp_vars[k].detach())   # [M]
                w = torch.softmax(log_w, dim=0)                  # [M], sums to 1
                kl_per_step[k + 1] = (w * kl).sum()
            else:
                kl_per_step[k + 1] = kl.mean()

        # ---- Time weighting ----
        t_frac = torch.arange(N_h_plus_1, dtype=dtype, device=device) / max(N_h, 1)
        if self.weighting == 'quadratic':
            weights = t_frac ** 2
        elif self.weighting == 'linear':
            weights = t_frac
        else:
            weights = torch.ones_like(t_frac)
        weighted_kl = weights * kl_per_step

        # ---- Chance constraints ----
        slack_per_step, _ = ur5_joint_total_slack(
            states_sequence, q_min=self.q_min, q_max=self.q_max,
            epsilon=self.epsilon,
        )

        # ---- Optional action regularisation ----
        if self.beta > 0 and inputs_sequence is not None:
            if self.u_max is not None:
                u_norm = inputs_sequence / self.u_max.unsqueeze(0).unsqueeze(0)
            else:
                u_norm = inputs_sequence
            action_per_step = (u_norm ** 2).sum(dim=-1).mean(dim=1)
        else:
            action_per_step = torch.zeros(N_h_plus_1, dtype=dtype, device=device)

        cost_per_step = (weighted_kl
                         + self.alpha * slack_per_step
                         + self.beta * action_per_step)

        self.last_kl_per_step     = kl_per_step.detach()
        self.last_action_per_step = action_per_step.detach()
        self.last_slack_per_step  = slack_per_step.detach()
        self.last_weights         = weights.detach()

        return cost_per_step

    def __call__(self, states_sequence, inputs_sequence, trial_index):
        cost_per_step = self.cost_function(states_sequence,
                                           inputs_sequence, trial_index)
        mean_cost = cost_per_step.sum()
        std_cost = torch.tensor(0.0, dtype=cost_per_step.dtype,
                                device=cost_per_step.device)
        return mean_cost, std_cost
