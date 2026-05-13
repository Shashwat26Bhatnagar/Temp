import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pytest
import torch
import math

from policy_learning.gfn_prior import GFNPrior
from policy_learning.kl_cost import (
    gaussian_moments_from_particles,
    reverse_kl_gaussian_diag,
    time_weighted_kl_sum,
)
from policy_learning.chance_constraint import (
    standard_normal_inverse_cdf,
    chance_constraint_slack,
    cartpole_total_slack,
)
from policy_learning.variant_b_cost import VariantB_Cost

DTYPE = torch.float64
DEVICE = torch.device('cpu')


class TestGFNPrior:

    def test_state_dim_is_4(self):
        prior = GFNPrior(dtype=DTYPE, device=DEVICE)
        assert prior.state_dim == 4

    def test_mu_p_is_zero(self):
        prior = GFNPrior(dtype=DTYPE, device=DEVICE)
        assert torch.allclose(prior.mu_p, torch.zeros(4, dtype=DTYPE))

    def test_sigma_diagonal_values(self):
        prior = GFNPrior(dtype=DTYPE, device=DEVICE)
        expected = torch.tensor([0.25, 0.25, 0.01, 0.01], dtype=DTYPE)
        assert torch.allclose(prior.Sigma_p_diag, expected)

    def test_target_at_step_returns_correct_shapes(self):
        prior = GFNPrior(dtype=DTYPE, device=DEVICE)
        mu, Sigma = prior.get_target_at_step(30, 60)
        assert mu.shape == (4,)
        assert Sigma.shape == (4, 4)

    def test_target_invariant_for_mvp(self):
        # MVP prior is time-independent: same target at any k
        prior = GFNPrior(dtype=DTYPE, device=DEVICE)
        mu_a, Sigma_a = prior.get_target_at_step(0, 60)
        mu_b, Sigma_b = prior.get_target_at_step(60, 60)
        assert torch.allclose(mu_a, mu_b)
        assert torch.allclose(Sigma_a, Sigma_b)


class TestGaussianMoments:

    def test_moments_shape(self):
        states = torch.randn(61, 400, 4, dtype=DTYPE)
        mu, var = gaussian_moments_from_particles(states)
        assert mu.shape == (61, 4)
        assert var.shape == (61, 4)

    def test_moments_match_manual(self):
        torch.manual_seed(0)
        states = torch.randn(10, 1000, 4, dtype=DTYPE) * 0.5 + 1.0
        mu, var = gaussian_moments_from_particles(states)
        # Mean should be ~1, var should be ~0.25 (sigma=0.5)
        assert torch.allclose(mu.mean(), torch.tensor(1.0, dtype=DTYPE), atol=0.05)
        assert torch.allclose(var.mean(), torch.tensor(0.25, dtype=DTYPE), atol=0.05)

    def test_var_has_floor(self):
        # All-identical particles → var should be tiny but not zero
        states = torch.ones(5, 100, 4, dtype=DTYPE) * 2.0
        mu, var = gaussian_moments_from_particles(states)
        assert (var > 0).all()


class TestReverseKL:

    def test_kl_pp_is_zero(self):
        # KL(p || p) = 0
        mu = torch.tensor([1.0, -0.5, 0.0, 0.3], dtype=DTYPE)
        sigma_diag = torch.tensor([0.25, 0.25, 0.01, 0.01], dtype=DTYPE)
        kl = reverse_kl_gaussian_diag(mu, sigma_diag, mu, sigma_diag)
        assert torch.allclose(kl, torch.tensor(0.0, dtype=DTYPE), atol=1e-10)

    def test_kl_is_non_negative(self):
        torch.manual_seed(0)
        mu_p = torch.zeros(4, dtype=DTYPE)
        Sp = torch.tensor([0.25, 0.25, 0.01, 0.01], dtype=DTYPE)
        for _ in range(5):
            mu_q = torch.randn(4, dtype=DTYPE)
            Sq = torch.rand(4, dtype=DTYPE) + 0.1
            kl = reverse_kl_gaussian_diag(mu_p, Sp, mu_q, Sq)
            assert kl.item() >= -1e-10

    def test_kl_batched(self):
        mu_p = torch.zeros(4, dtype=DTYPE)
        Sp = torch.tensor([0.25, 0.25, 0.01, 0.01], dtype=DTYPE)
        mu_q = torch.randn(61, 4, dtype=DTYPE)
        Sq = torch.rand(61, 4, dtype=DTYPE) + 0.1
        kl = reverse_kl_gaussian_diag(mu_p, Sp, mu_q, Sq)
        assert kl.shape == (61,)

    def test_kl_increases_with_mean_distance(self):
        mu_p = torch.zeros(4, dtype=DTYPE)
        Sp = torch.tensor([0.25, 0.25, 0.01, 0.01], dtype=DTYPE)
        Sq = Sp.clone()
        kl_near = reverse_kl_gaussian_diag(mu_p, Sp,
                                            torch.tensor([0.1,0,0,0], dtype=DTYPE), Sq)
        kl_far  = reverse_kl_gaussian_diag(mu_p, Sp,
                                            torch.tensor([1.0,0,0,0], dtype=DTYPE), Sq)
        assert kl_far.item() > kl_near.item()


class TestTimeWeightedKL:

    def test_returns_correct_shapes(self):
        prior = GFNPrior(dtype=DTYPE, device=DEVICE)
        states = torch.randn(61, 400, 4, dtype=DTYPE)
        per_step, total = time_weighted_kl_sum(states, prior, 'quadratic')
        assert per_step.shape == (61,)
        assert total.dim() == 0

    def test_quadratic_emphasises_end(self):
        # weight at k=0 is 0, weight at k=N is 1
        prior = GFNPrior(dtype=DTYPE, device=DEVICE)
        states = torch.randn(61, 400, 4, dtype=DTYPE) * 2.0  # large spread
        per_step, total_q = time_weighted_kl_sum(states, prior, 'quadratic')
        per_step, total_n = time_weighted_kl_sum(states, prior, 'none')
        # Quadratic total should be less than uniform total (down-weights early)
        assert total_q.item() < total_n.item()


class TestStandardNormalInverseCDF:

    def test_phi_inv_0_5_is_zero(self):
        p = torch.tensor(0.5, dtype=DTYPE)
        assert torch.allclose(standard_normal_inverse_cdf(p),
                              torch.tensor(0.0, dtype=DTYPE), atol=1e-6)

    def test_phi_inv_at_95_percent(self):
        # Phi^-1(0.95) ≈ 1.6449
        p = torch.tensor(0.95, dtype=DTYPE)
        result = standard_normal_inverse_cdf(p).item()
        assert abs(result - 1.6449) < 0.001


class TestChanceConstraintSlack:

    def test_slack_zero_inside_bounds(self):
        # Mean 0, tiny variance, bound 2.4 → comfortably safe
        mu = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=DTYPE)
        Sigma_diag = torch.tensor([0.01]*4, dtype=DTYPE)
        h = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE)
        slack = chance_constraint_slack(mu, Sigma_diag, h, b=2.4, epsilon=0.05)
        assert slack.item() < 1e-6

    def test_slack_positive_outside_bounds(self):
        # Mean 3.0 on position, bound 2.4 → violation
        mu = torch.tensor([3.0, 0.0, 0.0, 0.0], dtype=DTYPE)
        Sigma_diag = torch.tensor([0.01]*4, dtype=DTYPE)
        h = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE)
        slack = chance_constraint_slack(mu, Sigma_diag, h, b=2.4, epsilon=0.05)
        assert slack.item() > 0.5

    def test_slack_grows_with_variance(self):
        # Mean at boundary, but variance growing → slack should grow
        mu = torch.tensor([2.0, 0.0, 0.0, 0.0], dtype=DTYPE)
        h = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE)
        S_low  = torch.tensor([0.01]*4, dtype=DTYPE)
        S_high = torch.tensor([0.5]*4,  dtype=DTYPE)
        slack_low  = chance_constraint_slack(mu, S_low,  h, 2.4, 0.05)
        slack_high = chance_constraint_slack(mu, S_high, h, 2.4, 0.05)
        assert slack_high.item() > slack_low.item()


class TestCartPoleTotalSlack:

    def test_shapes(self):
        states = torch.zeros(61, 400, 4, dtype=DTYPE)
        per_step, total = cartpole_total_slack(states)
        assert per_step.shape == (61,)
        assert total.dim() == 0

    def test_zero_when_at_origin(self):
        states = torch.zeros(61, 400, 4, dtype=DTYPE)
        # Add small noise so variance isn't exactly 0
        states = states + torch.randn_like(states) * 0.01
        per_step, total = cartpole_total_slack(states)
        # Should be tiny — comfortably within bounds
        assert total.item() < 1.0

    def test_positive_when_far_outside(self):
        # All particles at angle = pi/2 (way past 12 deg bound)
        states = torch.zeros(61, 400, 4, dtype=DTYPE)
        states[:, :, 2] = math.pi / 2
        per_step, total = cartpole_total_slack(states)
        assert total.item() > 10.0


class TestVariantBCost:

    def test_cost_function_interface(self):
        # Matches Cart_pole_cost interface: returns [num_instants]
        cost = VariantB_Cost(dtype=DTYPE, device=DEVICE)
        states = torch.randn(61, 400, 4, dtype=DTYPE)
        inputs = torch.randn(61, 400, 1, dtype=DTYPE)
        result = cost.cost_function(states, inputs, trial_index=0)
        assert result.shape == (61,)

    def test_cost_is_finite(self):
        cost = VariantB_Cost(dtype=DTYPE, device=DEVICE)
        states = torch.randn(61, 400, 4, dtype=DTYPE)
        inputs = torch.randn(61, 400, 1, dtype=DTYPE)
        result = cost.cost_function(states, inputs, trial_index=0)
        assert torch.isfinite(result).all()

    def test_backward_works(self):
        # Critical: gradients must flow back through the cost
        cost = VariantB_Cost(dtype=DTYPE, device=DEVICE)
        states = torch.randn(61, 400, 4, dtype=DTYPE, requires_grad=True)
        inputs = torch.randn(61, 400, 1, dtype=DTYPE, requires_grad=True)
        result = cost.cost_function(states, inputs, trial_index=0)
        loss = result.sum()
        loss.backward()
        assert states.grad is not None
        assert torch.isfinite(states.grad).all()

    def test_alpha_changes_cost(self):
        # Higher alpha should produce higher total cost when constraints are violated
        states = torch.zeros(61, 400, 4, dtype=DTYPE)
        states[:, :, 0] = 3.0   # position violates bound
        inputs = torch.zeros(61, 400, 1, dtype=DTYPE)

        cost_low  = VariantB_Cost(alpha=1.0,  dtype=DTYPE, device=DEVICE)
        cost_high = VariantB_Cost(alpha=100.0, dtype=DTYPE, device=DEVICE)

        r_low  = cost_low.cost_function(states, inputs, 0).sum().item()
        r_high = cost_high.cost_function(states, inputs, 0).sum().item()
        assert r_high > r_low
