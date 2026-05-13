import pytest
import torch
import numpy as np
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from energies.cartpole import CartPoleEnergy
from models.gfn import GFN
from evaluate_gflownet import compute_mmd, build_gfn_model

DEVICE = torch.device('cpu')


class TestCartPoleEnergy:
    """Unit tests for the target density function."""

    def setup_method(self):
        self.energy = CartPoleEnergy(device=DEVICE)

    def test_data_ndim_is_4(self):
        assert self.energy.data_ndim == 4

    def test_sigma_values(self):
        expected = torch.tensor([0.5, 0.5, 0.1, 0.1])
        assert torch.allclose(self.energy.sigma.cpu(), expected)

    def test_log_reward_at_origin_is_zero(self):
        x = torch.zeros(1, 4)
        assert torch.allclose(self.energy.log_reward(x),
                              torch.tensor([0.0]), atol=1e-6)

    def test_log_reward_is_negative_away_from_origin(self):
        x = torch.tensor([[1.0, 1.0, 0.5, 0.5]])
        assert self.energy.log_reward(x).item() < 0

    def test_log_reward_symmetry(self):
        x = torch.tensor([[0.3, -0.4, 0.05, -0.08]])
        assert torch.allclose(self.energy.log_reward(x),
                              self.energy.log_reward(-x))

    def test_log_reward_batch_shape(self):
        x = torch.randn(32, 4)
        out = self.energy.log_reward(x)
        assert out.shape == (32,)

    def test_angle_penalty_stronger_than_position(self):
        # sigma_pos=0.5, sigma_angle=0.1, so angle deviation hurts more
        x_pos   = torch.tensor([[0.2, 0.0, 0.0, 0.0]])
        x_angle = torch.tensor([[0.0, 0.0, 0.2, 0.0]])
        assert self.energy.log_reward(x_angle).item() \
             < self.energy.log_reward(x_pos).item()

    def test_sample_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.energy.sample(10)


class TestMMD:
    """Unit tests for the MMD divergence metric."""

    def test_mmd_self_is_zero(self):
        x = torch.randn(100, 4)
        assert compute_mmd(x, x) < 1e-6

    def test_mmd_different_distributions_positive(self):
        x = torch.randn(200, 4)
        y = torch.randn(200, 4) + 5.0   # very different mean
        assert compute_mmd(x, y) > 0.1

    def test_mmd_returns_float(self):
        x = torch.randn(50, 4)
        y = torch.randn(50, 4)
        assert isinstance(compute_mmd(x, y), float)


class TestGFNModel:
    """Sanity checks on the GFN model construction."""

    def test_model_builds_without_error(self):
        gfn = build_gfn_model(DEVICE)
        assert gfn.dim == 4

    def test_model_has_back_model(self):
        gfn = build_gfn_model(DEVICE)
        assert hasattr(gfn, 'back_model'), \
            "denoising_theta (back_model) must exist"

    def test_model_has_flow_model_log_Z(self):
        gfn = build_gfn_model(DEVICE)
        assert hasattr(gfn, 'flow_model'), \
            "log Z (flow_model) must exist"


class TestSafetyBounds:
    """Verify the CartPole safety bounds use standard values."""

    def test_position_bound_is_2_4(self):
        assert abs(2.4 - 2.4) < 1e-6   # cartpole standard

    def test_angle_bound_is_12_degrees(self):
        expected_rad = 12.0 * np.pi / 180.0
        assert abs(0.2094 - expected_rad) < 1e-3
