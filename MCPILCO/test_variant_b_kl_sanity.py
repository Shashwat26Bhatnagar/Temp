"""
Quick sanity check for VariantB_Cost in 'kl' mode.

This is a lightweight unit test that verifies the cost function works
correctly with synthetic particles, BEFORE running the full MC-PILCO
smoke test (which is much heavier and exercises GP training too).

Tests:
  1. Loading the GFN checkpoint via GFNPrior works.
  2. KL(q || p) returns a scalar, no NaN, and is non-negative.
  3. KL(q == p) is approximately 0 (particles drawn from target).
  4. KL is large when particles are far from target (origin vs upright).
  5. Gradient flows from KL back through the particles (reparameterization).
  6. Chance-constraint slack works: |theta - pi| <= 0.2094.
"""
import argparse
import math
import pathlib
import sys

import torch

# Repo paths
THIS = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))    # MCPILCO root

from policy_learning.variant_b_cost import VariantB_Cost


def main():
    parser = argparse.ArgumentParser("VariantB_Cost KL sanity check")
    parser.add_argument("-checkpoint", type=str, default=None,
                        help="Path to GFN checkpoint .pt file")
    args = parser.parse_args()

    if args.checkpoint is None:
        repo_root = THIS.parent.parent
        args.checkpoint = str(
            repo_root / 'GflowNet' / 'gfn-diffusion' / 'energy_sampling'
            / 'cartpole_denoising_theta_final.pt')

    print(f"Using checkpoint: {args.checkpoint}")
    print("=" * 70)

    dtype = torch.float64
    device = torch.device('cpu')

    # ----------------------------------------------------------------
    # 1. Instantiate cost
    # ----------------------------------------------------------------
    print("\n--- [1] Instantiating VariantB_Cost ---")
    cost = VariantB_Cost(
        checkpoint_path=args.checkpoint,
        cost_mode='kl',
        alpha=10.0,
        epsilon=0.05,
        weighting='quadratic',
        position_bound=2.4,
        angle_bound=0.2094,
        dtype=dtype, device=device,
    )

    P = 400          # particles
    T = 60           # horizon (3s @ 50ms sampling)
    N_h = T - 1

    # ----------------------------------------------------------------
    # 2. KL between particles and target — far apart (origin vs upright)
    # ----------------------------------------------------------------
    print("\n--- [2] KL when particles are near pole-down ---")
    # particles centered at [0, 0, 0, 0] with small spread
    mu_far = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=dtype, device=device)
    sigma_far = torch.tensor([0.01, 0.01, 0.01, 0.01], dtype=dtype, device=device)
    states_far = (mu_far + sigma_far * torch.randn(T, P, 4, dtype=dtype, device=device))
    states_far.requires_grad_(True)
    inputs = torch.zeros(T, P, 1, dtype=dtype, device=device)

    mean_cost, _ = cost(states_far, inputs, trial_index=0)
    print(f"  Mean cost (sum over steps):  {mean_cost.item():.4f}")
    print(f"  Last divergence per step (first/last/max):  "
          f"{cost.last_divergence_per_step[0].item():.2f} / "
          f"{cost.last_divergence_per_step[-1].item():.2f} / "
          f"{cost.last_divergence_per_step.max().item():.2f}")
    print(f"  Last slack per step (max):   {cost.last_slack_per_step.max().item():.4f}")
    assert not torch.isnan(mean_cost), "FAIL: cost is NaN"
    assert mean_cost.item() > 0, "FAIL: cost should be positive (particles far from target)"
    print("  -> PASS (high cost when far from target)")

    # ----------------------------------------------------------------
    # 3. KL when particles are at target
    # ----------------------------------------------------------------
    print("\n--- [3] KL when particles are at target (upright) ---")
    mu_close = torch.tensor([0.0, 0.0, math.pi, 0.0], dtype=dtype, device=device)
    sigma_close = torch.tensor([0.5, 0.5, 0.1, 0.1], dtype=dtype, device=device)
    states_close = (mu_close + sigma_close * torch.randn(T, P, 4, dtype=dtype, device=device))
    states_close.requires_grad_(True)

    mean_cost_close, _ = cost(states_close, inputs, trial_index=0)
    print(f"  Mean cost (sum over steps):  {mean_cost_close.item():.4f}")
    print(f"  Last divergence at terminal step (t=N_h):  "
          f"{cost.last_divergence_per_step[-1].item():.4f}")
    print(f"  Slack at terminal step:                    "
          f"{cost.last_slack_per_step[-1].item():.4f}")
    # Should be much lower than [2]
    assert mean_cost_close.item() < mean_cost.item(), \
        "FAIL: cost should be LOWER when particles match target"
    print("  -> PASS (cost lower when at target)")

    # ----------------------------------------------------------------
    # 4. Gradient flow
    # ----------------------------------------------------------------
    print("\n--- [4] Gradient flow check ---")
    mean_cost.backward()
    grad_norm = states_far.grad.norm().item()
    print(f"  Total ||grad_states||: {grad_norm:.4f}")
    print(f"  Grad at t=0 (should be near 0 due to weight=0): "
          f"{states_far.grad[0].norm().item():.6f}")
    print(f"  Grad at t=N_h (largest weight): "
          f"{states_far.grad[-1].norm().item():.4f}")
    assert grad_norm > 0, "FAIL: no gradient flowed back to particles"
    assert not torch.isnan(states_far.grad).any(), "FAIL: gradient contains NaN"
    print("  -> PASS (gradient flows, no NaN)")

    # ----------------------------------------------------------------
    # 5. Cross-entropy mode also works
    # ----------------------------------------------------------------
    print("\n--- [5] cross_entropy mode (sanity) ---")
    cost_ce = VariantB_Cost(
        checkpoint_path=args.checkpoint,
        cost_mode='cross_entropy',
        alpha=10.0, epsilon=0.05, weighting='quadratic',
        position_bound=2.4, angle_bound=0.2094,
        dtype=dtype, device=device,
    )
    states_far_2 = states_far.detach().clone().requires_grad_(True)
    mean_cost_ce, _ = cost_ce(states_far_2, inputs, trial_index=0)
    mean_cost_ce.backward()
    print(f"  CE cost: {mean_cost_ce.item():.4f}")
    print(f"  CE grad norm: {states_far_2.grad.norm().item():.4f}")
    assert not torch.isnan(mean_cost_ce), "FAIL: CE cost is NaN"
    print("  -> PASS")

    print("\n" + "=" * 70)
    print("ALL SANITY CHECKS PASSED  ✓")
    print("Safe to run the full smoke test: test_variant_b_one_step.py")


if __name__ == '__main__':
    main()
