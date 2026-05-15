"""
Evaluation script for MC-PILCO Variant B (KL + chance-constraints) training results.

Usage:
    python evaluate_variant_b.py                          # uses results_variant_b/1/log.pkl
    python evaluate_variant_b.py -log results_variant_b/2/log.pkl
    python evaluate_variant_b.py -save plots/            # also save figures to disk

Generates:
    1. Cost curves  — policy optimisation loss per step for each trial
    2. Theta trajectory  — simulated particle rollouts (predicted by GP)
    3. Real-system rollouts  — actual CartPole trajectories collected per trial
    4. Position (x) trajectory  — cart position over real rollouts
    5. Divergence decomposition  — KL trend across trials at terminal step
    6. Summary table  — printed diagnostic numbers
"""

import argparse
import math
import os
import pathlib
import pickle as pkl
import sys

import matplotlib
matplotlib.use('Agg')          # headless; remove if you have a display
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser("evaluate_variant_b")
p.add_argument("-log", default="results_variant_b/1/log.pkl",
               help="Path to log.pkl produced by test_mcpilco_cartpole_variant_b.py")
p.add_argument("-save", default="results_variant_b/1/plots",
               help="Directory to save figures (created if missing). "
                    "Pass '' to skip saving.")
p.add_argument("-show", action="store_true",
               help="Show interactive matplotlib windows (needs a display)")
args = p.parse_args()

save_dir = pathlib.Path(args.save) if args.save else None
if save_dir:
    save_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print(f"Loading: {args.log}")
with open(args.log, "rb") as f:
    log = pkl.load(f)

PI = math.pi

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def savefig(fig, name):
    if save_dir:
        path = save_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved -> {path}")
    if args.show:
        plt.show()
    plt.close(fig)


def _arr(x):
    return np.array(x)


# ---------------------------------------------------------------------------
# 1. Cost curves per trial
# ---------------------------------------------------------------------------
print("\n[1] Plotting cost curves ...")

n_trials = len(log["cost_trial_list"])

# Layout: up to 6 columns per row, then wrap
n_cols = min(n_trials, 6)
n_rows = (n_trials + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(3.6 * n_cols, 3.8 * n_rows),
                         sharey=False)
axes = np.array(axes).reshape(-1)   # flatten for indexing
fig.suptitle("Policy optimisation cost per trial  (Variant B: KL + chance constraints)",
             fontsize=12, fontweight="bold")

for i in range(n_trials):
    cost = _arr(log["cost_trial_list"][i])
    ax = axes[i]
    ax.plot(cost, color="steelblue", lw=1)
    ax.set_title(f"Trial {i+1}\n"
                 f"start={cost[0]:.0f} → end={cost[-1]:.0f}\n"
                 f"min={cost.min():.0f}", fontsize=9)
    ax.set_xlabel("Opt step")
    ax.set_ylabel("Cost (sum)")
    ax.grid(True, alpha=0.3)
    # Horizontal line at minimum
    ax.axhline(cost.min(), color="red", ls="--", lw=0.8, alpha=0.7)

# Hide unused subplots if any
for j in range(n_trials, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
savefig(fig, "01_cost_curves.png")

# ---------------------------------------------------------------------------
# 2. Theta evolution — GP particle rollouts
# ---------------------------------------------------------------------------
print("[2] Plotting theta particle rollouts ...")

T_sampling = 0.05
n_cols = min(n_trials, 6)
n_rows = (n_trials + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(3.6 * n_cols, 3.8 * n_rows),
                         sharey=True)
axes = np.array(axes).reshape(-1)
fig.suptitle("Theta (angle) trajectory — GP particle rollouts\n"
             "Target: θ = π ≈ 3.14 rad  (green dashed = target ± 0.2094 rad safe zone)",
             fontsize=12, fontweight="bold")

for i in range(n_trials):
    ps = _arr(log["particles_states_list"][i])  # [T, P, 4]
    T, P, _ = ps.shape
    time_ax = np.arange(T) * T_sampling

    ax = axes[i]
    theta = ps[:, :, 2]   # [T, P]
    theta_mean = theta.mean(axis=1)
    theta_std  = theta.std(axis=1)

    ax.fill_between(time_ax,
                    theta_mean - 2*theta_std,
                    theta_mean + 2*theta_std,
                    alpha=0.2, color="steelblue", label="±2σ")
    ax.plot(time_ax, theta_mean, color="steelblue", lw=1.5, label="mean θ")
    # Target zone
    ax.axhline(PI,           color="green", ls="--", lw=1.2, label="target θ=π")
    ax.axhline(PI + 0.2094,  color="green", ls=":",  lw=0.8)
    ax.axhline(PI - 0.2094,  color="green", ls=":",  lw=0.8)
    ax.axhline(0.0,          color="gray",  ls="--", lw=0.8, alpha=0.5)

    final_mean = theta_mean[-1]
    ax.set_title(f"Trial {i+1}\n"
                 f"θ(T) mean = {final_mean:.3f} rad\n"
                 f"|θ(T)-π| = {abs(final_mean - PI):.3f}", fontsize=9)
    ax.set_xlabel("Time (s)")
    if i == 0:
        ax.set_ylabel("θ (rad)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, PI + 0.8)

# Hide unused subplots
for j in range(n_trials, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
savefig(fig, "02_theta_particles.png")

# ---------------------------------------------------------------------------
# 3. Real system trajectories  — theta
# ---------------------------------------------------------------------------
print("[3] Plotting real-system rollouts ...")

state_hist = log["state_samples_history"]   # list of [T, 4] arrays
n_hist = len(state_hist)                    # 1 exploration + n_trials control

# Layout: up to 7 columns per row (exploration + up to 6 trials), then wrap
n_cols = min(n_hist, 7)
n_rows_pair = (n_hist + n_cols - 1) // n_cols   # how many column-rows for theta
fig, axes = plt.subplots(2 * n_rows_pair, n_cols,
                         figsize=(3.0 * n_cols, 3.2 * n_rows_pair),
                         sharex=False)
# Ensure 2D
if axes.ndim == 1:
    axes = axes.reshape(2, -1)
fig.suptitle("Real CartPole system trajectories\n"
             "Top of each row pair: θ (angle)    Bottom: x (cart position)",
             fontsize=12, fontweight="bold")

for j in range(n_hist):
    traj = _arr(state_hist[j])    # [T, 4]
    T_traj = traj.shape[0]
    time_ax = np.arange(T_traj) * T_sampling
    label = "Exploration" if j == 0 else f"Trial {j}"

    theta = traj[:, 2]
    x_pos = traj[:, 0]
    frac_up = np.mean(np.abs(theta - PI) < 0.2094) * 100

    row_pair = j // n_cols   # which pair of (theta-row, x-row)
    col = j % n_cols
    row_theta = 2 * row_pair
    row_x = 2 * row_pair + 1

    # theta
    ax_th = axes[row_theta, col]
    ax_th.plot(time_ax, theta, color="steelblue", lw=1.5)
    ax_th.axhline(PI,          color="green", ls="--", lw=1.2)
    ax_th.axhline(PI + 0.2094, color="green", ls=":",  lw=0.8)
    ax_th.axhline(PI - 0.2094, color="green", ls=":",  lw=0.8)
    ax_th.set_title(f"{label}\nθ ∈ [{theta.min():.2f}, {theta.max():.2f}] rad\n"
                    f"upright {frac_up:.0f}% of time", fontsize=8)
    if col == 0:
        ax_th.set_ylabel("θ (rad)")
    ax_th.set_ylim(-PI - 0.3, PI + 0.8)
    ax_th.grid(True, alpha=0.3)

    # cart position
    ax_x = axes[row_x, col]
    ax_x.plot(time_ax, x_pos, color="darkorange", lw=1.5)
    ax_x.axhline( 2.4, color="red", ls="--", lw=0.8, label="±2.4 m")
    ax_x.axhline(-2.4, color="red", ls="--", lw=0.8)
    ax_x.set_xlabel("Time (s)")
    if col == 0:
        ax_x.set_ylabel("x (m)")
    ax_x.set_ylim(-3, 3)
    ax_x.grid(True, alpha=0.3)
    if j == 0:
        ax_x.legend(fontsize=7)

# Hide unused subplot slots
for j in range(n_hist, n_rows_pair * n_cols):
    row_pair = j // n_cols
    col = j % n_cols
    axes[2 * row_pair,     col].axis("off")
    axes[2 * row_pair + 1, col].axis("off")

plt.tight_layout()
savefig(fig, "03_real_trajectories.png")

# ---------------------------------------------------------------------------
# 4. Progress across trials — key metrics
# ---------------------------------------------------------------------------
print("[4] Plotting cross-trial progress ...")

trial_nums = np.arange(1, n_trials + 1)

# (a) Final cost per trial
final_costs  = [_arr(log["cost_trial_list"][i])[-1]  for i in range(n_trials)]
min_costs    = [_arr(log["cost_trial_list"][i]).min() for i in range(n_trials)]

# (b) Final theta (particles) per trial
theta_means  = []
theta_stds   = []
theta_errors = []
for i in range(n_trials):
    ps = _arr(log["particles_states_list"][i])
    t_final = ps[-1, :, 2]
    theta_means.append(t_final.mean())
    theta_stds.append(t_final.std())
    theta_errors.append(np.abs(t_final - PI).mean())

# (c) Max theta reached in real rollouts (excluding exploration)
real_theta_max = []
for j in range(1, n_hist):
    traj = _arr(state_hist[j])
    real_theta_max.append(traj[:, 2].max())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Learning progress across trials", fontsize=12, fontweight="bold")

ax = axes[0]
ax.plot(trial_nums, final_costs,  "o-", color="steelblue", label="final cost")
ax.plot(trial_nums, min_costs,    "s--", color="navy",     label="min cost")
ax.set_xlabel("Trial")
ax.set_ylabel("Cost")
ax.set_title("Policy optimisation cost")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.errorbar(trial_nums, theta_means, yerr=2*np.array(theta_stds),
            fmt="o-", color="steelblue", capsize=4, label="mean ± 2σ")
ax.axhline(PI,          color="green", ls="--", lw=1.5, label=f"target π={PI:.2f}")
ax.axhline(PI - 0.2094, color="green", ls=":",  lw=0.8)
ax.axhline(PI + 0.2094, color="green", ls=":",  lw=0.8)
ax.set_xlabel("Trial")
ax.set_ylabel("θ at terminal step (rad)")
ax.set_title("GP particle θ(T) — progress toward π")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[2]
if real_theta_max:
    ax.bar(trial_nums, real_theta_max, color="darkorange", alpha=0.8)
    ax.axhline(PI,          color="green", ls="--", lw=1.5, label=f"target π")
    ax.axhline(PI - 0.2094, color="green", ls=":",  lw=0.8, label="safe zone")
    ax.set_xlabel("Trial")
    ax.set_ylabel("max θ (rad)")
    ax.set_title("Real system: max θ reached per trial")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
savefig(fig, "04_progress.png")

# ---------------------------------------------------------------------------
# 5. TASK COST — evaluated on real system rollouts (NOT the KL training cost)
# ---------------------------------------------------------------------------
print("[5] Plotting task-cost evaluation (real rollouts) ...")

# Compute task metrics from state_samples_history (real system trajectories only)
task_metrics = []
for j in range(1, n_hist):   # skip j=0 (random exploration)
    traj = _arr(state_hist[j])    # [T, 4]
    theta  = traj[:, 2]
    x_pos  = traj[:, 0]

    # Angular error from upright target
    ang_err = np.abs(theta - PI)

    # Task cost 1: mean angular distance from upright (lower = better)
    mean_ang_err = ang_err.mean()

    # Task cost 2: terminal angular distance (how close at end)
    terminal_ang_err = ang_err[-1]

    # Task cost 3: % of time upright (|θ - π| < 0.2094 rad)
    pct_upright = np.mean(ang_err < 0.2094) * 100

    # Task cost 4: % of time cart is safe (|x| < 2.4 m)
    pct_cart_safe = np.mean(np.abs(x_pos) < 2.4) * 100

    # Task cost 5: max θ reached
    max_theta = theta.max()

    task_metrics.append({
        "mean_ang_err":      mean_ang_err,
        "terminal_ang_err":  terminal_ang_err,
        "pct_upright":       pct_upright,
        "pct_cart_safe":     pct_cart_safe,
        "max_theta":         max_theta,
    })

trial_nums_task = np.arange(1, len(task_metrics) + 1)
mean_err    = [m["mean_ang_err"]     for m in task_metrics]
term_err    = [m["terminal_ang_err"] for m in task_metrics]
pct_up      = [m["pct_upright"]      for m in task_metrics]
pct_safe    = [m["pct_cart_safe"]    for m in task_metrics]

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle("Task performance on REAL system rollouts\n"
             "(independent of KL training cost — this is what matters)",
             fontsize=12, fontweight="bold")

ax = axes[0]
ax.plot(trial_nums_task, mean_err, "o-", color="steelblue", label="mean |θ-π|")
ax.plot(trial_nums_task, term_err, "s--", color="navy",     label="terminal |θ-π|")
ax.axhline(0.2094, color="green", ls="--", lw=1.2, label="safe zone (0.2094)")
ax.set_xlabel("Trial")
ax.set_ylabel("|θ - π| (rad)")
ax.set_title("Angular error from upright\n(lower = better)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.bar(trial_nums_task, pct_up, color="steelblue", alpha=0.8)
ax.axhline(95, color="green", ls="--", lw=1.2, label="95% target")
ax.set_xlabel("Trial")
ax.set_ylabel("% time")
ax.set_title("Time spent upright\n(|θ - π| < 0.2094 rad, higher = better)")
ax.set_ylim(0, 105)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.bar(trial_nums_task, pct_safe, color="darkorange", alpha=0.8)
ax.axhline(95, color="green", ls="--", lw=1.2, label="95% target")
ax.set_xlabel("Trial")
ax.set_ylabel("% time")
ax.set_title("Time cart is safe\n(|x| < 2.4 m, higher = better)")
ax.set_ylim(0, 105)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[3]
max_th = [m["max_theta"] for m in task_metrics]
ax.bar(trial_nums_task, max_th, color="purple", alpha=0.7)
ax.axhline(PI,          color="green", ls="--", lw=1.5, label="target π")
ax.axhline(PI - 0.2094, color="green", ls=":",  lw=0.8, label="safe zone")
ax.set_xlabel("Trial")
ax.set_ylabel("max θ (rad)")
ax.set_title("Max angle reached\n(per real rollout)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
savefig(fig, "05_task_cost.png")

# ---------------------------------------------------------------------------
# 6. Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("EVALUATION SUMMARY -- MC-PILCO Variant B  (KL + chance constraints)")
print("=" * 90)
print(f"\nLog file : {args.log}")
print(f"Trials   : {n_trials}")
print("GP data points per trial (SOD): "
      + " -> ".join(str(len(log[f'gp_output_list_{i}'][0])) for i in range(n_trials)))

print()
print("--- KL TRAINING COST (what the optimiser minimises) ---")
print(f"{'Trial':>6}  {'OptSteps':>9}  {'KL start':>10}  {'KL end':>9}  {'KL min':>9}")
print("-" * 52)
for i in range(n_trials):
    cost_arr = _arr(log["cost_trial_list"][i])
    print(f"  {i+1:>4}  {len(cost_arr):>9}  {cost_arr[0]:>10.1f}  "
          f"{cost_arr[-1]:>9.1f}  {cost_arr.min():>9.1f}", flush=True)

print()
print("--- TASK COST (real system performance -- what actually matters) ---")
print(f"{'Trial':>6}  {'mean|th-pi|':>12}  {'term|th-pi|':>12}  "
      f"{'%upright':>9}  {'%cart_safe':>11}  {'max_theta':>10}")
print("-" * 72)
for i, m in enumerate(task_metrics):
    print(f"  {i+1:>4}  {m['mean_ang_err']:>12.4f}  {m['terminal_ang_err']:>12.4f}  "
          f"{m['pct_upright']:>9.1f}  {m['pct_cart_safe']:>11.1f}  "
          f"{m['max_theta']:>10.4f}", flush=True)

print()
overall_cost_drop = _arr(log["cost_trial_list"][0])[0] - _arr(log["cost_trial_list"][-1])[-1]
print(f"Total KL cost reduction (trial 1 start -> trial {n_trials} end): {overall_cost_drop:.1f}")

theta_improvement = theta_means[-1] - theta_means[0]
print(f"Particle th(T) improvement (GP rollout): {theta_means[0]:.4f} -> {theta_means[-1]:.4f} rad "
      f"(+{theta_improvement:.4f} rad)")

if task_metrics:
    print(f"Real system max theta: "
          f"{task_metrics[0]['max_theta']:.4f} -> {task_metrics[-1]['max_theta']:.4f} rad  "
          f"(best: {max(m['max_theta'] for m in task_metrics):.4f} rad)")
    swing_up_success = any(m["pct_upright"] > 0 for m in task_metrics)
    print(f"Any upright time achieved (|th-pi| < 0.2094): {swing_up_success}")

print()
print("DIAGNOSIS:")
if theta_means[-1] < PI / 2:
    print("  [!] Particles still in the lower half (theta < pi/2).")
    print("      The GP model lacks data near theta=pi -- more trials needed for")
    print("      the model to learn dynamics in the swing-up region.")
elif theta_means[-1] < PI - 0.5:
    print("  [~] Particles partially swinging up but not yet reaching target.")
    print("      Cost is decreasing -- continue training for more trials.")
else:
    print("  [OK] Particles approaching target (theta ~= pi). Swing-up in progress!")

print()
if save_dir:
    print(f"Plots saved to: {save_dir}/")
print("=" * 72)
