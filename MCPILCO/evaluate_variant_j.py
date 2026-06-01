"""
Evaluation script for MC-PILCO Variant J (per-particle GP-vs-GFN reverse KL)
on the cartpole system.

Usage:
    python evaluate_variant_j.py                                    # default
    python evaluate_variant_j.py -log results_variant_j/2/log.pkl
    python evaluate_variant_j.py -save plots/ -show

Generates:
    01_cost_curves.png         -- per-trial policy optimisation loss
    02_theta_particles.png     -- GP-simulated theta with relaxed safe zone
    03_real_trajectories.png   -- actual CartPole trajectories
    04_progress.png            -- cross-trial progress summary
    05_task_cost.png           -- task performance on real rollouts

Gracefully handles partial runs (training crashed mid-trial).
"""

import argparse
import math
import os
import pathlib
import pickle as pkl

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser("evaluate_variant_j (cartpole)")
p.add_argument("-log", default="results_variant_j/1/log.pkl",
               help="Path to log.pkl produced by test_mcpilco_cartpole_variant_j.py")
p.add_argument("-save", default="results_variant_j/1/plots",
               help="Directory to save figures.")
p.add_argument("-show", action="store_true",
               help="Show interactive matplotlib windows.")
args = p.parse_args()

save_dir = pathlib.Path(args.save) if args.save else None
if save_dir:
    save_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants matching Variant J training script (same safety as Variant C)
# ---------------------------------------------------------------------------
PI = math.pi
ANGLE_BOUND = 0.35
POSITION_BOUND = 2.4
T_sampling = 0.05

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print(f"Loading: {args.log}")
with open(args.log, "rb") as f:
    log = pkl.load(f)


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
# Partial-run detection
# ---------------------------------------------------------------------------
n_trials = len(log["cost_trial_list"])
state_hist = log["state_samples_history"]
n_hist = len(state_hist)
expected_completed = n_hist - 1

print(f"Trials with cost history     : {n_trials}")
print(f"Real-system policy rollouts  : {expected_completed} (+1 exploration)")

# ---------------------------------------------------------------------------
# 1. Cost curves per trial
# ---------------------------------------------------------------------------
print("\n[1] Plotting cost curves ...")

n_cols = min(max(n_trials, 1), 6)
n_rows = (max(n_trials, 1) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(3.6 * n_cols, 3.8 * n_rows),
                         sharey=False)
axes = np.array(axes).reshape(-1)
fig.suptitle("Policy optimisation cost per trial\n"
             "(Variant J: per-particle GP-vs-GFN reverse KL)",
             fontsize=12, fontweight="bold")

for i in range(n_trials):
    cost = _arr(log["cost_trial_list"][i])
    ax = axes[i]
    ax.plot(cost, color="teal", lw=1)
    ax.set_title(f"Trial {i+1}\n"
                 f"start={cost[0]:.0f} -> end={cost[-1]:.0f}\n"
                 f"min={cost.min():.0f}", fontsize=9)
    ax.set_xlabel("Opt step")
    ax.set_ylabel("Cost (sum)")
    ax.grid(True, alpha=0.3)
    ax.axhline(cost.min(), color="black", ls="--", lw=0.8, alpha=0.7)

for j in range(n_trials, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
savefig(fig, "01_cost_curves.png")

# ---------------------------------------------------------------------------
# 2. Theta evolution -- GP particle rollouts
# ---------------------------------------------------------------------------
print("[2] Plotting theta particle rollouts ...")

n_particles_trials = len(log["particles_states_list"])
n_cols = min(max(n_particles_trials, 1), 6)
n_rows = (max(n_particles_trials, 1) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(3.6 * n_cols, 3.8 * n_rows),
                         sharey=True)
axes = np.array(axes).reshape(-1)
fig.suptitle("Theta (angle) -- GP particle rollouts (Variant J)\n"
             f"Target: theta=pi  |  Green dashed = pi +/- {ANGLE_BOUND:.2f} rad safe zone",
             fontsize=12, fontweight="bold")

for i in range(n_particles_trials):
    ps = _arr(log["particles_states_list"][i])  # [T, P, 4]
    T, P, _ = ps.shape
    time_ax = np.arange(T) * T_sampling

    ax = axes[i]
    theta = ps[:, :, 2]
    theta_mean = theta.mean(axis=1)
    theta_std = theta.std(axis=1)

    ax.fill_between(time_ax,
                    theta_mean - 2 * theta_std,
                    theta_mean + 2 * theta_std,
                    alpha=0.2, color="teal", label="+/-2 sigma")
    ax.plot(time_ax, theta_mean, color="teal", lw=1.5, label="mean theta")
    ax.axhline(PI,                color="green", ls="--", lw=1.2, label="target theta=pi")
    ax.axhline(PI + ANGLE_BOUND,  color="green", ls=":",  lw=0.8)
    ax.axhline(PI - ANGLE_BOUND,  color="green", ls=":",  lw=0.8)
    ax.axhline(0.0,               color="gray",  ls="--", lw=0.8, alpha=0.5)

    final_mean = theta_mean[-1]
    ax.set_title(f"Trial {i+1}\n"
                 f"theta(T) mean = {final_mean:.3f} rad\n"
                 f"|theta(T)-pi| = {abs(final_mean - PI):.3f}", fontsize=9)
    ax.set_xlabel("Time (s)")
    if i == 0:
        ax.set_ylabel("theta (rad)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, PI + 0.8)

for j in range(n_particles_trials, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
savefig(fig, "02_theta_particles.png")

# ---------------------------------------------------------------------------
# 3. Real system trajectories
# ---------------------------------------------------------------------------
print("[3] Plotting real-system rollouts ...")

n_cols = min(max(n_hist, 1), 7)
n_rows_pair = (max(n_hist, 1) + n_cols - 1) // n_cols
fig, axes = plt.subplots(2 * n_rows_pair, n_cols,
                         figsize=(3.0 * n_cols, 3.2 * n_rows_pair),
                         sharex=False)
if axes.ndim == 1:
    axes = axes.reshape(2, -1)
fig.suptitle("Real CartPole system trajectories (Variant J)\n"
             "Top: theta (angle)    Bottom: x (cart position)",
             fontsize=12, fontweight="bold")

for j in range(n_hist):
    traj = _arr(state_hist[j])
    T_traj = traj.shape[0]
    time_ax = np.arange(T_traj) * T_sampling
    label = "Exploration" if j == 0 else f"Trial {j}"

    theta = traj[:, 2]
    x_pos = traj[:, 0]
    frac_up = np.mean(np.abs(np.abs(theta) - PI) < ANGLE_BOUND) * 100

    row_pair = j // n_cols
    col = j % n_cols
    row_theta = 2 * row_pair
    row_x = 2 * row_pair + 1

    ax_th = axes[row_theta, col]
    ax_th.plot(time_ax, theta, color="teal", lw=1.5)
    ax_th.axhline(PI,               color="green", ls="--", lw=1.2)
    ax_th.axhline(-PI,              color="green", ls="--", lw=1.2, alpha=0.5)
    ax_th.axhline(PI + ANGLE_BOUND, color="green", ls=":",  lw=0.8)
    ax_th.axhline(PI - ANGLE_BOUND, color="green", ls=":",  lw=0.8)
    ax_th.set_title(f"{label}\ntheta in [{theta.min():.2f}, {theta.max():.2f}] rad\n"
                    f"upright {frac_up:.0f}% of time", fontsize=8)
    if col == 0:
        ax_th.set_ylabel("theta (rad)")
    ax_th.set_ylim(-PI - 0.3, PI + 0.8)
    ax_th.grid(True, alpha=0.3)

    ax_x = axes[row_x, col]
    ax_x.plot(time_ax, x_pos, color="darkorange", lw=1.5)
    ax_x.axhline( POSITION_BOUND, color="red", ls="--", lw=0.8)
    ax_x.axhline(-POSITION_BOUND, color="red", ls="--", lw=0.8)
    ax_x.set_xlabel("Time (s)")
    if col == 0:
        ax_x.set_ylabel("x (m)")
    ax_x.set_ylim(-3, 3)
    ax_x.grid(True, alpha=0.3)

for j in range(n_hist, n_rows_pair * n_cols):
    row_pair = j // n_cols
    col = j % n_cols
    axes[2 * row_pair,     col].axis("off")
    axes[2 * row_pair + 1, col].axis("off")

plt.tight_layout()
savefig(fig, "03_real_trajectories.png")

# ---------------------------------------------------------------------------
# 4. Progress across trials
# ---------------------------------------------------------------------------
print("[4] Plotting cross-trial progress ...")

trial_nums = np.arange(1, n_trials + 1)
final_costs = [_arr(log["cost_trial_list"][i])[-1] for i in range(n_trials)]
min_costs   = [_arr(log["cost_trial_list"][i]).min() for i in range(n_trials)]

theta_means, theta_stds = [], []
for i in range(n_particles_trials):
    ps = _arr(log["particles_states_list"][i])
    t_final = ps[-1, :, 2]
    theta_means.append(t_final.mean())
    theta_stds.append(t_final.std())

real_theta_max = []
for j in range(1, n_hist):
    traj = _arr(state_hist[j])
    real_theta_max.append(traj[:, 2].max())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Learning progress across trials (Variant J)", fontsize=12, fontweight="bold")

ax = axes[0]
ax.plot(trial_nums, final_costs, "o-",  color="teal", label="final cost")
ax.plot(trial_nums, min_costs,   "s--", color="darkcyan", label="min cost")
ax.set_xlabel("Trial"); ax.set_ylabel("Cost")
ax.set_title("Policy optimisation cost\n(per-particle GP-vs-GFN KL + slack)")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
if n_particles_trials > 0:
    ax.errorbar(np.arange(1, n_particles_trials + 1),
                theta_means, yerr=2 * np.array(theta_stds),
                fmt="o-", color="teal", capsize=4, label="mean +/- 2 sigma")
ax.axhline(PI,                color="green", ls="--", lw=1.5, label=f"target pi={PI:.2f}")
ax.axhline(PI - ANGLE_BOUND,  color="green", ls=":",  lw=0.8)
ax.axhline(PI + ANGLE_BOUND,  color="green", ls=":",  lw=0.8)
ax.set_xlabel("Trial"); ax.set_ylabel("theta at terminal step (rad)")
ax.set_title("GP particle theta(T) -- progress toward pi")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[2]
if real_theta_max:
    ax.bar(np.arange(1, len(real_theta_max) + 1),
           real_theta_max, color="darkorange", alpha=0.8)
    ax.axhline(PI,               color="green", ls="--", lw=1.5, label="target pi")
    ax.axhline(PI - ANGLE_BOUND, color="green", ls=":",  lw=0.8, label="safe zone")
    ax.set_xlabel("Trial"); ax.set_ylabel("max theta (rad)")
    ax.set_title("Real system: max theta reached per trial")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
savefig(fig, "04_progress.png")

# ---------------------------------------------------------------------------
# 5. TASK COST on real rollouts
# ---------------------------------------------------------------------------
print("[5] Plotting task-cost evaluation (real rollouts) ...")

task_metrics = []
for j in range(1, n_hist):
    traj = _arr(state_hist[j])
    theta = traj[:, 2]
    x_pos = traj[:, 0]
    ang_err = np.minimum(np.abs(theta - PI), np.abs(theta + PI))
    task_metrics.append({
        "mean_ang_err":     ang_err.mean(),
        "terminal_ang_err": ang_err[-1],
        "pct_upright":      np.mean(ang_err < ANGLE_BOUND) * 100,
        "pct_cart_safe":    np.mean(np.abs(x_pos) < POSITION_BOUND) * 100,
        "max_theta":        theta.max(),
    })

if task_metrics:
    trial_nums_task = np.arange(1, len(task_metrics) + 1)
    mean_err = [m["mean_ang_err"]     for m in task_metrics]
    term_err = [m["terminal_ang_err"] for m in task_metrics]
    pct_up   = [m["pct_upright"]      for m in task_metrics]
    pct_safe = [m["pct_cart_safe"]    for m in task_metrics]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Task performance on REAL system (Variant J)",
                 fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.plot(trial_nums_task, mean_err, "o-",  color="teal", label="mean ||theta|-pi|")
    ax.plot(trial_nums_task, term_err, "s--", color="darkcyan", label="terminal ||theta|-pi|")
    ax.axhline(ANGLE_BOUND, color="green", ls="--", lw=1.2, label=f"safe zone ({ANGLE_BOUND})")
    ax.set_xlabel("Trial"); ax.set_ylabel("||theta| - pi| (rad)")
    ax.set_title("Angular error from upright\n(lower = better)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(trial_nums_task, pct_up, color="teal", alpha=0.8)
    ax.axhline(95, color="green", ls="--", lw=1.2, label="95% target")
    ax.set_xlabel("Trial"); ax.set_ylabel("% time")
    ax.set_title(f"Time spent upright\n(||theta|-pi| < {ANGLE_BOUND})")
    ax.set_ylim(0, 105); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.bar(trial_nums_task, pct_safe, color="darkorange", alpha=0.8)
    ax.axhline(95, color="green", ls="--", lw=1.2, label="95% target")
    ax.set_xlabel("Trial"); ax.set_ylabel("% time")
    ax.set_title(f"Time cart is safe (|x| < {POSITION_BOUND} m)")
    ax.set_ylim(0, 105); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig(fig, "05_task_cost.png")

# ---------------------------------------------------------------------------
# 6. Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("EVALUATION SUMMARY -- MC-PILCO Variant J  (per-particle GP-vs-GFN KL)")
print("=" * 80)
print(f"\nLog file       : {args.log}")
print(f"Trials (cost)  : {n_trials}")
print(f"Real rollouts  : {expected_completed} policy + 1 exploration")

print()
print("--- TRAINING COST ---")
print(f"{'Trial':>6}  {'OptSteps':>9}  {'start':>10}  {'end':>9}  {'min':>9}")
print("-" * 52)
for i in range(n_trials):
    cost_arr = _arr(log["cost_trial_list"][i])
    print(f"  {i+1:>4}  {len(cost_arr):>9}  {cost_arr[0]:>10.1f}  "
          f"{cost_arr[-1]:>9.1f}  {cost_arr.min():>9.1f}")

print()
print("--- TASK PERFORMANCE (real system) ---")
print(f"{'Trial':>6}  {'mean||th|-pi|':>14}  {'term||th|-pi|':>14}  "
      f"{'%upright':>9}  {'%cart_safe':>11}")
print("-" * 62)
for i, m in enumerate(task_metrics):
    print(f"  {i+1:>4}  {m['mean_ang_err']:>14.4f}  {m['terminal_ang_err']:>14.4f}  "
          f"{m['pct_upright']:>9.1f}  {m['pct_cart_safe']:>11.1f}")

print()
if save_dir:
    print(f"Plots saved to: {save_dir}/")
print("=" * 80)
