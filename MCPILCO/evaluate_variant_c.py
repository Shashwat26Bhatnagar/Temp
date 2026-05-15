"""
Evaluation script for MC-PILCO Variant C (REVERSE KL + RELAXED chance-constraints).

Usage:
    python evaluate_variant_c.py                                # uses results_variant_c_5/1/log.pkl
    python evaluate_variant_c.py -log results_variant_c_5/2/log.pkl
    python evaluate_variant_c.py -save plots/                   # also save figures to disk

Variant C constants used for plotting (must match training script):
    angle_bound    = 0.35 rad (~20 deg)
    position_bound = 2.4 m
    epsilon        = 0.10
    alpha          = 5.0

Generates:
    1. Cost curves            -- per-trial policy optimisation loss
    2. Theta particle rollouts -- GP-simulated theta with relaxed safe zone
    3. Real-system rollouts    -- actual CartPole trajectories
    4. Cross-trial progress    -- cost / theta / max-theta summaries
    5. Task cost on real rollouts
    6. Summary table (KL cost vs task cost)

Gracefully handles partial runs (training crashed mid-trial).
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
p = argparse.ArgumentParser("evaluate_variant_c")
p.add_argument("-log", default="results_variant_c_5/1/log.pkl",
               help="Path to log.pkl produced by test_mcpilco_cartpole_variant_c.py")
p.add_argument("-save", default="results_variant_c_5/1/plots",
               help="Directory to save figures (created if missing). "
                    "Pass '' to skip saving.")
p.add_argument("-show", action="store_true",
               help="Show interactive matplotlib windows (needs a display)")
args = p.parse_args()

save_dir = pathlib.Path(args.save) if args.save else None
if save_dir:
    save_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants matching Variant C training script (relaxed safety)
# ---------------------------------------------------------------------------
PI = math.pi
ANGLE_BOUND = 0.35        # was 0.2094 in Variant B (~12 deg)
POSITION_BOUND = 2.4
T_sampling = 0.05

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print(f"Loading: {args.log}")
with open(args.log, "rb") as f:
    log = pkl.load(f)

# ---------------------------------------------------------------------------
# Helpers
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
# Partial-run detection
# ---------------------------------------------------------------------------
n_trials = len(log["cost_trial_list"])
state_hist = log["state_samples_history"]
n_hist = len(state_hist)            # 1 exploration + (completed) policy rollouts
expected_completed = n_hist - 1     # trials whose real rollout finished

if n_trials < expected_completed:
    print(f"[!] cost_trial_list has {n_trials} entries but {expected_completed} "
          f"real rollouts exist. Run may have been interrupted mid-trial.")
if expected_completed < n_trials:
    print(f"[!] More cost curves ({n_trials}) than real rollouts ({expected_completed}). "
          f"Likely crashed before the policy was rolled out on the real system.")

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
fig.suptitle("Policy optimisation cost per trial  "
             "(Variant C: REVERSE KL + relaxed chance constraints)",
             fontsize=12, fontweight="bold")

for i in range(n_trials):
    cost = _arr(log["cost_trial_list"][i])
    ax = axes[i]
    ax.plot(cost, color="crimson", lw=1)
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
fig.suptitle("Theta (angle) -- GP particle rollouts (Variant C)\n"
             f"Target: theta=pi  |  Green dashed = pi +/- {ANGLE_BOUND:.2f} rad relaxed safe zone",
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
                    alpha=0.2, color="crimson", label="+/-2 sigma")
    ax.plot(time_ax, theta_mean, color="crimson", lw=1.5, label="mean theta")
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
fig.suptitle("Real CartPole system trajectories (Variant C)\n"
             "Top of each row pair: theta (angle)    Bottom: x (cart position)",
             fontsize=12, fontweight="bold")

for j in range(n_hist):
    traj = _arr(state_hist[j])
    T_traj = traj.shape[0]
    time_ax = np.arange(T_traj) * T_sampling
    label = "Exploration" if j == 0 else f"Trial {j}"

    theta = traj[:, 2]
    x_pos = traj[:, 0]
    frac_up = np.mean(np.abs(theta - PI) < ANGLE_BOUND) * 100

    row_pair = j // n_cols
    col = j % n_cols
    row_theta = 2 * row_pair
    row_x = 2 * row_pair + 1

    # theta
    ax_th = axes[row_theta, col]
    ax_th.plot(time_ax, theta, color="crimson", lw=1.5)
    ax_th.axhline(PI,               color="green", ls="--", lw=1.2)
    ax_th.axhline(PI + ANGLE_BOUND, color="green", ls=":",  lw=0.8)
    ax_th.axhline(PI - ANGLE_BOUND, color="green", ls=":",  lw=0.8)
    ax_th.set_title(f"{label}\ntheta in [{theta.min():.2f}, {theta.max():.2f}] rad\n"
                    f"upright {frac_up:.0f}% of time", fontsize=8)
    if col == 0:
        ax_th.set_ylabel("theta (rad)")
    ax_th.set_ylim(-PI - 0.3, PI + 0.8)
    ax_th.grid(True, alpha=0.3)

    # cart position
    ax_x = axes[row_x, col]
    ax_x.plot(time_ax, x_pos, color="darkorange", lw=1.5)
    ax_x.axhline( POSITION_BOUND, color="red", ls="--", lw=0.8, label=f"+/-{POSITION_BOUND} m")
    ax_x.axhline(-POSITION_BOUND, color="red", ls="--", lw=0.8)
    ax_x.set_xlabel("Time (s)")
    if col == 0:
        ax_x.set_ylabel("x (m)")
    ax_x.set_ylim(-3, 3)
    ax_x.grid(True, alpha=0.3)
    if j == 0:
        ax_x.legend(fontsize=7)

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

theta_means, theta_stds, theta_errors = [], [], []
for i in range(n_particles_trials):
    ps = _arr(log["particles_states_list"][i])
    t_final = ps[-1, :, 2]
    theta_means.append(t_final.mean())
    theta_stds.append(t_final.std())
    theta_errors.append(np.abs(t_final - PI).mean())

real_theta_max = []
for j in range(1, n_hist):
    traj = _arr(state_hist[j])
    real_theta_max.append(traj[:, 2].max())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Learning progress across trials (Variant C)", fontsize=12, fontweight="bold")

ax = axes[0]
ax.plot(trial_nums, final_costs, "o-",  color="crimson", label="final cost")
ax.plot(trial_nums, min_costs,   "s--", color="darkred", label="min cost")
ax.set_xlabel("Trial")
ax.set_ylabel("Cost")
ax.set_title("Policy optimisation cost (reverse KL + slack)")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
if n_particles_trials > 0:
    ax.errorbar(np.arange(1, n_particles_trials + 1),
                theta_means, yerr=2 * np.array(theta_stds),
                fmt="o-", color="crimson", capsize=4, label="mean +/- 2 sigma")
ax.axhline(PI,                color="green", ls="--", lw=1.5, label=f"target pi={PI:.2f}")
ax.axhline(PI - ANGLE_BOUND,  color="green", ls=":",  lw=0.8)
ax.axhline(PI + ANGLE_BOUND,  color="green", ls=":",  lw=0.8)
ax.set_xlabel("Trial")
ax.set_ylabel("theta at terminal step (rad)")
ax.set_title("GP particle theta(T) -- progress toward pi")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[2]
if real_theta_max:
    ax.bar(np.arange(1, len(real_theta_max) + 1),
           real_theta_max, color="darkorange", alpha=0.8)
    ax.axhline(PI,               color="green", ls="--", lw=1.5, label="target pi")
    ax.axhline(PI - ANGLE_BOUND, color="green", ls=":",  lw=0.8, label="safe zone")
    ax.set_xlabel("Trial")
    ax.set_ylabel("max theta (rad)")
    ax.set_title("Real system: max theta reached per trial")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
savefig(fig, "04_progress.png")

# ---------------------------------------------------------------------------
# 5. TASK COST -- evaluated on real rollouts
# ---------------------------------------------------------------------------
print("[5] Plotting task-cost evaluation (real rollouts) ...")

task_metrics = []
for j in range(1, n_hist):
    traj = _arr(state_hist[j])
    theta = traj[:, 2]
    x_pos = traj[:, 0]
    ang_err = np.abs(theta - PI)
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
    max_th   = [m["max_theta"]        for m in task_metrics]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("Task performance on REAL system rollouts (Variant C)\n"
                 "(independent of KL training cost -- this is what matters)",
                 fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.plot(trial_nums_task, mean_err, "o-",  color="crimson", label="mean |theta-pi|")
    ax.plot(trial_nums_task, term_err, "s--", color="darkred", label="terminal |theta-pi|")
    ax.axhline(ANGLE_BOUND, color="green", ls="--", lw=1.2, label=f"safe zone ({ANGLE_BOUND})")
    ax.set_xlabel("Trial")
    ax.set_ylabel("|theta - pi| (rad)")
    ax.set_title("Angular error from upright\n(lower = better)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(trial_nums_task, pct_up, color="crimson", alpha=0.8)
    ax.axhline(95, color="green", ls="--", lw=1.2, label="95% target")
    ax.set_xlabel("Trial")
    ax.set_ylabel("% time")
    ax.set_title(f"Time spent upright\n(|theta - pi| < {ANGLE_BOUND}, higher = better)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.bar(trial_nums_task, pct_safe, color="darkorange", alpha=0.8)
    ax.axhline(95, color="green", ls="--", lw=1.2, label="95% target")
    ax.set_xlabel("Trial")
    ax.set_ylabel("% time")
    ax.set_title(f"Time cart is safe\n(|x| < {POSITION_BOUND} m, higher = better)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.bar(trial_nums_task, max_th, color="purple", alpha=0.7)
    ax.axhline(PI,                color="green", ls="--", lw=1.5, label="target pi")
    ax.axhline(PI - ANGLE_BOUND,  color="green", ls=":",  lw=0.8, label="safe zone")
    ax.set_xlabel("Trial")
    ax.set_ylabel("max theta (rad)")
    ax.set_title("Max angle reached\n(per real rollout)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig(fig, "05_task_cost.png")
else:
    print("  [skip] no real-system rollouts to evaluate task cost on.")

# ---------------------------------------------------------------------------
# 6. Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("EVALUATION SUMMARY -- MC-PILCO Variant C  (REVERSE KL + RELAXED chance constraints)")
print("=" * 90)
print(f"\nLog file       : {args.log}")
print(f"Trials (cost)  : {n_trials}")
print(f"Real rollouts  : {expected_completed} policy + 1 exploration")
print(f"Safety bounds  : alpha=5.0  epsilon=0.10  angle_bound={ANGLE_BOUND}  "
      f"position_bound={POSITION_BOUND}")
gp_pts = []
for i in range(2):  # GPs trained
    key = f"gp_output_list_{i}"
    if key in log:
        try:
            gp_pts.append(str(len(log[key][0])))
        except Exception:
            gp_pts.append("?")
print(f"GP_0 data pts  : {gp_pts[0] if len(gp_pts) > 0 else 'n/a'}")
print(f"GP_1 data pts  : {gp_pts[1] if len(gp_pts) > 1 else 'n/a'}")

print()
print("--- KL TRAINING COST (what the optimiser minimises) ---")
print(f"{'Trial':>6}  {'OptSteps':>9}  {'KL start':>10}  {'KL end':>9}  {'KL min':>9}")
print("-" * 52)
for i in range(n_trials):
    cost_arr = _arr(log["cost_trial_list"][i])
    print(f"  {i+1:>4}  {len(cost_arr):>9}  {cost_arr[0]:>10.1f}  "
          f"{cost_arr[-1]:>9.1f}  {cost_arr.min():>9.1f}", flush=True)

print()
print("--- TASK COST (real system performance) ---")
print(f"{'Trial':>6}  {'mean|th-pi|':>12}  {'term|th-pi|':>12}  "
      f"{'%upright':>9}  {'%cart_safe':>11}  {'max_theta':>10}")
print("-" * 72)
for i, m in enumerate(task_metrics):
    print(f"  {i+1:>4}  {m['mean_ang_err']:>12.4f}  {m['terminal_ang_err']:>12.4f}  "
          f"{m['pct_upright']:>9.1f}  {m['pct_cart_safe']:>11.1f}  "
          f"{m['max_theta']:>10.4f}", flush=True)

print()
if n_trials > 0:
    overall_cost_drop = _arr(log["cost_trial_list"][0])[0] - _arr(log["cost_trial_list"][-1])[-1]
    print(f"Total KL cost reduction (trial 1 start -> trial {n_trials} end): {overall_cost_drop:.1f}")

if theta_means:
    theta_improvement = theta_means[-1] - theta_means[0]
    print(f"Particle theta(T) (GP rollout): {theta_means[0]:.4f} -> {theta_means[-1]:.4f} rad "
          f"(+{theta_improvement:.4f} rad)")

if task_metrics:
    print(f"Real system max theta: "
          f"{task_metrics[0]['max_theta']:.4f} -> {task_metrics[-1]['max_theta']:.4f} rad  "
          f"(best: {max(m['max_theta'] for m in task_metrics):.4f} rad)")
    swing_up_success = any(m["pct_upright"] > 0 for m in task_metrics)
    print(f"Any upright time achieved (|th-pi| < {ANGLE_BOUND}): {swing_up_success}")

print()
print("DIAGNOSIS:")
if not theta_means:
    print("  [!] No particle rollouts available -- training crashed before trial 1 finished.")
elif theta_means[-1] < PI / 2:
    print("  [!] Particles still in the lower half (theta < pi/2).")
    print("      The GP model lacks data near theta=pi -- more trials needed.")
elif theta_means[-1] < PI - 0.5:
    print("  [~] Particles partially swinging up but not yet reaching target.")
    print("      Reverse KL is engaging the prior; continue training for more trials.")
else:
    print("  [OK] Particles approaching target (theta ~= pi). Swing-up in progress!")

if expected_completed < 5:
    print()
    print(f"  Note: only {expected_completed}/5 trials completed -- run was interrupted.")
    print("  To resume, re-run: python test_mcpilco_cartpole_variant_c.py -seed 1")
    print("  (this will start fresh; MC-PILCO does not currently support mid-run resume.)")

print()
if save_dir:
    print(f"Plots saved to: {save_dir}/")
print("=" * 72)
