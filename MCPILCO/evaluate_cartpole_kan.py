"""
Evaluate & compare cartpole MC-PILCO: KAN policy vs RBF baseline.

Can evaluate a single run or overlay two runs for side-by-side comparison
of sample complexity (trials to swing up) and learning curves.

Usage:
    # Evaluate KAN run only
    python evaluate_cartpole_kan.py -kan results_cartpole_kan/1/log.pkl

    # Compare KAN vs RBF
    python evaluate_cartpole_kan.py \
        -kan results_cartpole_kan/1/log.pkl \
        -rbf results_tmp/1/log.pkl

    # Evaluate RBF only
    python evaluate_cartpole_kan.py -rbf results_tmp/1/log.pkl

    # Options
    python evaluate_cartpole_kan.py -kan ... -rbf ... -save my_plots/
    python evaluate_cartpole_kan.py -kan ... -no_anim
    python evaluate_cartpole_kan.py -kan ... -trial 3     # specific rollout

Outputs (into <save_dir>/):
    01_cost_curves.png         Per-trial policy optimisation cost
    02_state_trajectories.png  [p, p_dot, theta, theta_dot] vs time
    03_swingup_metric.png      Real-system cost per timestep per trial
    04_sample_complexity.png   Bar chart: trials to swing-up for each method
    05_cartpole_animation.gif  Stick-figure playback (unless -no_anim)
"""

import argparse
import pathlib
import pickle as pkl
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser("evaluate_cartpole_kan")
p.add_argument("-kan", type=str, default=None,
               help="Path to KAN log.pkl  (e.g. results_cartpole_kan/1/log.pkl)")
p.add_argument("-rbf", type=str, default=None,
               help="Path to RBF baseline log.pkl (e.g. results_tmp/1/log.pkl)")
p.add_argument("-save", type=str, default=None,
               help="Save directory (default: auto-generated).")
p.add_argument("-trial", type=int, default=-1,
               help="Which real-system rollout to plot (-1 = last).")
p.add_argument("-no_anim", action="store_true",
               help="Skip animation GIF generation.")
p.add_argument("-anim_step", type=int, default=1,
               help="Animate every N-th timestep (default 1 = all).")
p.add_argument("-swingup_thresh", type=float, default=0.2,
               help="Mean cost threshold to declare swing-up solved "
                    "(default 0.2).")
args = p.parse_args()

if args.kan is None and args.rbf is None:
    p.error("Provide at least one of -kan or -rbf.")

# ---------------------------------------------------------------------------
# Load logs
# ---------------------------------------------------------------------------
logs = {}
colours = {}
labels = {}

if args.kan is not None:
    kan_path = pathlib.Path(args.kan)
    if not kan_path.exists():
        raise SystemExit(f"KAN log not found: {kan_path}")
    print(f"Loading KAN: {kan_path}")
    with open(kan_path, "rb") as f:
        logs["KAN"] = pkl.load(f)
    colours["KAN"] = "royalblue"
    labels["KAN"] = "KAN policy"

if args.rbf is not None:
    rbf_path = pathlib.Path(args.rbf)
    if not rbf_path.exists():
        raise SystemExit(f"RBF log not found: {rbf_path}")
    print(f"Loading RBF: {rbf_path}")
    with open(rbf_path, "rb") as f:
        logs["RBF"] = pkl.load(f)
    colours["RBF"] = "crimson"
    labels["RBF"] = "RBF policy (200 basis)"

method_keys = list(logs.keys())
compare_mode = len(method_keys) == 2

# Resolve save directory
if args.save is not None:
    save_dir = pathlib.Path(args.save)
elif compare_mode:
    save_dir = pathlib.Path("plots_cartpole_comparison")
elif "KAN" in logs:
    save_dir = kan_path.parent / "plots"
else:
    save_dir = rbf_path.parent / "plots"
save_dir.mkdir(parents=True, exist_ok=True)
print(f"Save dir: {save_dir}")

# Environment constants
T_sampling = 0.05
T_control = 3.0
N_steps = int(T_control / T_sampling)

STATE_NAMES = ["Cart pos (m)", "Cart vel (m/s)",
               "Pole angle (rad)", "Pole ang vel (rad/s)"]
STATE_KEYS  = ["p", "p_dot", "theta", "theta_dot"]


def savefig(fig, name):
    path = save_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_real_cost(states, target_theta=np.pi, target_x=0.0,
                      l_theta=3.0, l_x=1.0):
    """Cart-pole saturated cost on a real rollout.
    states: [T, 4] with columns [p, p_dot, theta, theta_dot]
    Returns: [T] cost in [0, 1].
    """
    x = states[:, 0]
    theta = states[:, 2]
    cost = 1.0 - np.exp(
        -((np.abs(theta) - target_theta) / l_theta) ** 2
        - ((x - target_x) / l_x) ** 2
    )
    return cost


def trials_to_swingup(log, threshold=0.2):
    """Return the first trial index where the real-system rollout achieves
    mean cost < threshold over the last 25% of the trajectory.
    Returns None if never solved.
    """
    hist = log.get("state_samples_history", [])
    for i, states in enumerate(hist):
        if i == 0:
            continue  # skip exploration trial
        states = np.asarray(states)
        cost = compute_real_cost(states)
        tail = cost[int(0.75 * len(cost)):]
        if tail.mean() < threshold:
            return i  # trial index (1-based policy trial = i)
    return None


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("CARTPOLE EVALUATION SUMMARY")
print("=" * 70)
for mk in method_keys:
    log = logs[mk]
    n_trials = len(log.get("cost_trial_list", []))
    n_hist = len(log.get("state_samples_history", []))
    solved = trials_to_swingup(log, args.swingup_thresh)
    print(f"\n  [{mk}]")
    print(f"    Real rollouts : {n_hist}  (1 exploration + {n_hist - 1} policy)")
    print(f"    Policy trials : {n_trials} completed")
    if n_trials > 0:
        for i in range(n_trials):
            c = np.asarray(log["cost_trial_list"][i])
            print(f"      Trial {i+1}: opt_steps={len(c):>5}  "
                  f"start={c[0]:>10.2f}  end={c[-1]:>10.2f}  "
                  f"min={c.min():>10.2f}")
    if solved is not None:
        print(f"    Swing-up solved at rollout {solved} "
              f"(policy trial {solved})")
    else:
        print(f"    Swing-up NOT solved (threshold={args.swingup_thresh})")
print("=" * 70)


# ---------------------------------------------------------------------------
# 1) Cost curves -- policy optimisation cost per trial
# ---------------------------------------------------------------------------
print("\n[1] Cost curves ...")
max_trials = max(len(logs[mk].get("cost_trial_list", []))
                 for mk in method_keys)
if max_trials > 0:
    n_cols = min(max_trials, 5)
    n_rows = (max_trials + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.8 * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)
    title = "Cartpole -- policy optimisation cost per trial"
    if compare_mode:
        title += "  (KAN vs RBF)"
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for i in range(max_trials):
        ax = axes[i]
        for mk in method_keys:
            trials = logs[mk].get("cost_trial_list", [])
            if i < len(trials):
                c = np.asarray(trials[i])
                ax.plot(c, color=colours[mk], lw=1.2, alpha=0.9,
                        label=f"{mk} (end={c[-1]:.1f})")
        ax.set_title(f"Trial {i+1}", fontsize=10)
        ax.set_xlabel("Opt step")
        ax.set_ylabel("Cost")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    for j in range(max_trials, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    savefig(fig, "01_cost_curves.png")
else:
    print("  skipped (no completed policy trials).")


# ---------------------------------------------------------------------------
# 2) State trajectories from real system
# ---------------------------------------------------------------------------
print("[2] State trajectories ...")
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
axes = axes.flatten()
title = "Cartpole -- real system rollout"
if compare_mode:
    title += "  (KAN vs RBF)"
fig.suptitle(title, fontsize=12, fontweight="bold")

for mk in method_keys:
    hist = logs[mk].get("state_samples_history", [])
    n_hist = len(hist)
    if n_hist == 0:
        continue
    trial_idx = args.trial if args.trial >= 0 else (n_hist - 1)
    trial_idx = max(0, min(trial_idx, n_hist - 1))
    states = np.asarray(hist[trial_idx])
    t_axis = np.arange(len(states)) * T_sampling

    for j in range(4):
        ax = axes[j]
        ax.plot(t_axis, states[:, j], color=colours[mk], lw=1.5,
                alpha=0.9, label=f"{mk} (rollout {trial_idx})")
        ax.set_title(STATE_NAMES[j], fontsize=10)
        ax.set_ylabel(STATE_NAMES[j].split("(")[-1].replace(")", ""))
        ax.grid(True, alpha=0.3)

# Add target lines on the angle subplot
axes[2].axhline(np.pi, color="green", ls="--", lw=1.2, alpha=0.7,
                label="target theta = pi")
axes[2].axhline(-np.pi, color="green", ls="--", lw=1.2, alpha=0.7)
axes[0].axhline(0.0, color="green", ls="--", lw=1.0, alpha=0.5,
                label="target x = 0")

for j in range(4):
    axes[j].legend(fontsize=7, loc="best")
    if j >= 2:
        axes[j].set_xlabel("Time (s)")
plt.tight_layout()
savefig(fig, "02_state_trajectories.png")


# ---------------------------------------------------------------------------
# 3) Swing-up metric: real-system cost per timestep per trial
# ---------------------------------------------------------------------------
print("[3] Swing-up metric (real-system cost) ...")
fig, axes_grid = plt.subplots(1, len(method_keys),
                               figsize=(7 * len(method_keys), 5),
                               squeeze=False)

for col, mk in enumerate(method_keys):
    ax = axes_grid[0, col]
    hist = logs[mk].get("state_samples_history", [])
    n_hist = len(hist)
    if n_hist <= 1:
        ax.text(0.5, 0.5, "No policy rollouts", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")
        ax.set_title(f"{mk} -- real system cost per trial")
        continue

    cmap = plt.cm.viridis
    for i in range(1, n_hist):  # skip exploration
        states = np.asarray(hist[i])
        cost = compute_real_cost(states)
        t_ax = np.arange(len(cost)) * T_sampling
        c_val = (i - 1) / max(n_hist - 2, 1)
        tail_mean = cost[int(0.75 * len(cost)):].mean()
        ax.plot(t_ax, cost, color=cmap(c_val), lw=1.3, alpha=0.85,
                label=f"trial {i} (tail={tail_mean:.3f})")

    ax.axhline(args.swingup_thresh, color="red", ls=":", lw=1.2,
               label=f"threshold = {args.swingup_thresh}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Saturated cost")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{mk} -- real system cost per trial", fontsize=11)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
savefig(fig, "03_swingup_metric.png")


# ---------------------------------------------------------------------------
# 4) Sample complexity comparison (bar chart)
# ---------------------------------------------------------------------------
if compare_mode:
    print("[4] Sample complexity bar chart ...")
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle("Sample complexity: trials to swing-up",
                 fontsize=12, fontweight="bold")

    bar_data = []
    bar_colours = []
    bar_labels = []
    for mk in method_keys:
        solved = trials_to_swingup(logs[mk], args.swingup_thresh)
        if solved is not None:
            bar_data.append(solved)
        else:
            n_hist = len(logs[mk].get("state_samples_history", []))
            bar_data.append(n_hist)  # max available (not solved)
        bar_colours.append(colours[mk])
        bar_labels.append(labels[mk])

    bars = ax.bar(bar_labels, bar_data, color=bar_colours, alpha=0.8,
                  edgecolor="black", linewidth=0.8)

    for i, (mk, val) in enumerate(zip(method_keys, bar_data)):
        solved = trials_to_swingup(logs[mk], args.swingup_thresh)
        lbl = str(val) if solved else f"{val}+ (not solved)"
        ax.text(i, val + 0.1, lbl, ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_ylabel("Trials (exploration + policy)")
    ax.set_ylim(0, max(bar_data) * 1.3 + 0.5)
    ax.grid(axis="y", alpha=0.3)

    # Param count annotation
    annot_lines = []
    for mk in method_keys:
        # Try to read param count from config
        if mk == "KAN" and args.kan:
            cfg_path = pathlib.Path(args.kan).parent / "config_log.pkl"
        elif mk == "RBF" and args.rbf:
            cfg_path = pathlib.Path(args.rbf).parent / "config_log.pkl"
        else:
            cfg_path = None
        if cfg_path and cfg_path.exists():
            try:
                with open(cfg_path, "rb") as f:
                    cfg = pkl.load(f)
                cli = cfg.get("cli_args", {})
                if mk == "KAN":
                    hidden = cli.get("hidden", [22])
                    grid = cli.get("grid", 5)
                    order = cli.get("order", 3)
                    annot_lines.append(
                        f"KAN: arch=[5]+{hidden}+[1], "
                        f"grid={grid}, order={order}")
                else:
                    annot_lines.append("RBF: 200 basis functions")
            except Exception:
                pass

    if annot_lines:
        ax.text(0.02, 0.97, "\n".join(annot_lines),
                transform=ax.transAxes, fontsize=8, va="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.8))

    plt.tight_layout()
    savefig(fig, "04_sample_complexity.png")
else:
    print("[4] Sample complexity bar chart skipped (single method).")


# ---------------------------------------------------------------------------
# 5) Cartpole animation
# ---------------------------------------------------------------------------
if not args.no_anim:
    from matplotlib.animation import FuncAnimation, PillowWriter

    # Animate the last (or selected) rollout of the first method listed
    mk0 = method_keys[0]
    hist = logs[mk0].get("state_samples_history", [])
    n_hist = len(hist)
    if n_hist > 0:
        print(f"[5] Cartpole animation ({mk0}) ...")
        trial_idx = args.trial if args.trial >= 0 else (n_hist - 1)
        trial_idx = max(0, min(trial_idx, n_hist - 1))
        states = np.asarray(hist[trial_idx])

        # Cart-pole dimensions
        cart_w, cart_h = 0.4, 0.2
        pole_len = 0.6

        step = max(1, args.anim_step)
        frame_indices = np.arange(0, len(states), step)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-1.0, 1.2)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", lw=0.5)

        cart_patch = FancyBboxPatch(
            (0, 0), cart_w, cart_h,
            boxstyle="round,pad=0.02",
            facecolor="steelblue", edgecolor="black", lw=1.5)
        ax.add_patch(cart_patch)
        pole_line, = ax.plot([], [], "o-", color="crimson",
                             lw=3, markersize=8, zorder=5)
        time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                            fontsize=10, fontweight="bold")
        cost_text = ax.text(0.02, 0.88, "", transform=ax.transAxes,
                            fontsize=9, color="purple")
        title_text = ax.set_title(
            f"{mk0} policy -- cartpole rollout {trial_idx}",
            fontsize=11, fontweight="bold")

        def init():
            cart_patch.set_xy((-cart_w / 2, -cart_h / 2))
            pole_line.set_data([], [])
            time_text.set_text("")
            cost_text.set_text("")
            return cart_patch, pole_line, time_text, cost_text

        def update(fnum):
            idx = frame_indices[fnum]
            x = states[idx, 0]
            theta = states[idx, 2]
            t = idx * T_sampling

            # Cart
            cart_patch.set_xy((x - cart_w / 2, -cart_h / 2))

            # Pole (theta=0 is down, theta=pi is up)
            pole_x = x + pole_len * np.sin(theta)
            pole_y = pole_len * np.cos(theta)
            pole_line.set_data([x, pole_x], [0, pole_y])

            # Cost
            cost = compute_real_cost(states[idx:idx+1])[0]
            time_text.set_text(f"t = {t:.2f} s")
            cost_text.set_text(f"cost = {cost:.3f}")

            return cart_patch, pole_line, time_text, cost_text

        anim = FuncAnimation(fig, update, init_func=init,
                             frames=len(frame_indices),
                             interval=50, blit=False)
        out_path = save_dir / "05_cartpole_animation.gif"
        anim.save(out_path, writer=PillowWriter(fps=20))
        plt.close(fig)
        print(f"  saved -> {out_path}")
    else:
        print("[5] No rollouts to animate.")
else:
    print("[5] Animation skipped (-no_anim).")


# ---------------------------------------------------------------------------
# 6) Learning curve overlay (final optimisation cost per trial)
# ---------------------------------------------------------------------------
if compare_mode:
    print("[6] Learning curve overlay ...")
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Learning curve: final optimisation cost per trial",
                 fontsize=12, fontweight="bold")

    for mk in method_keys:
        trials = logs[mk].get("cost_trial_list", [])
        if len(trials) == 0:
            continue
        final_costs = [np.asarray(t)[-1] for t in trials]
        min_costs = [np.asarray(t).min() for t in trials]
        trial_nums = np.arange(1, len(trials) + 1)
        ax.plot(trial_nums, final_costs, "o-", color=colours[mk],
                lw=2, markersize=8, alpha=0.9,
                label=f"{labels[mk]} (final cost)")
        ax.plot(trial_nums, min_costs, "s--", color=colours[mk],
                lw=1.2, markersize=6, alpha=0.5,
                label=f"{labels[mk]} (min cost)")

    ax.set_xlabel("Trial number")
    ax.set_ylabel("Optimisation cost")
    ax.set_xticks(range(1, max(len(logs[mk].get("cost_trial_list", []))
                               for mk in method_keys) + 1))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig(fig, "06_learning_curves.png")


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
if compare_mode:
    print("COMPARISON RESULT")
    print("=" * 70)
    for mk in method_keys:
        solved = trials_to_swingup(logs[mk], args.swingup_thresh)
        n_hist = len(logs[mk].get("state_samples_history", []))
        if solved is not None:
            print(f"  {labels[mk]:30s}: SOLVED at rollout {solved} "
                  f"(= {solved} real interactions)")
        else:
            print(f"  {labels[mk]:30s}: NOT SOLVED after {n_hist} rollouts")

    kan_solved = trials_to_swingup(logs.get("KAN", {}), args.swingup_thresh)
    rbf_solved = trials_to_swingup(logs.get("RBF", {}), args.swingup_thresh)
    if kan_solved and rbf_solved:
        diff = rbf_solved - kan_solved
        if diff > 0:
            print(f"\n  >> KAN is {diff} trial(s) more sample-efficient.")
        elif diff < 0:
            print(f"\n  >> RBF is {-diff} trial(s) more sample-efficient.")
        else:
            print(f"\n  >> Both solve in the same number of trials.")
else:
    print("SINGLE-METHOD EVALUATION COMPLETE")
    print("=" * 70)
    mk = method_keys[0]
    solved = trials_to_swingup(logs[mk], args.swingup_thresh)
    if solved is not None:
        print(f"  {labels[mk]}: SOLVED at rollout {solved}")
    else:
        n_hist = len(logs[mk].get("state_samples_history", []))
        print(f"  {labels[mk]}: NOT SOLVED after {n_hist} rollouts")

print(f"\nPlots saved to: {save_dir.resolve()}")
print("=" * 70)
