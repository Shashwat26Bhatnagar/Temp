# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Test MC-PILCO on UR5 trajectory tracking -- Variant H with KAN policy.

Identical to test_mcpilco_ur5_variant_h.py EXCEPT the control policy:
  * Original: Sum_of_gaussians (100 RBF basis, ~1812 params)
  * This:     KAN_Policy        (B-spline edges, comparable params)

The goal is to compare sample complexity (trials to track) between
the RBF and KAN policy representations under the same MC-PILCO
framework, GP model, cost function, and optimizer settings.

KAN parameter count matching:
  RBF(100 basis):  log_lengthscales(12) + centers(100x12) + weights(6x100) = 1812
  KAN[12,11,6]:    edges(198) x params_per_edge(9) = 1782   [grid=5, order=3]

Usage:
    python test_mcpilco_ur5_variant_h_kan.py -seed 1
    python test_mcpilco_ur5_variant_h_kan.py -seed 1 --bptt 1 --beta 0.1
    python test_mcpilco_ur5_variant_h_kan.py -seed 1 -hidden 11 -grid 5
    python test_mcpilco_ur5_variant_h_kan.py -seed 1 -hidden 8 8  # deeper KAN
"""

import argparse
import os
import pathlib
import pickle as pkl
import sys

import numpy as np
import torch

import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood
import gpr_lib.Utils.Parameters_covariance_functions as cov_func
import model_learning.Model_learning as ML
import policy_learning.Policy as Policy

from simulation_class.ur5_system import UR5System
from simulation_class.ur5_trajectory import generate_circle_trajectory
from simulation_class.ur5_model_wrapper import UR5_Model

from policy_learning.Policy_KAN import KAN_Policy


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser("test ur5 variant H KAN")
parser.add_argument("-seed", type=int, default=1)
parser.add_argument("-checkpoint", type=str, default=None,
                    help="GFN checkpoint .pt (defaults to step25000).")
parser.add_argument("-num_trials", type=int, default=5,
                    help="TOTAL desired number of policy-optimisation trials.")
parser.add_argument("-results_root", type=str,
                    default="results_ur5_variant_h_kan",
                    help="Root directory for results.")
parser.add_argument("-hidden", type=int, nargs='+', default=[11],
                    help="KAN hidden layer sizes (e.g. 11 or 8 8 for 2 hidden)")
parser.add_argument("-grid", type=int, default=5,
                    help="B-spline grid size (default 5)")
parser.add_argument("-order", type=int, default=3,
                    help="B-spline order (default 3 = cubic)")
parser.add_argument("--bptt", type=int, default=None,
                    help="BPTT truncation length (steps). None = full BPTT. "
                         "1 = pure 1-step lookahead policy gradient.")
parser.add_argument("--fresh", action="store_true",
                    help="Ignore existing log.pkl and start over.")
parser.add_argument("--weighting", type=str, default="uniform",
                    choices=["uniform", "linear", "quadratic"])
parser.add_argument("--beta", type=float, default=0.0,
                    help="Action regularisation weight. 0 = off. "
                         "Try 0.01-1.0 to penalise large torques.")
args = parser.parse_args()

if args.checkpoint is None:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    args.checkpoint = str(repo_root / 'GflowNet' / 'gfn-diffusion'
                          / 'energy_sampling'
                          / 'ur5_denoising_theta_step25000.pt')

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)

dtype  = torch.float64
device = torch.device("cpu")
torch.set_num_threads(1)

# ---------------------------------------------------------------------------
# Environment & reference trajectory
# ---------------------------------------------------------------------------
print("---- Set environment parameters ----")
num_trials_requested = args.num_trials
T_sampling     = 0.02
T_exploration  = 4.0
T_control      = 4.0
state_dim      = 12
input_dim      = 6
num_gp         = 6
gp_input_dim   = state_dim + input_dim
u_max          = np.array([150., 150., 100., 28., 28., 28.])

std_noise = 1e-2
std_list  = std_noise * np.ones(state_dim)

ur5_env = UR5System()
def f_sim(state, t, u):
    return np.zeros_like(state)

print("\n---- Generate reference trajectory ----")
t_arr, q_ref, dq_ref, ee_ref = generate_circle_trajectory(
    center=(-0.6, 0.0, 0.4), radius=0.15,
    duration=T_control, dt=T_sampling, env=ur5_env)
print(f"Reference: {q_ref.shape[0]} waypoints, "
      f"end-effector circle radius 0.15 m around (-0.6, 0, 0.4)")

# ---------------------------------------------------------------------------
# Initial state -- ZEROS (same as RBF baseline variant H)
# ---------------------------------------------------------------------------
q_home  = np.zeros(6)
dq_home = np.zeros(6)
initial_state_mean = np.concatenate([q_home, dq_home])

_d_start_to_ref0 = float(np.linalg.norm(q_ref[0] - q_home))
print(f"[variant_h_kan] initial state = zeros (aligned with GFN)")
print(f"[variant_h_kan] ||zeros - q_ref[0]|| = {_d_start_to_ref0:.3f} rad")

# ---------------------------------------------------------------------------
# Model learning params (identical to RBF baseline)
# ---------------------------------------------------------------------------
print("\n---- Set model learning parameters ----")
f_model_learning   = ML.Speed_Model_learning_RBF_MPK_angle_state
model_learning_par = dict(
    num_gp=num_gp,
    angle_indeces=[],
    not_angle_indeces=list(range(12)),
    T_sampling=T_sampling,
    vel_indeces=[6, 7, 8, 9, 10, 11],
    not_vel_indeces=[0, 1, 2, 3, 4, 5],
    device=device, dtype=dtype,
    approximation_mode="SOD",
    approximation_dict=dict(
        SOD_threshold_mode="relative",
        SOD_threshold=0.5,
        flg_SOD_permutation=False,
    ),
)

init_dict_RBF = dict(
    active_dims=np.arange(0, gp_input_dim),
    lengthscales_init=np.ones(gp_input_dim),
    flg_train_lengthscales=True,
    lambda_init=np.ones(1),
    flg_train_lambda=False,
    sigma_n_init=1 * np.ones(1),
    flg_train_sigma_n=True,
    sigma_n_num=None,
    dtype=dtype, device=device,
)
init_dict_MPK = dict(
    active_dims=np.arange(0, gp_input_dim),
    poly_deg=2,
    Sigma_pos_par_init_list=[np.ones(gp_input_dim + 1)] + [
        np.ones((deg + 1) * gp_input_dim) for deg in range(1, 2)
    ],
    flg_train_Sigma_pos_par_list=[True] * 2,
    dtype=dtype, device=device,
)
model_learning_par["init_dict_list"] = [[init_dict_RBF, init_dict_MPK]] * num_gp

# ---------------------------------------------------------------------------
# Exploration policy (identical to RBF baseline)
# ---------------------------------------------------------------------------
print("\n---- Set exploration policy parameters ----")
f_rand_exploration_policy   = Policy.Random_exploration
rand_exploration_policy_par = dict(
    state_dim=state_dim, input_dim=input_dim,
    u_max=u_max, dtype=dtype, device=device,
)

# ---------------------------------------------------------------------------
# Control policy: KAN (replaces Sum_of_gaussians)
# ---------------------------------------------------------------------------
print("\n---- Set KAN control policy parameters ----")
f_control_policy = KAN_Policy
control_policy_par = dict(
    state_dim=state_dim,
    input_dim=input_dim,
    hidden_sizes=args.hidden,
    grid_size=args.grid,
    spline_order=args.order,
    grid_range=(-4.0, 4.0),
    flg_squash=True,
    u_max=u_max,
    flg_drop=True,
    dtype=dtype,
    device=device,
)

# For KAN, reinit just re-randomises weights; these keys are accepted
# but ignored (interface compat with Sum_of_gaussians.reinit).
policy_reinit_dict = dict(
    lenghtscales_par=2.0 * np.ones(state_dim),
    centers_par=np.abs(initial_state_mean) + 1.0,
    weight_par=float(u_max.max()),
)

# ---------------------------------------------------------------------------
# Cost (Variant H -- identical to RBF baseline)
# ---------------------------------------------------------------------------
print(f"\n---- Set cost function (Variant H: per-step local Mahalanobis) ----")
from policy_learning.ur5_variant_h_cost import UR5_VariantH_Cost
f_cost_function = UR5_VariantH_Cost
cost_function_par = dict(
    checkpoint_path=args.checkpoint,
    q_ref=q_ref, dq_ref=dq_ref, T_control=T_control,
    alpha=5.0, epsilon=0.10,
    weighting=args.weighting,
    sigma_p_q=None,    # use GFN-trained 0.10
    sigma_p_dq=None,   # use GFN-trained 0.50
    beta=args.beta,
    u_max=tuple(u_max),
    q_min=tuple(ur5_env.q_min), q_max=tuple(ur5_env.q_max),
    dtype=dtype, device=device,
)

# ---------------------------------------------------------------------------
# Build the MC_PILCO object
# ---------------------------------------------------------------------------
log_dir  = pathlib.Path(args.results_root) / f"{seed}"
log_file = log_dir / "log.pkl"
log_dir.mkdir(parents=True, exist_ok=True)

log_dir_abs  = log_dir.resolve()
log_file_abs = log_file.resolve()
print(f"\n[variant_h_kan] CWD            : {pathlib.Path.cwd()}")
print(f"[variant_h_kan] log_dir (rel)  : {log_dir}")
print(f"[variant_h_kan] log_dir (abs)  : {log_dir_abs}")
print(f"[variant_h_kan] log_file (abs) : {log_file_abs}")

# ---------------------------------------------------------------------------
# SAFE-SAVE PATCH (same as RBF baseline)
# ---------------------------------------------------------------------------
import policy_learning.MC_PILCO as _MC_PILCO_module
_orig_pkl_dump = _MC_PILCO_module.pkl.dump

def _safe_pkl_dump(obj, file_obj, *a, **kw):
    try:
        path = getattr(file_obj, "name", None)
        if isinstance(path, str) and file_obj.mode == "wb":
            tmp_path = path + ".tmp"
            try:
                file_obj.close()
            except Exception:
                pass
            with open(tmp_path, "wb") as tmp_f:
                _orig_pkl_dump(obj, tmp_f, *a, **kw)
                tmp_f.flush()
                os.fsync(tmp_f.fileno())
            os.replace(tmp_path, path)
            print(f"[safe_save] wrote {path} ({os.path.getsize(path)} bytes)")
            return
    except Exception as e:
        print(f"[safe_save] WARNING: atomic write failed ({e}); "
              f"falling back to direct dump.")
    _orig_pkl_dump(obj, file_obj, *a, **kw)

_MC_PILCO_module.pkl.dump = _safe_pkl_dump
print("[variant_h_kan] safe-save patch active.")

print("\n---- Init policy learning object ----")
MC_PILCO_init_dict = dict(
    T_sampling=T_sampling,
    state_dim=state_dim, input_dim=input_dim,
    f_sim=f_sim,
    std_meas_noise=np.array(std_list),
    f_model_learning=f_model_learning,
    model_learning_par=model_learning_par,
    f_rand_exploration_policy=f_rand_exploration_policy,
    rand_exploration_policy_par=rand_exploration_policy_par,
    f_control_policy=f_control_policy,
    control_policy_par=control_policy_par,
    f_cost_function=f_cost_function,
    cost_function_par=cost_function_par,
    log_path=str(log_dir),
    dtype=dtype, device=device,
)

if args.bptt is not None:
    from policy_learning.mc_pilco_local import MC_PILCO_Local
    PL_obj = MC_PILCO_Local(bptt_truncate=args.bptt, **MC_PILCO_init_dict)
else:
    import policy_learning.MC_PILCO as MC_PILCO
    PL_obj = MC_PILCO.MC_PILCO(**MC_PILCO_init_dict)

PL_obj.system = UR5_Model(env=ur5_env)
print(f"[variant_h_kan] Replaced PL_obj.system with UR5_Model (MuJoCo, "
      f"dt={ur5_env.CONTROL_DT}s)")

# ---------------------------------------------------------------------------
# Resume detection (same as RBF baseline)
# ---------------------------------------------------------------------------
resume_from = 0
if log_file.exists() and not args.fresh:
    try:
        with open(log_file, "rb") as f:
            existing_log = pkl.load(f)
        n_completed = len(existing_log.get("cost_trial_list", []))
        n_real      = len(existing_log.get("state_samples_history", []))
        print(f"\n[resume] Found existing log: "
              f"{n_completed} completed policy trials, "
              f"{n_real} real-system rollouts.")
        if n_completed >= num_trials_requested:
            print(f"[resume] Already have {n_completed} trials (>= "
                  f"requested {num_trials_requested}). Nothing to do.")
            sys.exit(0)
        n_loadable = min(n_completed, max(n_real - 1, 0))
        if n_loadable != n_completed:
            print(f"[resume] WARNING: cost_trial_list has {n_completed} "
                  f"entries but state_samples_history has only {n_real}. "
                  f"Clamping to {n_loadable} loadable trials.")
        if n_loadable > 0:
            PL_obj.load_model_from_log(num_trial=n_loadable,
                                       folder=str(log_dir) + "/")
            resume_from = n_loadable
            print(f"[resume] Loaded state through trial {n_loadable}; "
                  f"will run {num_trials_requested - n_loadable} more.")
        else:
            print(f"[resume] Existing log has no loadable trials -- "
                  f"starting fresh.")
    except Exception as e:
        print(f"[resume] WARNING: failed to load log ({e}); starting fresh.")
        resume_from = 0
elif args.fresh and log_file.exists():
    print(f"\n[resume] --fresh given; ignoring existing log at {log_file}")

# ---------------------------------------------------------------------------
# MC-PILCO options (identical to RBF baseline)
# ---------------------------------------------------------------------------
print("\n---- Set MC-PILCO options ----")
model_optimization_opt_dict = dict(
    f_optimizer="lambda p : torch.optim.Adam(p, lr = 0.01)",
    criterion=Likelihood.Marginal_log_likelihood,
    N_epoch=1501,
    N_epoch_print=500,
)
model_optimization_opt_list = [model_optimization_opt_dict] * num_gp

policy_optimization_dict = dict(
    num_particles=50,
    opt_steps_list=[1000] + [2000] * (num_trials_requested - 1),
    lr_list=[0.005] * num_trials_requested,
    f_optimizer="lambda p, lr : torch.optim.Adam(p, lr)",
    num_step_print=100,
    p_dropout_list=[0.25] * num_trials_requested,
    p_drop_reduction=0.25 / 2,
    alpha_diff_cost=0.99,
    min_diff_cost=0.08,
    num_min_diff_cost=200,
    min_step=200,
    lr_min=0.0025,
    policy_reinit_dict=policy_reinit_dict,
)

reinforce_param_dict = dict(
    initial_state=initial_state_mean,
    initial_state_var=1e-4 * np.ones(state_dim),
    T_exploration=T_exploration,
    T_control=T_control,
    num_trials=num_trials_requested - resume_from,
    model_optimization_opt_list=model_optimization_opt_list,
    policy_optimization_dict=policy_optimization_dict,
    loaded_model=(resume_from > 0),
)

# ---------------------------------------------------------------------------
# Save config snapshot
# ---------------------------------------------------------------------------
print("\n---- Save test configuration ----")
config_log_dict = dict(
    MC_PILCO_init_dict=MC_PILCO_init_dict,
    reinforce_param_dict=reinforce_param_dict,
    q_ref=q_ref, dq_ref=dq_ref, ee_ref=ee_ref,
    cli_args=vars(args),
)
with open(log_dir / "config_log.pkl", "wb") as f:
    pkl.dump(config_log_dict, f)

# ---------------------------------------------------------------------------
# Print comparison info
# ---------------------------------------------------------------------------
n_kan = sum(pp.numel() for pp in PL_obj.control_policy.parameters()
            if pp.requires_grad)
print(f"\n{'='*60}")
print(f"UR5 VARIANT H -- KAN EXPERIMENT")
print(f"{'='*60}")
print(f"KAN architecture : [12] + {args.hidden} + [6]")
print(f"Grid size        : {args.grid}")
print(f"Spline order     : {args.order}")
print(f"Grid range       : (-4.0, 4.0)")
print(f"KAN params       : {n_kan}")
print(f"RBF baseline     : ~1812 params (100 basis)")
print(f"Param ratio      : {n_kan / 1812:.3f}x")
print(f"BPTT truncation  : {args.bptt}")
print(f"Action penalty   : beta={args.beta}")
print(f"Results dir      : {log_dir_abs}")
print(f"{'='*60}")

# ---------------------------------------------------------------------------
# Snapshot helper
# ---------------------------------------------------------------------------
def _snapshot(reason=""):
    if not hasattr(PL_obj, "log_dict"):
        return
    PL_obj.log_dict["state_samples_history"]    = PL_obj.state_samples_history
    PL_obj.log_dict["input_samples_history"]    = PL_obj.input_samples_history
    PL_obj.log_dict["noiseless_states_history"] = PL_obj.noiseless_states_history
    tmp = str(log_file_abs) + ".tmp"
    with open(tmp, "wb") as f:
        _orig_pkl_dump(PL_obj.log_dict, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, str(log_file_abs))
    print(f"[snapshot] {reason} -> {log_file_abs} "
          f"({os.path.getsize(log_file_abs)} bytes)")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print(f"\n[variant_h_kan] Starting reinforce() with "
      f"num_trials={reinforce_param_dict['num_trials']}, "
      f"loaded_model={reinforce_param_dict['loaded_model']}")

_orig_get_data = PL_obj.get_data_from_system
def _get_data_with_snapshot(*a, **kw):
    out = _orig_get_data(*a, **kw)
    _snapshot(reason=f"after rollout (trial_index={kw.get('trial_index','?')}, "
                     f"exploration={kw.get('flg_exploration','?')})")
    return out
PL_obj.get_data_from_system = _get_data_with_snapshot

try:
    PL_obj.reinforce(**reinforce_param_dict)
finally:
    try:
        _snapshot(reason="final (post-reinforce or post-exception)")
    except Exception as e:
        print(f"[snapshot] final save FAILED: {e}")

print(f"\n[variant_h_kan] Done. Log saved to {log_file_abs}")
