# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Test MC-PILCO on UR5 -- Variant K (per-particle KL vs GFN's TIME-VARYING
diffusion marginals).

Maps each control step to the corresponding GFN diffusion step and compares
each particle's GP Gaussian against the GFN's diffusion marginal at that step.

CAVEAT: the UR5 GFN was trained only on the terminal pose, so its diffusion
path goes zeros -> terminal pose (NOT along the q_ref circle). This variant
is therefore a GOAL-REACHING curriculum, not circle tracking. See
ur5_variant_k_cost.py docstring.

Requires MC_PILCO_GPStats. Resume + BPTT support same as Variant H/J.

Usage:
    python test_mcpilco_ur5_variant_k.py -seed 1 -num_trials 5
    python test_mcpilco_ur5_variant_k.py -seed 1 --bptt 10
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

parser = argparse.ArgumentParser("test ur5 variant K (time-varying GFN diffusion KL)")
parser.add_argument("-seed", type=int, default=1)
parser.add_argument("-checkpoint", type=str, default=None)
parser.add_argument("-num_trials", type=int, default=5)
parser.add_argument("-results_root", type=str, default="results_ur5_variant_k")
parser.add_argument("--bptt", type=int, default=None)
parser.add_argument("--fresh", action="store_true")
parser.add_argument("--weighting", type=str, default="uniform",
                    choices=["uniform", "linear", "quadratic"])
parser.add_argument("--beta", type=float, default=0.0)
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
# Environment & reference trajectory (identical to Variant H/J)
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

q_home  = np.zeros(6)
dq_home = np.zeros(6)
initial_state_mean = np.concatenate([q_home, dq_home])
print(f"[variant_k] initial state = zeros (aligned with GFN diffusion start)")

# ---------------------------------------------------------------------------
# Model learning (identical to Variant H/J)
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
# Policy (identical to Variant H/J)
# ---------------------------------------------------------------------------
print("\n---- Set exploration policy parameters ----")
f_rand_exploration_policy   = Policy.Random_exploration
rand_exploration_policy_par = dict(
    state_dim=state_dim, input_dim=input_dim,
    u_max=u_max, dtype=dtype, device=device,
)

print("\n---- Set control policy parameters ----")
num_basis = 100
f_control_policy = Policy.Sum_of_gaussians
control_policy_par = dict(
    state_dim=state_dim, input_dim=input_dim,
    u_max=u_max, num_basis=num_basis,
    dtype=dtype, device=device,
)
centers_init = initial_state_mean[np.newaxis, :] + \
               0.5 * np.random.randn(num_basis, state_dim)
control_policy_par["centers_init"]      = centers_init
control_policy_par["lengthscales_init"] = 2.0 * np.ones(state_dim)
control_policy_par["weight_init"]       = 0.01 * (np.random.rand(input_dim, num_basis) - 0.5) * \
                                          u_max.reshape(-1, 1)
control_policy_par["flg_squash"]        = True
control_policy_par["flg_drop"]          = True
policy_reinit_dict = dict(
    lenghtscales_par=control_policy_par["lengthscales_init"],
    centers_par=np.abs(initial_state_mean) + 1.0,
    weight_par=float(u_max.max()),
)

# ---------------------------------------------------------------------------
# Cost (Variant K)
# ---------------------------------------------------------------------------
print(f"\n---- Set cost function (Variant K: time-varying GFN diffusion KL) ----")
from policy_learning.ur5_variant_k_cost import UR5_VariantK_Cost
f_cost_function = UR5_VariantK_Cost
cost_function_par = dict(
    checkpoint_path=args.checkpoint,
    q_ref=q_ref, dq_ref=dq_ref, T_control=T_control,
    alpha=5.0, epsilon=0.10,
    weighting=args.weighting,
    beta=args.beta,
    u_max=tuple(u_max),
    q_min=tuple(ur5_env.q_min), q_max=tuple(ur5_env.q_max),
    dtype=dtype, device=device,
)

# ---------------------------------------------------------------------------
log_dir  = pathlib.Path(args.results_root) / f"{seed}"
log_file = log_dir / "log.pkl"
log_dir.mkdir(parents=True, exist_ok=True)
print(f"\n[variant_k] log_dir (abs): {log_dir.resolve()}")

# SAFE-SAVE PATCH
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
        print(f"[safe_save] WARNING: atomic write failed ({e}); falling back.")
    _orig_pkl_dump(obj, file_obj, *a, **kw)
_MC_PILCO_module.pkl.dump = _safe_pkl_dump
print("[variant_k] safe-save patch active.")

print("\n---- Init policy learning object (MC_PILCO_GPStats) ----")
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

from policy_learning.mc_pilco_gp_stats import MC_PILCO_GPStats
PL_obj = MC_PILCO_GPStats(bptt_truncate=args.bptt, **MC_PILCO_init_dict)
PL_obj.system = UR5_Model(env=ur5_env)
print(f"[variant_k] Replaced PL_obj.system with UR5_Model.")

# Resume detection
resume_from = 0
if log_file.exists() and not args.fresh:
    try:
        with open(log_file, "rb") as f:
            existing_log = pkl.load(f)
        n_completed = len(existing_log.get("cost_trial_list", []))
        n_real      = len(existing_log.get("state_samples_history", []))
        print(f"\n[resume] {n_completed} completed policy trials, "
              f"{n_real} real rollouts.")
        if n_completed >= num_trials_requested:
            print(f"[resume] Already have {n_completed} trials. Nothing to do.")
            sys.exit(0)
        n_loadable = min(n_completed, max(n_real - 1, 0))
        if n_loadable > 0:
            PL_obj.load_model_from_log(num_trial=n_loadable, folder=str(log_dir) + "/")
            resume_from = n_loadable
            print(f"[resume] Loaded through trial {n_loadable}.")
    except Exception as e:
        print(f"[resume] WARNING: failed to load log ({e}); starting fresh.")
elif args.fresh and log_file.exists():
    print(f"\n[resume] --fresh given; ignoring existing log.")

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
    lr_min=0.001,
    policy_reinit_dict=policy_reinit_dict,
)

reinforce_param_dict = dict(
    initial_state=initial_state_mean,
    initial_state_var=1e-4 * np.ones(state_dim),
    T_exploration=T_exploration,
    T_control=T_control,
    num_trials=num_trials_requested,
    model_optimization_opt_list=model_optimization_opt_list,
    policy_optimization_dict=policy_optimization_dict,
)

print("\n---- Save test configuration ----")
config_log_dict = dict(
    MC_PILCO_init_dict=MC_PILCO_init_dict,
    reinforce_param_dict=reinforce_param_dict,
)
pkl.dump(config_log_dict, open(str(log_dir / "config_log.pkl"), "wb"))

print(f"\n---- Start learning (Variant K) ----")
if resume_from > 0:
    reinforce_param_dict["loaded_model"] = True
PL_obj.reinforce(**reinforce_param_dict)
