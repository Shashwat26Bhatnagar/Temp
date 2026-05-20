# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Test MC-PILCO on UR5 trajectory tracking - Variant D (Wasserstein-2 cost).

State  x = [q1..q6, dq1..dq6]    (12-D)
Action u = [tau1..tau6]           (6-D joint torques, Nm)
dt       = 0.02 s
T_control= 4.0 s   (one full revolution of the reference circle, N_h = 200)

Cost:
    L = sum_t  w_t * W2^2( p_GFN(t), q_t )  +  alpha * sum_t  slack_t

with TIME-VARYING target mu_p(t) sliding along the reference q_ref(t).

W2^2 for diagonal Gaussians has closed form:
    || mu_p - mu_q ||^2 + || sigma_p - sigma_q ||^2

so the mean term is EXACTLY the pointwise tracking distance, while the
sigma term penalises over- and under-dispersion symmetrically.

Compared to Variant C (reverse KL):
  * No sigma_q in denominator -> no variance-inflation cheat
  * Symmetric (no fwd/rev KL ambiguity)
  * Numerically stable
  * Faster per opt step (no log, no division)
  * Pointwise tracking signal built in

Tighter target sigmas (0.05 rad / 0.20 rad/s) are used here to enforce
trajectory tracking rather than terminal-state matching.
"""

import argparse
import os
import pathlib
import pickle as pkl

import numpy as np
import torch

import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood
import gpr_lib.Utils.Parameters_covariance_functions as cov_func   # needed by MPK kernel
import model_learning.Model_learning as ML
import policy_learning.MC_PILCO as MC_PILCO
import policy_learning.Policy as Policy

from simulation_class.ur5_system import UR5System
from simulation_class.ur5_trajectory import generate_circle_trajectory
from simulation_class.ur5_model_wrapper import UR5_Model

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser("test ur5 variant D (Wasserstein-2, time-varying target)")
parser.add_argument("-seed", type=int, default=1, help="random seed")
parser.add_argument("-checkpoint", type=str, default=None,
                    help="Path to ur5 GFN checkpoint .pt (defaults to step25000)")
parser.add_argument("-num_trials", type=int, default=5,
                    help="Total MC-PILCO trials (default 5)")
args = parser.parse_args()

if args.checkpoint is None:
    repo_root  = pathlib.Path(__file__).resolve().parent.parent
    args.checkpoint = str(repo_root / 'GflowNet' / 'gfn-diffusion' / 'energy_sampling'
                          / 'ur5_denoising_theta_step25000.pt')

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)

dtype  = torch.float64
device = torch.device("cpu")
torch.set_num_threads(1)

# ---------------------------------------------------------------------------
# Environment parameters
# ---------------------------------------------------------------------------
print("---- Set environment parameters ----")
num_trials     = args.num_trials
T_sampling     = 0.02            # MC-PILCO time step (s)
T_exploration  = 4.0             # First trial: 4 s of random exploration
T_control      = 4.0             # Each control trial: 4 s (one circle)
state_dim      = 12              # [q (6), dq (6)]
input_dim      = 6               # 6 joint torques
num_gp         = 6               # one GP per joint (speed model)
gp_input_dim   = state_dim + input_dim   # 12 + 6 = 18
u_max          = np.array([150., 150., 100., 28., 28., 28.])   # tau limits (XML)

std_noise = 1e-2
std_list  = std_noise * np.ones(state_dim)

# UR5 environment as the "true" simulator.
# We supply a dummy f_sim because MC_PILCO's constructor wraps it with
# simulation_class.model.Model, but immediately afterwards we replace
# `PL_obj.system` with our MuJoCo-aware UR5_Model.
ur5_env = UR5System()
def f_sim(state, t, u):
    # Never actually invoked -- placeholder so MC_PILCO.__init__ succeeds.
    return np.zeros_like(state)

# Reference trajectory
print("\n---- Generate reference trajectory ----")
t_arr, q_ref, dq_ref, ee_ref = generate_circle_trajectory(
    center=(-0.6, 0.0, 0.4), radius=0.15,
    duration=T_control, dt=T_sampling, env=ur5_env)
print(f"Reference: {q_ref.shape[0]} waypoints, "
      f"end-effector circle radius 0.15 m around (-0.6, 0, 0.4)")

q_home  = q_ref[0]                # start of circle = end of circle (closed loop)
dq_home = dq_ref[0]
initial_state_mean = np.concatenate([q_home, dq_home])

# ---------------------------------------------------------------------------
# Model learning parameters
# ---------------------------------------------------------------------------
print("\n---- Set model learning parameters ----")
f_model_learning   = ML.Speed_Model_learning_RBF_MPK_angle_state
model_learning_par = {}
model_learning_par["num_gp"]            = num_gp
model_learning_par["angle_indeces"]     = []                          # no angle wrap
model_learning_par["not_angle_indeces"] = list(range(12))             # all state dims
model_learning_par["T_sampling"]        = T_sampling
model_learning_par["vel_indeces"]       = [6, 7, 8, 9, 10, 11]
model_learning_par["not_vel_indeces"]   = [0, 1, 2, 3, 4, 5]
model_learning_par["device"]            = device
model_learning_par["dtype"]             = dtype
model_learning_par["approximation_mode"] = "SOD"
model_learning_par["approximation_dict"] = {
    "SOD_threshold_mode": "relative",
    "SOD_threshold": 0.5,
    "flg_SOD_permutation": False,
}

# Kernel RBF initial parameters
init_dict_RBF = {}
init_dict_RBF["active_dims"]            = np.arange(0, gp_input_dim)
init_dict_RBF["lengthscales_init"]      = np.ones(gp_input_dim)
init_dict_RBF["flg_train_lengthscales"] = True
init_dict_RBF["lambda_init"]            = np.ones(1)
init_dict_RBF["flg_train_lambda"]       = False
init_dict_RBF["sigma_n_init"]           = 1 * np.ones(1)
init_dict_RBF["flg_train_sigma_n"]      = True
init_dict_RBF["sigma_n_num"]            = None
init_dict_RBF["dtype"]                  = dtype
init_dict_RBF["device"]                 = device

# Kernel MPK initial parameters
init_dict_MPK = {}
init_dict_MPK["active_dims"]                  = np.arange(0, gp_input_dim)
init_dict_MPK["poly_deg"]                     = 2
init_dict_MPK["Sigma_pos_par_init_list"]      = [np.ones(gp_input_dim + 1)] + [
    np.ones((deg + 1) * gp_input_dim) for deg in range(1, init_dict_MPK["poly_deg"])
]
init_dict_MPK["flg_train_Sigma_pos_par_list"] = [True] * init_dict_MPK["poly_deg"]
init_dict_MPK["dtype"]                        = dtype
init_dict_MPK["device"]                       = device

model_learning_par["init_dict_list"] = [[init_dict_RBF, init_dict_MPK]] * num_gp

# ---------------------------------------------------------------------------
# Exploration & control policy
# ---------------------------------------------------------------------------
print("\n---- Set exploration policy parameters ----")
f_rand_exploration_policy  = Policy.Random_exploration
rand_exploration_policy_par = {
    "state_dim": state_dim, "input_dim": input_dim,
    "u_max":      u_max,
    "dtype":      dtype,    "device":    device,
}

print("\n---- Set control policy parameters ----")
num_basis = 100   # reduced from 200 -- 12-D state needs less basis density
f_control_policy = Policy.Sum_of_gaussians       # no angle wraparound needed
control_policy_par = {}
control_policy_par["state_dim"]         = state_dim
control_policy_par["input_dim"]         = input_dim
control_policy_par["u_max"]             = u_max
control_policy_par["num_basis"]         = num_basis
control_policy_par["dtype"]             = dtype
control_policy_par["device"]            = device

# Initialise basis-function centres around the home configuration.
# centers_init shape must be [num_basis, state_dim] = [num_basis, 12].
centers_init = initial_state_mean[np.newaxis, :] + \
               0.5 * np.random.randn(num_basis, state_dim)
control_policy_par["centers_init"]      = centers_init
# Lengthscales larger -> smoother policy (less likely to give extreme outputs)
control_policy_par["lengthscales_init"] = 2.0 * np.ones(state_dim)
# *** KEY FIX ***  initial weights ~0 -> policy starts at near-zero torque,
# preventing particle states from blowing up during the first GP rollouts.
control_policy_par["weight_init"]       = 0.01 * (np.random.rand(input_dim, num_basis) - 0.5) * \
                                          u_max.reshape(-1, 1)
control_policy_par["flg_squash"]        = True
control_policy_par["flg_drop"]          = True

policy_reinit_dict = {
    "lenghtscales_par": control_policy_par["lengthscales_init"],
    "centers_par":      np.abs(initial_state_mean) + 1.0,   # span of state-space
    "weight_par":       float(u_max.max()),
}

# ---------------------------------------------------------------------------
# Cost function (Variant C: reverse KL + chance constraints)
# ---------------------------------------------------------------------------
print("\n---- Set cost function (Variant D: W2^2 + chance constraints) ----")
from policy_learning.ur5_variant_d_cost import UR5_VariantD_Cost
f_cost_function = UR5_VariantD_Cost
cost_function_par = {
    'checkpoint_path': args.checkpoint,
    'q_ref':           q_ref,
    'dq_ref':          dq_ref,
    'T_control':       T_control,
    'alpha':           5.0,
    'epsilon':         0.10,
    'weighting':       'quadratic',
    # Tighter target sigmas than the GFN's trained (0.10 / 0.50) values to
    # enforce trajectory tracking rather than terminal-state matching.
    'sigma_p_q':       0.05,         # 0.05 rad   (was 0.10 in Variant C)
    'sigma_p_dq':      0.20,         # 0.20 rad/s (was 0.50 in Variant C)
    'q_min':           tuple(ur5_env.q_min),
    'q_max':           tuple(ur5_env.q_max),
    'dtype':           dtype,
    'device':          device,
}

# ---------------------------------------------------------------------------
# MC-PILCO init
# ---------------------------------------------------------------------------
print("\n---- Init policy learning object ----")
MC_PILCO_init_dict = {
    "T_sampling":                 T_sampling,
    "state_dim":                  state_dim,
    "input_dim":                  input_dim,
    "f_sim":                      f_sim,
    "std_meas_noise":             np.array(std_list),
    "f_model_learning":           f_model_learning,
    "model_learning_par":         model_learning_par,
    "f_rand_exploration_policy":  f_rand_exploration_policy,
    "rand_exploration_policy_par":rand_exploration_policy_par,
    "f_control_policy":           f_control_policy,
    "control_policy_par":         control_policy_par,
    "f_cost_function":            f_cost_function,
    "cost_function_par":          cost_function_par,
    "log_path":                   f"results_ur5_variant_d_5/{seed}",
    "dtype":                      dtype,
    "device":                     device,
}
PL_obj = MC_PILCO.MC_PILCO(**MC_PILCO_init_dict)

# Replace MC-PILCO's internal scipy.odeint-based Model with our MuJoCo
# UR5 wrapper so real-system rollouts use the MuJoCo physics.
PL_obj.system = UR5_Model(env=ur5_env)
print(f"[test_mcpilco_ur5_variant_c] Replaced PL_obj.system with UR5_Model "
      f"(MuJoCo-based, dt={ur5_env.CONTROL_DT}s)")

# ---------------------------------------------------------------------------
# Training options
# ---------------------------------------------------------------------------
print("\n---- Set MC-PILCO options ----")
model_optimization_opt_dict = {
    "f_optimizer":   "lambda p : torch.optim.Adam(p, lr = 0.01)",
    "criterion":     Likelihood.Marginal_log_likelihood,
    "N_epoch":       1501,
    "N_epoch_print": 500,
}
model_optimization_opt_list = [model_optimization_opt_dict] * num_gp

policy_optimization_dict = {
    "num_particles":     50,         # reduced further to lower memory pressure
    "opt_steps_list":    [1000] + [2000] * (num_trials - 1),   # smaller per-trial budget
    "lr_list":           [0.005] * num_trials,                 # lower LR for 12-D
    "f_optimizer":       "lambda p, lr : torch.optim.Adam(p, lr)",
    "num_step_print":    100,
    "p_dropout_list":    [0.25] * num_trials,
    "p_drop_reduction":  0.25 / 2,
    "alpha_diff_cost":   0.99,
    "min_diff_cost":     0.08,
    "num_min_diff_cost": 200,
    "min_step":          200,
    "lr_min":            0.0025,
    "policy_reinit_dict":policy_reinit_dict,
}

# initial state distribution
reinforce_param_dict = {
    "initial_state":      initial_state_mean,
    "initial_state_var":  1e-4 * np.ones(state_dim),
    "T_exploration":      T_exploration,
    "T_control":          T_control,
    "num_trials":         num_trials,
    "model_optimization_opt_list": model_optimization_opt_list,
    "policy_optimization_dict":    policy_optimization_dict,
}

print("\n---- Save test configuration ----")
log_dir = pathlib.Path("results_ur5_variant_d_5") / str(seed)
log_dir.mkdir(parents=True, exist_ok=True)
config_log_dict = {
    "MC_PILCO_init_dict":   MC_PILCO_init_dict,
    "reinforce_param_dict": reinforce_param_dict,
    "q_ref":  q_ref,
    "dq_ref": dq_ref,
    "ee_ref": ee_ref,
}
with open(log_dir / "config_log.pkl", "wb") as f:
    pkl.dump(config_log_dict, f)

# ---------------------------------------------------------------------------
# Start training
# ---------------------------------------------------------------------------
PL_obj.reinforce(**reinforce_param_dict)
