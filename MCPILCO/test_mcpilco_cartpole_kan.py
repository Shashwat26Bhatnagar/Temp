# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Test MC-PILCO on cartpole swing-up with KAN policy.

Identical to test_mcpilco_cartpole.py EXCEPT the control policy:
  * Original: Sum_of_gaussians_with_angles (200 RBF basis, ~1205 params)
  * This:     KAN_Policy_with_angles       (B-spline edges, comparable params)

The goal is to compare sample complexity (trials to solve) between
the RBF and KAN policy representations under the same MC-PILCO
framework, GP model, cost function, and optimizer settings.

KAN parameter count matching:
  RBF(200 basis):  log_lengthscales(5) + centers(200x5) + weights(1x200) = 1205
  KAN[5,22,1]:     edges(132) x params_per_edge(9) = 1188   [grid=5, order=3]

Usage:
    python test_mcpilco_cartpole_kan.py -seed 1
    python test_mcpilco_cartpole_kan.py -seed 1 -hidden 22 -grid 5
    python test_mcpilco_cartpole_kan.py -seed 1 -hidden 10 10   # deeper KAN
"""

import argparse
import pickle as pkl

import numpy as np
import torch

import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood
import gpr_lib.Utils.Parameters_covariance_functions as cov_func
import model_learning.Model_learning as ML
import policy_learning.Cost_function as Cost_function
import policy_learning.MC_PILCO as MC_PILCO
import policy_learning.Policy as Policy

from policy_learning.Policy_KAN import KAN_Policy_with_angles

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser("test cartpole KAN")
p.add_argument("-seed", type=int, default=1, help="Random seed")
p.add_argument("-hidden", type=int, nargs='+', default=[22],
               help="KAN hidden layer sizes (e.g. 22 or 10 10 for 2 hidden)")
p.add_argument("-grid", type=int, default=5,
               help="B-spline grid size (default 5)")
p.add_argument("-order", type=int, default=3,
               help="B-spline order (default 3 = cubic)")
p.add_argument("-num_trials", type=int, default=5)
p.add_argument("-results_root", type=str, default="results_cartpole_kan",
               help="Root directory for results.")
args = p.parse_args()

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)

dtype = torch.float64
device = torch.device("cpu")
torch.set_num_threads(1)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
print("---- Set environment parameters ----")
num_trials = args.num_trials
T_sampling = 0.05
T_exploration = 3.0
T_control = 3.0
state_dim = 4
input_dim = 1
num_gp = int(state_dim / 2)
gp_input_dim = 6
import simulation_class.ode_systems as f_ode
ode_fun = f_ode.cartpole
u_max = 10.0
std_noise = 1e-2
std_list = std_noise * np.ones(state_dim)

# ---------------------------------------------------------------------------
# GP model learning (identical to baseline)
# ---------------------------------------------------------------------------
print("\n---- Set model learning parameters ----")
f_model_learning = ML.Speed_Model_learning_RBF_MPK_angle_state
model_learning_par = dict(
    num_gp=num_gp,
    angle_indeces=[2],
    not_angle_indeces=[0, 1, 3],
    T_sampling=T_sampling,
    vel_indeces=[1, 3],
    not_vel_indeces=[0, 2],
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
# Exploration policy (identical to baseline)
# ---------------------------------------------------------------------------
print("\n---- Set exploration policy parameters ----")
f_rand_exploration_policy = Policy.Random_exploration
rand_exploration_policy_par = dict(
    state_dim=state_dim, input_dim=input_dim,
    u_max=u_max, dtype=dtype, device=device,
)

# ---------------------------------------------------------------------------
# Control policy: KAN (replaces Sum_of_gaussians_with_angles)
# ---------------------------------------------------------------------------
print("\n---- Set KAN control policy parameters ----")
f_control_policy = KAN_Policy_with_angles
control_policy_par = dict(
    state_dim=state_dim,
    input_dim=input_dim,
    hidden_sizes=args.hidden,
    angle_indices=np.array([2]),
    non_angle_indices=np.array([0, 1, 3]),
    grid_size=args.grid,
    spline_order=args.order,
    grid_range=(-2.0, 2.0),
    flg_squash=True,
    u_max=u_max,
    flg_drop=True,
    dtype=dtype,
    device=device,
)

# For KAN, reinit just re-randomises weights; these keys are accepted
# but ignored (interface compat with Sum_of_gaussians.reinit).
policy_reinit_dict = dict(
    lenghtscales_par=np.ones(state_dim + 1),
    centers_par=np.array([np.pi, np.pi, np.pi, 1.0, 1.0]),
    weight_par=u_max,
)

# ---------------------------------------------------------------------------
# Cost function (identical to baseline)
# ---------------------------------------------------------------------------
print("\n---- Set cost function ----")
f_cost_function = Cost_function.Cart_pole_cost
cost_function_par = dict(
    pos_index=0,
    angle_index=2,
    target_state=torch.tensor([np.pi, 0.0], dtype=dtype, device=device),
    lengthscales=torch.tensor([3.0, 1.0], dtype=dtype, device=device),
)

# ---------------------------------------------------------------------------
# MC-PILCO init
# ---------------------------------------------------------------------------
print("\n---- Init policy learning object ----")
import pathlib
log_dir = pathlib.Path(args.results_root) / str(seed)
log_dir.mkdir(parents=True, exist_ok=True)

MC_PILCO_init_dict = dict(
    T_sampling=T_sampling,
    state_dim=state_dim, input_dim=input_dim,
    f_sim=ode_fun,
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
PL_obj = MC_PILCO.MC_PILCO(**MC_PILCO_init_dict)

# ---------------------------------------------------------------------------
# MC-PILCO options (identical to baseline)
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
    num_particles=400,
    opt_steps_list=[2000] + [4000] * (num_trials - 1),
    lr_list=[0.01] * num_trials,
    f_optimizer="lambda p, lr : torch.optim.Adam(p, lr)",
    num_step_print=100,
    p_dropout_list=[0.25] * num_trials,
    p_drop_reduction=0.25 / 2,
    alpha_diff_cost=0.99,
    min_diff_cost=0.08,
    num_min_diff_cost=200,
    min_step=200,
    lr_min=0.0025,
    policy_reinit_dict=policy_reinit_dict,
)

reinforce_param_dict = dict(
    initial_state=np.array([0.0, 0.0, 0.0, 0.0]),
    initial_state_var=np.array([0.0001, 0.0001, 0.0001, 0.0001]),
    T_exploration=T_exploration,
    T_control=T_control,
    num_trials=num_trials,
    model_optimization_opt_list=model_optimization_opt_list,
    policy_optimization_dict=policy_optimization_dict,
)

# ---------------------------------------------------------------------------
# Save config
# ---------------------------------------------------------------------------
print("\n---- Save test configuration ----")
config_log_dict = dict(
    MC_PILCO_init_dict=MC_PILCO_init_dict,
    reinforce_param_dict=reinforce_param_dict,
    cli_args=vars(args),
)
pkl.dump(config_log_dict, open(str(log_dir / "config_log.pkl"), "wb"))

# ---------------------------------------------------------------------------
# Print comparison info
# ---------------------------------------------------------------------------
n_kan = sum(pp.numel() for pp in PL_obj.control_policy.parameters()
            if pp.requires_grad)
print(f"\n{'='*60}")
print(f"CARTPOLE KAN EXPERIMENT")
print(f"{'='*60}")
print(f"KAN architecture : [5] + {args.hidden} + [1]")
print(f"Grid size        : {args.grid}")
print(f"Spline order     : {args.order}")
print(f"KAN params       : {n_kan}")
print(f"RBF baseline     : ~1205 params (200 basis)")
print(f"Param ratio      : {n_kan / 1205:.2f}x")
print(f"Results dir      : {log_dir.resolve()}")
print(f"{'='*60}")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
PL_obj.reinforce(**reinforce_param_dict)
print(f"\n[cartpole_kan] Done. Log saved to {log_dir.resolve() / 'log.pkl'}")
