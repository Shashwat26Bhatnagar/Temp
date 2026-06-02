# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Test MC-PILCO on cartpole -- Variant K (per-particle KL vs GFN's
TIME-VARYING diffusion marginals).

Unlike Variant J (which compares every step against the GFN's frozen
terminal target), Variant K maps each control step to the corresponding
GFN diffusion step and compares against the GFN's distribution AT THAT
diffusion step. The GFN's intermediate marginals are wide early and tight
late -- a natural curriculum that avoids Variant J's variance blow-up.

Requires MC_PILCO_GPStats (captures per-particle GP mean/variance).
Everything else matches Variant C/J for a clean comparison.
"""

import argparse
import os
import pickle as pkl

import numpy as np
import torch

import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood
import gpr_lib.Utils.Parameters_covariance_functions as cov_func
import model_learning.Model_learning as ML
import policy_learning.Cost_function as Cost_function
import policy_learning.Policy as Policy
import simulation_class.ode_systems as f_ode

p = argparse.ArgumentParser("test cartpole variant K (time-varying GFN diffusion KL)")
p.add_argument("-seed", type=int, default=1)
p.add_argument("-checkpoint", type=str, default=None)
p.add_argument("-num_trials", type=int, default=5)
p.add_argument("-bptt", type=int, default=None,
               help="BPTT truncation length (steps). None = full BPTT. "
                    "1 = per-step LOCAL update: each step's KL updates the "
                    "policy only through that step's action (no backprop "
                    "through the GP rollout chain).")
locals().update(vars(p.parse_known_args()[0]))

import pathlib
if checkpoint is None:
    _repo_root = pathlib.Path(__file__).resolve().parent.parent
    checkpoint = str(_repo_root / 'GflowNet' / 'gfn-diffusion' / 'energy_sampling'
                     / 'cartpole_denoising_theta_final.pt')

torch.manual_seed(seed)
np.random.seed(seed)

dtype = torch.float64
device = torch.device("cpu")
torch.set_num_threads(1)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
print("---- Set environment parameters ----")
T_sampling = 0.05
T_exploration = 3.0
T_control = 3.0
state_dim = 4
input_dim = 1
num_gp = int(state_dim / 2)
gp_input_dim = 6
ode_fun = f_ode.cartpole
u_max = 10.0
std_noise = 1e-2
std_list = std_noise * np.ones(state_dim)

# ---------------------------------------------------------------------------
# Model learning (identical to Variant C/J)
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
# Policy (identical to Variant C/J)
# ---------------------------------------------------------------------------
print("\n---- Set exploration policy parameters ----")
f_rand_exploration_policy = Policy.Random_exploration
rand_exploration_policy_par = dict(
    state_dim=state_dim, input_dim=input_dim,
    u_max=u_max, dtype=dtype, device=device,
)

print("\n---- Set control policy parameters ----")
num_basis = 200
f_control_policy = Policy.Sum_of_gaussians_with_angles
control_policy_par = dict(
    state_dim=state_dim, input_dim=input_dim,
    angle_indices=np.array([2]),
    non_angle_indices=np.array([0, 1, 3]),
    u_max=u_max, num_basis=num_basis,
    dtype=dtype, device=device,
)
angle_centers = np.pi * 2 * (np.random.rand(num_basis, 1) - 0.5)
cos_centers = np.cos(angle_centers)
sin_centers = np.sin(angle_centers)
not_angle_centers = np.pi * 2 * (np.random.rand(num_basis, 3) - 0.5)
control_policy_par["centers_init"] = np.concatenate(
    [not_angle_centers, cos_centers, sin_centers], 1)
control_policy_par["lengthscales_init"] = 1 * np.ones(state_dim + 1)
control_policy_par["weight_init"] = u_max * (np.random.rand(input_dim, num_basis) - 0.5)
control_policy_par["flg_squash"] = True
control_policy_par["flg_drop"] = True
policy_reinit_dict = dict(
    lenghtscales_par=control_policy_par["lengthscales_init"],
    centers_par=np.array([np.pi, np.pi, np.pi, 1.0, 1.0]),
    weight_par=u_max,
)

# ---------------------------------------------------------------------------
# Cost (Variant K: time-varying GFN diffusion marginals)
# ---------------------------------------------------------------------------
print("\n---- Set cost function (Variant K: time-varying GFN diffusion KL) ----")
from policy_learning.variant_k_cost import VariantK_Cost
f_cost_function = VariantK_Cost
cost_function_par = dict(
    checkpoint_path=checkpoint,
    alpha=5.0,
    epsilon=0.10,
    weighting='uniform',          # time-varying target IS the curriculum
    position_bound=2.4,
    angle_bound=0.35,
    dtype=dtype,
    device=device,
)

# ---------------------------------------------------------------------------
# Build MC_PILCO_GPStats
# ---------------------------------------------------------------------------
print("\n---- Init policy learning object (MC_PILCO_GPStats) ----")
results_dir = f"results_variant_k/{seed}"
os.makedirs(results_dir, exist_ok=True)

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
    log_path=results_dir,
    dtype=dtype, device=device,
)

from policy_learning.mc_pilco_gp_stats import MC_PILCO_GPStats
PL_obj = MC_PILCO_GPStats(bptt_truncate=bptt, **MC_PILCO_init_dict)
if bptt is not None:
    print(f"[variant_k] PER-STEP LOCAL updates active (bptt_truncate={bptt}): "
          f"each step's KL updates the policy only through that step's action.")
else:
    print(f"[variant_k] FULL BPTT active (gradient flows backward through the "
          f"whole trajectory). Pass -bptt 1 for per-step local updates.")

# ---------------------------------------------------------------------------
# MC-PILCO options (identical to Variant C/J)
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

print("\n---- Save test configuration ----")
config_log_dict = dict(
    MC_PILCO_init_dict=MC_PILCO_init_dict,
    reinforce_param_dict=reinforce_param_dict,
)
pkl.dump(config_log_dict, open(results_dir + "/config_log.pkl", "wb"))

print("\n---- Start learning (Variant K) ----")
PL_obj.reinforce(**reinforce_param_dict)
