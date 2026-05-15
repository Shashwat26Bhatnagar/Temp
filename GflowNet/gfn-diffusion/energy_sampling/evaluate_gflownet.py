import torch
import numpy as np
import argparse
import pathlib
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.gfn import GFN
from energies.cartpole import CartPoleEnergy

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def build_gfn_model(device):
    gfn_model = GFN(
        dim=4, s_emb_dim=64, hidden_dim=64,
        harmonics_dim=64, t_dim=64, log_var_range=4.0,
        t_scale=5.0, learned_variance=True, partial_energy=False,
        clipping=True, lgv_clip=1e2, gfn_clip=1e4,
        pb_scale_range=0.1, learn_pb=True, device=device,
        langevin_scaling_per_dimension=False
    )
    return gfn_model

def compute_mmd(x, y, sigma=1.0):
    # x, y are torch tensors shape [N, 4]
    # Gaussian kernel: k(a,b) = exp(-||a-b||^2 / (2*sigma^2))
    # MMD^2 = E[k(x,x)] + E[k(y,y)] - 2*E[k(x,y)]
    xx = torch.cdist(x, x, p=2.0) ** 2
    yy = torch.cdist(y, y, p=2.0) ** 2
    xy = torch.cdist(x, y, p=2.0) ** 2
    
    k_xx = torch.exp(-xx / (2 * sigma ** 2)).mean()
    k_yy = torch.exp(-yy / (2 * sigma ** 2)).mean()
    k_xy = torch.exp(-xy / (2 * sigma ** 2)).mean()
    
    mmd_sq = k_xx + k_yy - 2 * k_xy
    return float(mmd_sq.item())

def sample_from_model(gfn_model, energy, n=1000, traj_length=100):
    init_state = torch.zeros(n, 4).to(gfn_model.device)
    
    states, _, _, _ = gfn_model.get_trajectory_fwd(
        init_state, None, energy.log_reward)
    return states

def plot_phase_portraits(terminal_states_np, output_dir):
    import math as _math
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.tight_layout(pad=4.0)

    # Target mean in MC-PILCO frame: upright is theta = pi.
    MU = [0.0, 0.0, _math.pi, 0.0]

    def add_energy_contour(ax, dim_i, dim_j, sigmas, lims):
        # sigmas: [sigma_i, sigma_j]
        # lims:   [[xmin,xmax],[ymin,ymax]]
        # Contour drawn relative to MU[dim_i], MU[dim_j].
        pi = np.linspace(lims[0][0], lims[0][1], 100)
        pj = np.linspace(lims[1][0], lims[1][1], 100)
        XX, YY = np.meshgrid(pi, pj)
        E = 0.5 * (((XX - MU[dim_i]) / sigmas[0]) ** 2
                 + ((YY - MU[dim_j]) / sigmas[1]) ** 2)
        ax.contour(XX, YY, E, levels=6, colors='red', alpha=0.6,
                   linewidths=0.8)

    col0 = terminal_states_np[:, 0]
    col1 = terminal_states_np[:, 1]
    col2 = terminal_states_np[:, 2]
    col3 = terminal_states_np[:, 3]

    pi_lo, pi_hi = _math.pi - 0.5, _math.pi + 0.5

    # Panel [0,0] — position vs angle:
    ax = axs[0, 0]
    hb = ax.hexbin(col0, col2, gridsize=40, cmap='Blues')
    add_energy_contour(ax, 0, 2, sigmas=[0.5, 0.1],
                       lims=[[-2, 2], [pi_lo, pi_hi]])
    fig.colorbar(hb, ax=ax, label='count')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Angle (rad)')
    ax.set_title('Position vs Angle (target theta=pi)')

    # Panel [0,1] — velocity vs angular velocity:
    ax = axs[0, 1]
    hb = ax.hexbin(col1, col3, gridsize=40, cmap='Blues')
    add_energy_contour(ax, 1, 3, sigmas=[0.5, 0.1],
                       lims=[[-2, 2], [-0.5, 0.5]])
    fig.colorbar(hb, ax=ax, label='count')
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Angular velocity (rad/s)')
    ax.set_title('Velocity vs Angular Velocity')

    # Panel [1,0] — position vs velocity:
    ax = axs[1, 0]
    hb = ax.hexbin(col0, col1, gridsize=40, cmap='Blues')
    add_energy_contour(ax, 0, 1, sigmas=[0.5, 0.5],
                       lims=[[-2, 2], [-2, 2]])
    fig.colorbar(hb, ax=ax, label='count')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Position vs Velocity')

    # Panel [1,1] — angle vs angular velocity:
    ax = axs[1, 1]
    hb = ax.hexbin(col2, col3, gridsize=40, cmap='Blues')
    add_energy_contour(ax, 2, 3, sigmas=[0.1, 0.1],
                       lims=[[pi_lo, pi_hi], [-0.5, 0.5]])
    fig.colorbar(hb, ax=ax, label='count')
    ax.set_xlabel('Angle (rad)')
    ax.set_ylabel('Angular velocity (rad/s)')
    ax.set_title('Angle vs Angular Velocity')

    out_path = output_dir / 'phase_portraits.png'
    fig.savefig(out_path, dpi=150)
    print(f"Saved -> phase_portraits.png")

def analyze_safety_and_timeseries(states_np, output_dir):
    # SECTION 1 — Safety violation percentages:
    # Upright in MC-PILCO frame is theta = pi, so safety is |theta - pi| < 12 deg.
    import math as _math
    pos_violations   = np.any(np.abs(states_np[:,:,0]) > 2.4, axis=1)
    angle_violations = np.any(np.abs(states_np[:,:,2] - _math.pi) > 0.2094, axis=1)
    any_violations   = pos_violations | angle_violations
    print("\n=== Safety Verification ===")
    print(f"Position violations (>2.4m)       : {pos_violations.mean()*100:.1f}%")
    print(f"Angle violations (|theta-pi|>12d) : {angle_violations.mean()*100:.1f}%")
    print(f"Any bound violated                : {any_violations.mean()*100:.1f}%")

    # SECTION 2 — Time-series variance plot:
    timesteps = np.linspace(0, 1, states_np.shape[1])
    pos_var   = states_np[:, :, 0].var(axis=0)                   # [T+1]
    angle_var = (states_np[:, :, 2] - _math.pi).var(axis=0)      # [T+1] about pi

    fig, axs = plt.subplots(2, 1, figsize=(12, 6))

    # Top subplot:
    axs[0].plot(timesteps, pos_var, color='steelblue', label='empirical')
    axs[0].axhline(y=0.25, color='red', linestyle='--', label='target variance (σ²=0.25)')
    axs[0].set_ylabel('Position variance')
    axs[0].set_xlabel('Normalised time t')
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    # Bottom subplot:
    axs[1].plot(timesteps, angle_var, color='darkorange', label='empirical')
    axs[1].axhline(y=0.01, color='red', linestyle='--', label='target variance (σ²=0.01)')
    axs[1].set_ylabel('Angle variance')
    axs[1].set_xlabel('Normalised time t')
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    fig.suptitle('State variance over trajectory time', fontsize=13)
    plt.tight_layout()
    out_path = output_dir / 'timeseries_variance.png'
    fig.savefig(out_path, dpi=150)
    print(f"Saved -> timeseries_variance.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--traj_length', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    energy = CartPoleEnergy(device=device)
    gfn_model = build_gfn_model(device).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        gfn_model.load_state_dict(ckpt['model_state_dict'])
    else:
        gfn_model.load_state_dict(ckpt)
    gfn_model.eval()
    
    print(f"Loaded checkpoint: {args.checkpoint}")
    if 'global_step' in ckpt: print(f"  step : {ckpt['global_step']}")
    if 'log_Z' in ckpt:       print(f"  log_Z: {ckpt['log_Z']:.4f}")

    # Sample
    with torch.no_grad():
        states = sample_from_model(gfn_model, energy, args.n_samples, args.traj_length)
    terminal_states = states[:, -1, :]   # [N, 4]

    # Reference samples from true target
    sigma = torch.tensor([0.5, 0.5, 0.1, 0.1])
    mu = energy.mu.detach().cpu()
    gt_samples = mu + torch.randn(args.n_samples, 4) * sigma

    # MMD
    mmd = compute_mmd(terminal_states.cpu(), gt_samples)

    # Print results
    print("\n=== Target Density Alignment ===")
    print(f"Sample mean : {terminal_states.mean(0).tolist()}")
    print(f"Sample std  : {terminal_states.std(0).tolist()}")
    print(f"Target mean : {mu.tolist()}")
    print(f"Target std  : [0.5, 0.5, 0.1, 0.1]")
    print(f"MMD score   : {mmd:.6f}  (lower is better, 0 = perfect)")

    output_dir = pathlib.Path(args.checkpoint).parent
    plot_phase_portraits(terminal_states.cpu().numpy(), output_dir)
    
    analyze_safety_and_timeseries(states.cpu().numpy(), output_dir)

    print("\n=== Evaluation Complete ===")
    print(f"All outputs saved to: {output_dir}")
