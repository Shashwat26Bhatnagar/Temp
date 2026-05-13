import torch
from models.gfn import GFN
from energies.cartpole import CartPoleEnergy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reload
checkpoint = torch.load('cartpole_denoising_theta_final.pt', map_location=device)
energy = CartPoleEnergy(device=device)

gfn_model = GFN(
    dim=4,
    s_emb_dim=64,
    hidden_dim=128,
    harmonics_dim=64,
    t_dim=64,
    log_var_range=4.0,
    t_scale=1.0,
    learned_variance=True,
    partial_energy=False,
    clipping=True,
    lgv_clip=1e2,
    gfn_clip=1e4,
    pb_scale_range=0.5,
    learn_pb=True,
    device=device
).to(device)

gfn_model.load_state_dict(checkpoint['model_state_dict'])
gfn_model.eval()

# Sample from learned denoising process_theta
samples = gfn_model.sample(2000, None, energy.log_reward)
# samples shape: [2000, 4]
# These are x_1 terminal states drawn from denoising_theta
# [pos, vel, angle, ang_vel] — concentrated near [0,0,0,0]

print("Sample mean:", samples.mean(dim=0))
print("Sample std: ", samples.std(dim=0))

# denoising_theta network itself (back_model = s_theta MLP):
denoising_theta = gfn_model.back_model
print("denoising_theta architecture:", denoising_theta)

torch.save(samples, 'cartpole_samples.pt')
print("Saved 2000 samples to cartpole_samples.pt")
