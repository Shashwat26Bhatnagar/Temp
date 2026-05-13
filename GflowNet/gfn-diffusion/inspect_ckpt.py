import torch
ckpt = torch.load('cartpole_denoising_theta_final.pt', map_location='cpu', weights_only=False)
print("Keys in checkpoint:", ckpt.keys())
print("loss:", ckpt.get('loss', 'not saved'))
print("step:", ckpt.get('global_step', 'not saved'))
# Check if back_model weights are non-trivial
sd = ckpt['model_state_dict']
back_keys = [k for k in sd.keys() if 'back' in k]
print("back_model keys:", back_keys[:3])
if back_keys:
    print("back_model first weight mean:", sd[back_keys[0]].mean().item())
    print("back_model first weight std: ", sd[back_keys[0]].std().item())
