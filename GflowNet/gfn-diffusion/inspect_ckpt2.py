import torch
ckpt = torch.load('cartpole_denoising_theta_final.pt', map_location='cpu', weights_only=False)
args = ckpt['args']
print("learn_pb:", args.get('learn_pb'))
print("epochs:", args.get('epochs'))
print("mode_fwd:", args.get('mode_fwd'))
print("batch_size:", args.get('batch_size'))
print("lr_policy:", args.get('lr_policy'))
