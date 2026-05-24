"""
KAN (Kolmogorov-Arnold Network) policy for MC-PILCO.

Drop-in replacement for Sum_of_gaussians / Sum_of_gaussians_with_angles.
Uses learnable B-spline activation functions on edges instead of fixed
Gaussian RBF basis functions.

KAN key idea:  instead of fixed activations on nodes (like ReLU in MLP),
KAN places learnable univariate functions on EDGES. Each edge (i->j)
computes:
    phi_{ij}(x_i) = w_base_{ij} * silu(x_i) + sum_k c_{ijk} * B_k(x_i)

where B_k are B-spline basis functions on a uniform grid and c_{ijk}
are learnable coefficients.

Parameter count for a [d_in, H, d_out] KAN:
    edges = d_in * H + H * d_out
    params_per_edge = (grid_size + spline_order) + 1   [spline coeffs + base weight]
    total = edges * params_per_edge

Example: [5, 22, 1] with grid_size=5, spline_order=3 ->
    edges = 5*22 + 22*1 = 132
    params_per_edge = 8 + 1 = 9
    total = 132 * 9 = 1188  (comparable to 200-RBF with 1205 params)

Reference:
    Liu et al., "KAN: Kolmogorov-Arnold Networks", arXiv:2404.19756
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from policy_learning.Policy import Policy


# ---------------------------------------------------------------------------
# KAN building block: a single layer with B-spline edges
# ---------------------------------------------------------------------------
class KANLinear(nn.Module):
    """
    Single KAN layer: in_features -> out_features.

    Each of the (in_features * out_features) edges carries:
      * a SiLU-gated linear base path   w_base * silu(x)
      * a B-spline nonlinear path       sum_k c_k * B_k(x)
    """

    def __init__(self, in_features, out_features,
                 grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0,
                 grid_range=(-2.0, 2.0),
                 dtype=torch.float64, device=torch.device('cpu')):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Uniform B-spline knot vector, extended by spline_order on each side
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1,
                             dtype=dtype, device=device) * h
                + grid_range[0])
        # Expand for each input dimension: [in_features, n_knots]
        self.register_buffer(
            'grid', grid.unsqueeze(0).expand(in_features, -1).contiguous())

        n_bases = grid_size + spline_order  # number of B-spline basis fns

        # Learnable B-spline coefficients: [out, in, n_bases]
        self.spline_weight = nn.Parameter(
            scale_noise * torch.randn(
                out_features, in_features, n_bases,
                dtype=dtype, device=device))

        # Residual base weight (linear through SiLU): [out, in]
        self.base_weight = nn.Parameter(
            (scale_base / math.sqrt(in_features))
            * torch.randn(out_features, in_features,
                           dtype=dtype, device=device))

    @property
    def n_params(self):
        return self.spline_weight.numel() + self.base_weight.numel()

    def b_splines(self, x):
        """
        Evaluate B-spline basis functions via de Boor recursion.

        Args:
            x: [batch, in_features]
        Returns:
            bases: [batch, in_features, n_bases]
        """
        x = x.unsqueeze(-1)           # [batch, in, 1]
        grid = self.grid               # [in, n_knots]

        # Order-0: piecewise constant on each knot interval
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        # Recursive de Boor for orders 1..spline_order
        for k in range(1, self.spline_order + 1):
            left_num = x - grid[:, :-(k + 1)]
            left_den = (grid[:, k:-1] - grid[:, :-(k + 1)]).clamp(min=1e-10)
            right_num = grid[:, k + 1:] - x
            right_den = (grid[:, k + 1:] - grid[:, 1:(-k)]).clamp(min=1e-10)

            bases = ((left_num / left_den) * bases[:, :, :-1]
                     + (right_num / right_den) * bases[:, :, 1:])

        return bases  # [batch, in_features, n_bases]

    def forward(self, x):
        """
        Args:
            x: [batch, in_features]
        Returns:
            y: [batch, out_features]
        """
        # Residual path: silu(x) @ base_weight^T
        base_out = F.linear(F.silu(x), self.base_weight)

        # Spline path: sum_i sum_k  c_{jik} * B_k(x_i)
        bases = self.b_splines(x)        # [batch, in, n_bases]
        spline_out = torch.einsum(
            'bik,jik->bj', bases, self.spline_weight)

        return base_out + spline_out


# ---------------------------------------------------------------------------
# KAN Policy (direct state input)
# ---------------------------------------------------------------------------
class KAN_Policy(Policy):
    """
    KAN-based control policy for MC-PILCO.

    Drop-in replacement for Sum_of_gaussians.

    Args:
        state_dim:     raw state dimension
        input_dim:     action dimension
        hidden_sizes:  list of hidden layer widths, e.g. [22] for 1 hidden
        grid_size:     number of B-spline grid intervals (default 5)
        spline_order:  B-spline order (default 3 = cubic)
        grid_range:    (lo, hi) for the spline knot grid
        flg_squash:    squash output to [-u_max, u_max] via tanh
        u_max:         action bound
        flg_drop:      enable dropout between hidden layers
    """

    def __init__(self, state_dim, input_dim, hidden_sizes,
                 grid_size=5, spline_order=3, grid_range=(-2.0, 2.0),
                 flg_squash=False, u_max=1.0,
                 flg_drop=True,
                 dtype=torch.float64, device=torch.device('cpu')):
        super().__init__(
            state_dim=state_dim, input_dim=input_dim,
            flg_squash=flg_squash, u_max=u_max,
            dtype=dtype, device=device)

        self.hidden_sizes = list(hidden_sizes)
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Build KAN layers
        dims = [state_dim] + self.hidden_sizes + [input_dim]
        self.kan_layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.kan_layers.append(KANLinear(
                dims[i], dims[i + 1],
                grid_size=grid_size, spline_order=spline_order,
                grid_range=grid_range,
                dtype=dtype, device=device,
            ))

        # Dropout
        if flg_drop:
            self.f_drop = lambda x, p: F.dropout(
                x, p=p, training=self.training)
        else:
            self.f_drop = lambda x, p: x

        # Report
        n_total = sum(p.numel() for p in self.parameters()
                      if p.requires_grad)
        print(f"[KAN_Policy] arch={dims} grid={grid_size} "
              f"order={spline_order} | {n_total} trainable params")

    def reinit(self, **kwargs):
        """Re-initialize KAN parameters (called by MC-PILCO on NaN cost).

        Accepts and ignores RBF-specific kwargs (lenghtscales_par,
        centers_par, weight_par) for interface compatibility.
        """
        for layer in self.kan_layers:
            nn.init.normal_(layer.spline_weight, 0.0, 0.1)
            nn.init.normal_(
                layer.base_weight, 0.0,
                1.0 / math.sqrt(layer.in_features))
        print("[KAN_Policy] reinit: all spline/base weights randomised.")

    def forward(self, states, t=None, p_dropout=0.0):
        x = states.reshape(-1, self.state_dim)
        for layer in self.kan_layers[:-1]:
            x = layer(x)
            x = self.f_drop(x, p_dropout)
        x = self.kan_layers[-1](x)       # no dropout on output layer
        return self.f_squash(x.reshape(-1, self.input_dim))


# ---------------------------------------------------------------------------
# KAN Policy with angle wrapping (cos/sin)
# ---------------------------------------------------------------------------
class KAN_Policy_with_angles(KAN_Policy):
    """
    KAN policy that maps angle indices through cos/sin before the network.

    Drop-in replacement for Sum_of_gaussians_with_angles.

    For cartpole: state = [p, p_dot, theta, theta_dot]
        angle_indices    = [2]        -> theta
        non_angle_indices = [0, 1, 3] -> p, p_dot, theta_dot

    Augmented input to KAN: [p, p_dot, theta_dot, cos(theta), sin(theta)]
    -> dimension = 3 + 2*1 = 5
    """

    def __init__(self, state_dim, input_dim, hidden_sizes,
                 angle_indices, non_angle_indices,
                 grid_size=5, spline_order=3, grid_range=(-2.0, 2.0),
                 flg_squash=False, u_max=1.0,
                 flg_drop=True,
                 dtype=torch.float64, device=torch.device('cpu')):
        self.angle_indices = np.asarray(angle_indices)
        self.non_angle_indices = np.asarray(non_angle_indices)
        self.num_angle_indices = self.angle_indices.size

        # Augmented dim: non-angle dims + 2 * angle dims (cos + sin)
        augmented_dim = self.non_angle_indices.size + 2 * self.num_angle_indices

        super().__init__(
            state_dim=augmented_dim,
            input_dim=input_dim,
            hidden_sizes=hidden_sizes,
            grid_size=grid_size, spline_order=spline_order,
            grid_range=grid_range,
            flg_squash=flg_squash, u_max=u_max,
            flg_drop=flg_drop,
            dtype=dtype, device=device)

        # Store the raw state dim for the angle transformation
        self._raw_state_dim = state_dim

    def forward(self, states, t=None, p_dropout=0.0):
        states = states.reshape(-1, self._raw_state_dim)
        # Augment: [non-angle features, cos(angles), sin(angles)]
        augmented = torch.cat([
            states[:, self.non_angle_indices],
            torch.cos(states[:, self.angle_indices]),
            torch.sin(states[:, self.angle_indices]),
        ], dim=1)
        return super().forward(augmented, t=t, p_dropout=p_dropout)
