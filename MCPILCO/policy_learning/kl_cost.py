import torch


def gaussian_moments_from_particles(states_sequence):
    """
    Compute per-timestep mean and diagonal covariance of particles.

    Args:
        states_sequence: [num_instants, num_particles, state_dim]

    Returns:
        mu:        [num_instants, state_dim]
        Sigma_diag:[num_instants, state_dim]  (diagonal of covariance)
    """
    mu = states_sequence.mean(dim=1)                    # [N+1, 4]
    Sigma_diag = states_sequence.var(dim=1)             # [N+1, 4]
    # Add tiny floor for numerical stability (avoid log(0) and div by 0)
    Sigma_diag = Sigma_diag + 1e-8
    return mu, Sigma_diag


def reverse_kl_gaussian_diag(mu_p, Sigma_p_diag, mu_q, Sigma_q_diag):
    """
    Closed-form KL(p || q) for two Gaussians with diagonal covariances.

    KL(p||q) = 0.5 * [ tr(Sigma_q^-1 Sigma_p)
                       + (mu_q - mu_p)^T Sigma_q^-1 (mu_q - mu_p)
                       - d
                       + log(|Sigma_q| / |Sigma_p|) ]

    Args:
        mu_p:         [state_dim]         — prior mean (fixed)
        Sigma_p_diag: [state_dim]         — prior variance diagonal
        mu_q:         [..., state_dim]    — GP mean  (batched over time)
        Sigma_q_diag: [..., state_dim]    — GP variance diagonal

    Returns:
        kl: [...]   — KL per batch element
    """
    d = mu_p.shape[-1]

    # tr(Sigma_q^-1 Sigma_p)  for diagonal = sum(Sigma_p / Sigma_q)
    trace_term = (Sigma_p_diag / Sigma_q_diag).sum(dim=-1)

    # Mahalanobis term
    diff = mu_q - mu_p
    maha_term = ((diff ** 2) / Sigma_q_diag).sum(dim=-1)

    # log|Sigma_q| - log|Sigma_p|  for diagonal
    logdet_q = torch.log(Sigma_q_diag).sum(dim=-1)
    logdet_p = torch.log(Sigma_p_diag).sum(dim=-1)
    logdet_term = logdet_q - logdet_p

    kl = 0.5 * (trace_term + maha_term - d + logdet_term)
    return kl


def forward_kl_gaussian_diag(mu_q, Sigma_q_diag, mu_p, Sigma_p_diag):
    """
    Closed-form KL(q || p) for two Gaussians with diagonal covariances.

    This is the divergence we MINIMIZE in Variant B:
        q = particle-induced distribution from GP rollout (fitted Gaussian)
        p = GFN target distribution (fixed, known parameters)

    KL(q||p) = 0.5 * [ tr(Sigma_p^-1 Sigma_q)
                       + (mu_q - mu_p)^T Sigma_p^-1 (mu_q - mu_p)
                       - d
                       + log(|Sigma_p| / |Sigma_q|) ]

    For diagonal covariances:
        = 0.5 * sum_d [ sigma_q^2/sigma_p^2
                       + (mu_q - mu_p)^2 / sigma_p^2
                       - 1
                       + log(sigma_p^2 / sigma_q^2) ]

    Properties:
        - KL(q||p) >= 0,  = 0 iff q == p
        - Mode-seeking: gradient pushes q's mean toward p's mean and
          q's variance toward p's variance.
        - Numerically more stable than KL(p||q) when q can be very tight
          (because the prior variance Sigma_p is in the denominator,
          not Sigma_q which depends on particles).

    Args:
        mu_q:         [..., state_dim]    — particle mean   (batched over time)
        Sigma_q_diag: [..., state_dim]    — particle variance diagonal
        mu_p:         [state_dim]         — target mean (fixed)
        Sigma_p_diag: [state_dim]         — target variance diagonal

    Returns:
        kl: [...]   — KL per batch element
    """
    d = mu_p.shape[-1]

    # tr(Sigma_p^-1 Sigma_q)  for diagonal = sum(Sigma_q / Sigma_p)
    trace_term = (Sigma_q_diag / Sigma_p_diag).sum(dim=-1)

    # Mahalanobis term: (mu_q - mu_p)^T Sigma_p^-1 (mu_q - mu_p)
    diff = mu_q - mu_p
    maha_term = ((diff ** 2) / Sigma_p_diag).sum(dim=-1)

    # log|Sigma_p| - log|Sigma_q|  for diagonal
    logdet_p = torch.log(Sigma_p_diag).sum(dim=-1)
    logdet_q = torch.log(Sigma_q_diag).sum(dim=-1)
    logdet_term = logdet_p - logdet_q

    kl = 0.5 * (trace_term + maha_term - d + logdet_term)
    return kl


def time_weighted_kl_sum(states_sequence, gfn_prior, weighting='quadratic'):
    """
    Sum the reverse KL across the rollout horizon with optional
    time weighting that emphasizes hitting the target at the end.

    Args:
        states_sequence: [num_instants, num_particles, state_dim]
        gfn_prior: GFNPrior instance
        weighting: 'none' | 'linear' | 'quadratic'

    Returns:
        kl_per_step: [num_instants]  — for logging
        kl_total:    scalar          — weighted sum, this goes in the loss
    """
    N_h_plus_1 = states_sequence.shape[0]
    N_h = N_h_plus_1 - 1
    device = states_sequence.device
    dtype = states_sequence.dtype

    mu_q, Sigma_q_diag = gaussian_moments_from_particles(states_sequence)

    # For MVP the prior is constant across time, but we query per-step
    # to keep the interface compatible with future time-conditioning.
    kl_steps = []
    for k in range(N_h_plus_1):
        mu_p, Sigma_p = gfn_prior.get_target_at_step(k, N_h)
        Sigma_p_diag = torch.diagonal(Sigma_p)
        kl_k = reverse_kl_gaussian_diag(
            mu_p, Sigma_p_diag, mu_q[k], Sigma_q_diag[k]
        )
        kl_steps.append(kl_k)
    kl_per_step = torch.stack(kl_steps)   # [N_h+1]

    # Time weights
    t = torch.arange(N_h_plus_1, dtype=dtype, device=device) / N_h
    if weighting == 'none':
        weights = torch.ones_like(t)
    elif weighting == 'linear':
        weights = t
    elif weighting == 'quadratic':
        weights = t ** 2
    else:
        raise ValueError(f"Unknown weighting: {weighting}")

    kl_total = (weights * kl_per_step).sum()
    return kl_per_step, kl_total
