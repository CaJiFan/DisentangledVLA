import math
import numpy as np
import torch


def compute_mig(z_means: torch.Tensor, labels: torch.Tensor, num_bins: int = 20) -> float:
    """
    Mutual Information Gap (MIG) disentanglement metric (Chen et al., NeurIPS 2018).

    Measures how exclusively a single latent dimension captures each generative factor.
    A score of 1 means one latent perfectly and uniquely represents the factor.
    A score of 0 means no latent is better than any other at capturing the factor.

    Args:
        z_means:  (N, latent_dim) encoded posterior means (use mu, not z samples)
        labels:   (N,) integer factor labels, e.g. task ID (0-indexed, contiguous or not)
        num_bins: number of bins used to discretise each continuous z dimension

    Returns:
        MIG score as a float in [0, 1], higher is better.
    """
    z = z_means.detach().cpu().float().numpy()
    v = labels.detach().cpu().long().numpy()
    N, D = z.shape

    unique_factors = np.unique(v)
    K = len(unique_factors)

    if K < 2:
        return 0.0  # Only one class — MI is trivially 0

    # Re-map labels to a contiguous 0..K-1 range
    remap = {old: new for new, old in enumerate(unique_factors)}
    v = np.array([remap[vi] for vi in v], dtype=np.int64)

    # H(v) — marginal entropy of the factor (in bits)
    p_v = np.bincount(v, minlength=K).astype(np.float64) / N
    H_v = -np.sum(p_v * np.log2(p_v + 1e-10))

    if H_v < 1e-8:
        return 0.0

    # For each latent dimension d, estimate I(z_d; v) via histogram discretisation.
    # I(z_d; v) = H(z_d) - H(z_d | v)
    #           = sum_{z,v} p(z,v) * log2[ p(z,v) / (p(z) * p(v)) ]
    mi = np.zeros(D, dtype=np.float64)
    for d in range(D):
        z_d = z[:, d]
        edges = np.linspace(z_d.min() - 1e-6, z_d.max() + 1e-6, num_bins + 1)
        z_bins = np.clip(np.digitize(z_d, edges) - 1, 0, num_bins - 1)

        # Build joint distribution p(z_bin, v)
        joint = np.zeros((num_bins, K), dtype=np.float64)
        np.add.at(joint, (z_bins, v), 1.0 / N)

        # Marginals
        p_zd = joint.sum(axis=1, keepdims=True)  # (num_bins, 1)

        # MI via log-ratio — clamp to avoid log(0)
        ratio = joint / (p_zd * p_v[np.newaxis, :] + 1e-10)
        mi[d] = np.sum(joint * np.log2(ratio + 1e-10))

    mi = np.maximum(mi, 0.0)  # clamp numerical negatives to 0

    # MIG = (top-1 MI — top-2 MI) / H(v)
    # This is the single-factor version; with multiple factors you'd average over them.
    sorted_mi = np.sort(mi)[::-1]
    mig = (sorted_mi[0] - sorted_mi[1]) / H_v if D > 1 else 0.0

    return float(np.clip(mig, 0.0, 1.0))


def cyclic_beta_schedule(
    step: int,
    max_steps: int,
    beta_max: float,
    n_cycles: int = 4,
    warmup_ratio: float = 0.5,
) -> float:
    """
    Cyclic cosine beta annealing schedule (Li et al., "Don't Blame the ELBO", 2019).

    Divides training into `n_cycles` equal periods. Within each period:
      - First `warmup_ratio` fraction: beta ramps 0 → beta_max via a cosine curve.
      - Remaining fraction:            beta stays at beta_max.

    This lets the model re-learn good reconstruction at the start of every cycle,
    then be pushed toward disentanglement in the plateau phase, repeatedly.

    Args:
        step:          current training step (0-indexed)
        max_steps:     total number of training steps
        beta_max:      peak beta value (your --beta argument)
        n_cycles:      number of complete cycles over max_steps (default 4)
        warmup_ratio:  fraction of each cycle used for ramping (default 0.5)

    Returns:
        current beta value (float)
    """
    if n_cycles <= 0:
        # Fallback: original linear single-shot warmup
        warmup = max(1, int(max_steps * warmup_ratio))
        return beta_max * min(1.0, step / warmup)

    period = max_steps / n_cycles
    cycle_pos = step % period
    ramp_end = warmup_ratio * period

    if cycle_pos < ramp_end:
        return beta_max * 0.5 * (1.0 - math.cos(math.pi * cycle_pos / ramp_end))
    else:
        return beta_max
