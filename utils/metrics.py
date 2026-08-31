import math
import numpy as np
import torch
import torch.nn.functional as F


def compute_supcon_loss(
    z: torch.Tensor,
    task_ids: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised Contrastive Loss (SupCon, Khosla et al., NeurIPS 2020).

    Pulls z's from the same task together and pushes z's from different
    tasks apart, using all within-batch (anchor, positive) pairs.

    Critical for forcing task-discriminative structure into z when the
    decoder's text conditioning would otherwise let z collapse to noise.

    Args:
        z:          (B, Z) latent codes (z samples or mu — either works)
        task_ids:   (B,)   integer task labels (0-indexed)
        temperature: controls sharpness of the similarity distribution

    Returns:
        Scalar loss tensor. Returns 0.0 if no anchor has a positive pair.
    """
    B = z.shape[0]
    z_norm = F.normalize(z, dim=1)                                  # (B, Z)

    # Pairwise cosine similarity matrix, scaled by temperature
    sim = torch.matmul(z_norm, z_norm.T) / temperature              # (B, B)

    # Numerical stability: shift by row-max (doesn't change softmax)
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    # Positive mask: same task, excluding self
    task_col = task_ids.view(-1, 1)                                  # (B, 1)
    pos_mask = (task_col == task_ids.view(1, -1)).float()            # (B, B)
    pos_mask.fill_diagonal_(0)

    # Denominator: sum over all non-self pairs
    exp_sim = torch.exp(sim)
    self_mask = torch.eye(B, device=z.device, dtype=torch.bool)
    exp_sim = exp_sim.masked_fill(self_mask, 0.0)
    log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)  # (B, 1)

    # Per-anchor loss: mean of -log(exp(sim_pos) / sum_all) over positives
    log_prob = sim - log_denom                                       # (B, B)
    n_pos = pos_mask.sum(dim=1)                                      # (B,)
    has_pos = n_pos > 0

    if not has_pos.any():
        return z.sum() * 0.0  # zero with gradient graph intact

    per_anchor = -(log_prob * pos_mask).sum(dim=1) / (n_pos + 1e-8)  # (B,)
    return per_anchor[has_pos].mean()


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


def get_beta_schedule(
    step: int,
    max_steps: int,
    beta_max: float,
    n_cycles: int = 0,
    warmup_ratio: float = 0.05,
    schedule_type: str = "warmup",
    beta_high: float = 1.0,
) -> float:
    """
    Beta annealing/scheduling strategies for VAEs/CVAEs.

    Supports 4 major paradigms from literature:
      1. 'fixed': Constant beta = beta_max throughout training.
      2. 'warmup': Low-to-high linear warmup from 0 -> beta_max over the first
                  `warmup_ratio` fraction of steps (e.g., first 5%), then fixed.
                  (Bowman et al. 2016 / Sønderby et al. 2016)
      3. 'high_to_low': Starts at `beta_high` early in training to force high-level
                        disentanglement/task separation in z, then decays down to
                        `beta_max` (or lower) over the first `warmup_ratio` fraction
                        so decoder can focus on fine-grained trajectory reconstruction.
                        (Burgess et al. 2018 "Understanding disentangling in beta-VAE")
      4. 'cyclic': Cyclical cosine annealing over `n_cycles` periods (Fu et al. 2019).
    """
    if schedule_type == "fixed":
        return float(beta_max)

    if schedule_type == "high_to_low":
        warmup_steps = max(1, int(max_steps * warmup_ratio))
        if step < warmup_steps:
            # Linear decay from beta_high down to beta_max
            progress = step / warmup_steps
            return float(beta_high - progress * (beta_high - beta_max))
        else:
            return float(beta_max)

    if schedule_type == "warmup" or n_cycles <= 0:
        warmup_steps = max(1, int(max_steps * warmup_ratio))
        if step < warmup_steps:
            return float(beta_max * (step / warmup_steps))
        else:
            return float(beta_max)

    # Cyclical schedule (Fu et al., NAACL 2019)
    period = max_steps / n_cycles
    cycle_pos = step % period
    ramp_end = warmup_ratio * period

    if cycle_pos < ramp_end:
        return float(beta_max * 0.5 * (1.0 - math.cos(math.pi * cycle_pos / ramp_end)))
    else:
        return float(beta_max)


def cyclic_beta_schedule(
    step: int,
    max_steps: int,
    beta_max: float,
    n_cycles: int = 4,
    warmup_ratio: float = 0.5,
) -> float:
    """Backward-compatible wrapper for cyclic_beta_schedule."""
    if n_cycles <= 0:
        return get_beta_schedule(step, max_steps, beta_max, n_cycles=0, warmup_ratio=0.05, schedule_type="warmup")
    return get_beta_schedule(step, max_steps, beta_max, n_cycles=n_cycles, warmup_ratio=warmup_ratio, schedule_type="cyclic")

