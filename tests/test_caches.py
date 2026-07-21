"""
Cache Sanity Checks
====================
Verifies normalization consistency across all VLA embedding caches and
the reference dataset_statistics.pt used during VAE training / unnormalization.

Checks performed per cache:
  1. Embedding dim matches expected (384 for octo-small, 4096 for openvla)
  2. train_actions range — must be in [-1.05, 1.05] (normalised to [-1,1] with Tanh)
  3. train_teacher_mu range — should be moderate (roughly ±5 for a well-trained VAE)
  4. train_teacher_lv range — should not be collapsed (std > 0.01) or exploded (< 50)
  5. Cross-check: re-encode one batch of stored actions through the VAE and compare
     to stored teacher_mu — if they match, normalization was correct at cache-build time
  6. Unnorm round-trip: unnorm(norm(raw_action)) ≈ raw_action

Run inside the DisentangledVLA container:
    python3 tests/test_caches.py
"""

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import numpy as np

from src.disentanglers import TCNTextActionBetaTCVAE, TCNTextActionCVAE

# ---------------------------------------------------------------------------
# Config — edit these to match your setup
# ---------------------------------------------------------------------------
STATS_PATH  = "./checkpoints/text_tcvae/libero_spatial/dataset_statistics.pt"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHUNK_SIZE  = 8
ACTION_DIM  = 7
N_BLOCKS    = max(3, (CHUNK_SIZE - 1).bit_length())
STEP        = 100000

CACHES = [
    dict(
        label      = "octo / TCN",
        path       = "./checkpoints/projectors/octo/libero_spatial/vla_emb_cache_text_cond_beta_tcvae_arch_tcn_beta0.001_z128.pt",
        vla_type   = "octo",
        expected_emb_dim = 384,
        vae_type   = "text_cond_beta_tcvae",
        z_dim      = 128,
        beta       = 0.001,
        vae_seed   = 2,
        suite      = "libero_spatial",
    ),
    dict(
        label      = "octo / CVAE",
        path       = "./checkpoints/projectors/octo/libero_spatial/vla_emb_cache_text_cvae_arch_cvae_beta0.001_z64.pt",
        vla_type   = "octo",
        expected_emb_dim = 384,
        vae_type   = "text_cvae",
        z_dim      = 64,
        beta       = 0.001,
        vae_seed   = 1,
        suite      = "libero_spatial",
    ),
    dict(
        label      = "openvla / TCN",
        path       = "./checkpoints/projectors/openvla/libero_spatial/vla_emb_cache_text_cond_beta_tcvae_arch_tcn_beta0.001_z128_text_clip.pt",
        vla_type   = "openvla",
        expected_emb_dim = 4096,
        vae_type   = "text_cond_beta_tcvae",
        z_dim      = 128,
        beta       = 0.001,
        vae_seed   = 2,
        suite      = "libero_spatial",
    ),
    dict(
        label      = "openvla / CVAE",
        path       = "./checkpoints/projectors/openvla/libero_spatial/vla_emb_cache_text_cvae_arch_cvae_beta0.001_z64_text_clip.pt",
        vla_type   = "openvla",
        expected_emb_dim = 4096,
        vae_type   = "text_cvae",
        z_dim      = 64,
        beta       = 0.001,
        vae_seed   = 1,
        suite      = "libero_spatial",
    ),
]

VAE_CKPT_TEMPLATE = (
    "./checkpoints/text_tcvae/libero_spatial/"
    "rw100_dropout0.15_beta{beta}_z{z_dim}_alpha1.0_chunk8_std_cyc4_vel0.5_{arch}_seed_{seed}_step_{step}.pt"
)

# ---------------------------------------------------------------------------
PASS = "\033[92m✅ PASS\033[0m"
FAIL = "\033[91m❌ FAIL\033[0m"
WARN = "\033[93m⚠️  WARN\033[0m"

def pf(ok, msg):
    print(f"  {'  PASS  ' if ok else '  FAIL  '} {msg}")
    return ok

def check(cond, msg):
    tag = PASS if cond else FAIL
    print(f"  {tag}  {msg}")
    return cond

# ---------------------------------------------------------------------------
def load_stats(stats_path, suite):
    full = torch.load(stats_path, map_location="cpu")
    key  = f"{suite}_no_noops"
    if key not in full:
        key = "libero_spatial_no_noops"
    s   = full[key]["action"]
    return (
        torch.tensor(s["min"]).float(),
        torch.tensor(s["max"]).float(),
        torch.tensor(s["mask"]).float(),
    )

def normalize(actions, amin, amax, amask):
    norm = (actions - amin) / (amax - amin + 1e-5) * 2.0 - 1.0
    return norm * amask + actions * (1.0 - amask)

def unnormalize(actions_norm, amin, amax, amask):
    unnorm = (actions_norm + 1.0) / 2.0 * (amax - amin) + amin
    return unnorm * amask + actions_norm * (1.0 - amask)

def load_vae(cfg):
    arch = "tcn" if cfg["vae_type"] == "text_cond_beta_tcvae" else "cvae"
    ckpt = VAE_CKPT_TEMPLATE.format(
        beta=cfg["beta"], z_dim=cfg["z_dim"],
        arch=arch, seed=cfg["vae_seed"], step=STEP
    )
    if not os.path.exists(ckpt):
        print(f"  {WARN}  VAE checkpoint not found: {ckpt} — skipping re-encode check")
        return None

    if arch == "tcn":
        vae = TCNTextActionBetaTCVAE(
            action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE, latent_dim=cfg["z_dim"],
            text_emb_dim=512, beta=cfg["beta"], dropout=0.15,
            hidden_channels=64, n_blocks=N_BLOCKS,
        ).to(DEVICE)
    else:
        vae = TCNTextActionCVAE(
            action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE, latent_dim=cfg["z_dim"],
            text_emb_dim=512, beta=cfg["beta"], dropout=0.15,
            hidden_channels=64, n_blocks=N_BLOCKS, enc_text_gate_init=0.0,
        ).to(DEVICE)
    vae.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    vae.eval()
    return vae

# ---------------------------------------------------------------------------
def run_checks(cfg):
    label = cfg["label"]
    path  = cfg["path"]
    print(f"\n{'='*60}")
    print(f"  Cache: {label}")
    print(f"  Path:  {path}")
    print(f"{'='*60}")

    if not os.path.exists(path):
        print(f"  {FAIL}  File not found — skipping")
        return

    c = torch.load(path, map_location="cpu")
    all_ok = True

    # -- Check 1: required keys --
    required = ["train_emb", "train_teacher_mu", "train_teacher_lv",
                "train_clip_emb", "train_actions",
                "test_emb",  "test_teacher_mu",  "test_teacher_lv",
                "test_clip_emb",  "test_actions"]
    missing = [k for k in required if k not in c]
    all_ok &= check(not missing, f"All required keys present" + (f" [missing: {missing}]" if missing else ""))
    if missing:
        return

    train_emb  = c["train_emb"]
    train_acts = c["train_actions"]
    train_mu   = c["train_teacher_mu"]
    train_lv   = c["train_teacher_lv"]

    # -- Check 2: embedding dim --
    actual_dim = train_emb.shape[-1]
    all_ok &= check(actual_dim == cfg["expected_emb_dim"],
                    f"Embedding dim = {actual_dim} (expected {cfg['expected_emb_dim']})")

    # -- Check 3: teacher_mu z_dim --
    actual_z = train_mu.shape[-1]
    all_ok &= check(actual_z == cfg["z_dim"],
                    f"teacher_mu latent dim = {actual_z} (expected {cfg['z_dim']})")

    # -- Check 4: action normalization range [-1, 1] --
    act_min = train_acts.min().item()
    act_max = train_acts.max().item()
    act_ok  = act_min >= -1.05 and act_max <= 1.05
    all_ok &= check(act_ok,
                    f"train_actions range [{act_min:.4f}, {act_max:.4f}] "
                    f"{'✓ in [-1,1]' if act_ok else '✗ OUT OF RANGE — normalization missing!'}")

    # -- Check 5: teacher_mu sanity (not collapsed to zero, not exploded) --
    mu_norm = train_mu.norm(dim=-1).mean().item()
    mu_std  = train_mu.std(dim=0).mean().item()
    all_ok &= check(mu_norm > 0.1,  f"teacher_mu mean L2 norm = {mu_norm:.4f} (should be > 0.1)")
    all_ok &= check(mu_std  > 0.01, f"teacher_mu dim-wise std  = {mu_std:.4f}  (should be > 0.01)")

    # -- Check 6: teacher_lv not collapsed --
    lv_std = train_lv.std().item()
    lv_mean = train_lv.mean().item()
    all_ok &= check(lv_std > 0.01,
                    f"teacher_lv std = {lv_std:.4f}, mean = {lv_mean:.4f} "
                    f"({'ok' if lv_std > 0.01 else 'collapsed!'})")

    # -- Check 7: action stats round-trip (norm → unnorm → norm) --
    amin, amax, amask = load_stats(STATS_PATH, cfg["suite"])
    renorm = normalize(unnormalize(train_acts, amin, amax, amask), amin, amax, amask)
    rt_err = (renorm - train_acts).abs().max().item()
    all_ok &= check(rt_err < 1e-4,
                    f"Norm round-trip error = {rt_err:.6f} (should be < 1e-4)")

    # -- Check 8: re-encode 128 stored actions through the VAE; compare to stored mu --
    vae = load_vae(cfg)
    if vae is not None:
        with torch.no_grad():
            batch_a = train_acts[:128].to(DEVICE)
            if cfg["vae_type"] == "text_cvae":
                # Same convention as cache build: zero text in encoder
                zero_t = torch.zeros(128, 512, device=DEVICE)
                re_mu, _ = vae.encode(batch_a, zero_t)
            else:
                re_mu, _ = vae.encode(batch_a)
        re_mu = re_mu.float().cpu()
        stored_mu = train_mu[:128]
        cos_sim = F.cosine_similarity(re_mu, stored_mu, dim=-1).mean().item()
        mse     = F.mse_loss(re_mu, stored_mu).item()
        cos_ok  = cos_sim > 0.95
        all_ok &= check(cos_ok,
                        f"Re-encode match: cos_sim = {cos_sim:.4f}, MSE = {mse:.6f} "
                        f"({'✓ actions were correctly normalised at build time' if cos_ok else '✗ MISMATCH — actions were NOT correctly normalised!'})")

    # -- Check 9: CLIP emb sanity --
    clip_norm = c["train_clip_emb"].norm(dim=-1).mean().item()
    all_ok &= check(0.5 < clip_norm < 50.0,
                    f"CLIP emb mean L2 norm = {clip_norm:.4f} (expected ~15–25)")

    # -- Check 10: train/test split sizes --
    n_train = train_emb.shape[0]
    n_test  = c["test_emb"].shape[0]
    print(f"  {'  INFO  '}  train N = {n_train}, test N = {n_test}, "
          f"ratio = {n_train/(n_train+n_test):.2f}")

    print(f"\n  {'ALL CHECKS PASSED ✅' if all_ok else 'SOME CHECKS FAILED ❌'}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(STATS_PATH):
        print(f"ERROR: stats file not found at {STATS_PATH}")
        sys.exit(1)

    print(f"\nLoading action stats from: {STATS_PATH}")
    amin, amax, amask = load_stats(STATS_PATH, "libero_spatial")
    print(f"  action_min  = {amin.tolist()}")
    print(f"  action_max  = {amax.tolist()}")
    print(f"  action_mask = {amask.tolist()}")

    for cfg in CACHES:
        run_checks(cfg)

    print(f"\n{'='*60}")
    print("  Done.")
