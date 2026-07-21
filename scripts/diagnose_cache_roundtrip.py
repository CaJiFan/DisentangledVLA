"""
Diagnoses why train/teacher_action_recon_loss is stuck at ~0.078.
Runs inside the Docker container: python3 scripts/diagnose_cache_roundtrip.py
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn.functional as F

from src.disentanglers import TCNTextActionBetaTCVAE, TCNTextActionCVAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT   = "/workspace/DisentangledFlow"

# ── VAE config ────────────────────────────────────────────────────────────────
VAE_CKPT = f"{ROOT}/checkpoints/text_tcvae/libero_spatial/rw100_dropout0.15_beta0.001_z128_alpha1.0_chunk8_std_cyc4_vel0.5_tcn_seed_2_step_100000.pt"

# ── Cache paths ────────────────────────────────────────────────────────────────
CACHES = {
    "OpenVLA_TCN": f"{ROOT}/checkpoints/projectors/openvla/libero_spatial/vla_emb_cache_text_cond_beta_tcvae.pt",
    "Octo_TCN":    f"{ROOT}/checkpoints/projectors/octo/libero_spatial/vla_emb_cache_text_cond_beta_tcvae_arch_tcn_beta0.001_z128.pt",
}

# ── Load VAE ──────────────────────────────────────────────────────────────────
print("\n==== Loading VAE ====")
vae = TCNTextActionBetaTCVAE(
    action_dim=7, chunk_size=8, latent_dim=128, text_emb_dim=512,
    beta=0.001, dropout=0.15, hidden_channels=64, n_blocks=3,
).to(DEVICE).eval()
vae.load_state_dict(torch.load(VAE_CKPT, map_location=DEVICE))
print(f"  Loaded: {VAE_CKPT}")

N = 512  # samples to check

for name, path in CACHES.items():
    if not os.path.exists(path):
        print(f"\n[SKIP] {name}: {path} not found")
        continue

    print(f"\n{'='*60}")
    print(f"  Cache: {name}")
    c = torch.load(path, map_location="cpu")

    acts     = c["train_actions"][:N].to(DEVICE)
    mu       = c["train_teacher_mu"][:N].to(DEVICE)
    clip_emb = c.get("train_clip_emb", torch.zeros(N, 512))[:N].to(DEVICE)

    print(f"  Actions  : shape={tuple(acts.shape)} | min={acts.min():.3f} max={acts.max():.3f} mean={acts.mean():.3f} std={acts.std():.3f}")
    print(f"  teacher_mu: shape={tuple(mu.shape)} | norm={mu.norm(dim=-1).mean():.3f}")
    print(f"  clip_emb  : shape={tuple(clip_emb.shape)} | zeros={( clip_emb.abs().sum(-1)==0).float().mean():.1%} | norm={clip_emb.norm(dim=-1).mean():.3f}")

    with torch.no_grad():
        # 1. Direct roundtrip: encode(acts) → decode with real CLIP
        mu2, lv2 = vae.encode(acts)
        recon_from_encode = vae.decode(mu2, clip_emb)
        mse_direct = F.mse_loss(recon_from_encode, acts).item()
        print(f"\n  [A] Encode→Decode MSE (real clip)    : {mse_direct:.5f}  ← VAE reconstruction floor")

        # 2. Direct roundtrip with zero CLIP
        zero_clip = torch.zeros_like(clip_emb)
        recon_zero = vae.decode(mu2, zero_clip)
        mse_zero_clip = F.mse_loss(recon_zero, acts).item()
        print(f"  [B] Encode→Decode MSE (zero  clip)   : {mse_zero_clip:.5f}")

        # 3. Stored teacher_mu → decode with stored clip  (this IS teacher_action_recon_loss)
        recon_teacher_real = vae.decode(mu, clip_emb)
        mse_teacher_real = F.mse_loss(recon_teacher_real, acts).item()
        print(f"  [C] StoredMu→Decode MSE (real clip)  : {mse_teacher_real:.5f}  ← matches wandb metric?")

        # 4. Stored teacher_mu → decode with zero CLIP
        recon_teacher_zero = vae.decode(mu, zero_clip)
        mse_teacher_zero = F.mse_loss(recon_teacher_zero, acts).item()
        print(f"  [D] StoredMu→Decode MSE (zero  clip) : {mse_teacher_zero:.5f}")

        # 5. Cosine sim between re-encoded mu and stored mu
        cos = F.cosine_similarity(mu2, mu).mean().item()
        print(f"\n  [E] CosSim(re-encoded mu, stored mu) : {cos:.4f}")

        # 6. MSE between re-encoded mu and stored mu
        mu_mse = F.mse_loss(mu2, mu).item()
        print(f"  [F] MSE(re-encoded mu, stored mu)    : {mu_mse:.5f}")

        # 7. Check if stored clip_emb actually matters for decode quality
        # by testing decode quality variation across tasks
        if clip_emb.abs().sum() > 0:
            # sample 2 different tasks' clip vecs and swap
            clip_a = clip_emb[:N//2]
            clip_b = clip_emb[N//2:]
            acts_a = acts[:N//2]
            mu_a   = mu[:N//2]
            recon_swapped = vae.decode(mu_a, clip_b)
            mse_swapped = F.mse_loss(recon_swapped, acts_a).item()
            print(f"  [G] StoredMu→Decode MSE (swapped clip): {mse_swapped:.5f}  ← how much text matters")

        # 8. Per-DoF error to spot which dimension is worst
        recon_c = vae.decode(mu, clip_emb)
        per_dof_mse = ((recon_c - acts)**2).mean(dim=(0,1))  # (action_dim,)
        print(f"\n  Per-DoF MSE [C]: {[f'{x:.4f}' for x in per_dof_mse.tolist()]}")
        print(f"   → dims 0-5 (continuous), dim 6 (gripper)")

print("\n==== Done ====")
