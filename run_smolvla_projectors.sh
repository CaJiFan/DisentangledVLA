#!/bin/bash

# SmolVLA projector training runs (KL distillation, cache-based).
# Requires the SmolVLA embedding caches to exist first:
#   checkpoints/projectors/smolvla/libero_spatial/vla_emb_cache_text_cond_beta_tcvae_arch_tcn_beta0.001_z128.pt
#   checkpoints/projectors/smolvla/libero_spatial/vla_emb_cache_text_cvae_arch_cvae_beta0.001_z64.pt
#
# Run this in smolvla_worker (lerobot installed, LIBERO sim installed).

# MuJoCo headless rendering — use osmesa (software) or egl (GPU).
# osmesa is more reliable in Docker; switch to egl if you have egl_probe installed.
export MUJOCO_GL="egl"

DROPOUT=0.15
CHUNK_SIZE=8
ALPHA=1.0
STEP=100000
RECON_WEIGHT=100
SUITE="libero_spatial"
VAE_SUITE="libero_spatial"
TEXT_BACKBONE="smollm"
ACTION_RECON_WEIGHT=0.5

# ==============================================================================
# SmolVLA + TCN (z=128, beta=0.001, seed=2)
# ==============================================================================
BETA=0.001
Z_DIM=128
VAE_SEED=1
ARCH="tcn"
VAE_CKPT="./checkpoints/text_tcvae/${VAE_SUITE}/rw${RECON_WEIGHT}_dropout${DROPOUT}_beta${BETA}_z${Z_DIM}_alpha${ALPHA}_chunk${CHUNK_SIZE}_std_text_${TEXT_BACKBONE}_seed_${VAE_SEED}_cyc4_vel0.5_${ARCH}_seed_${VAE_SEED}_step_${STEP}.pt"

for SEED in 1
do
    echo "--------------------------------------------------"
    echo "Running Suite: ${SUITE} (SmolVLA TCN, gripfix ARW=${ACTION_RECON_WEIGHT}, seed=${SEED})"
    echo "--------------------------------------------------"

    python3 scripts/train_projector.py \
        --suite ${SUITE} \
        --vla_type smolvla \
        --smolvla_model lerobot/smolvla_base \
        --vae_z_dim ${Z_DIM} \
        --projector_type prob \
        --vae_type "text_cond_beta_tcvae" \
        --loss kl \
        --text_backbone ${TEXT_BACKBONE} \
        --vae_checkpoint ${VAE_CKPT} \
        --projector_arch mlp \
        --emb_noise_std 0.0 \
        --normalize_emb 0 \
        --dropout 0.1 \
        --weight_decay 0.01 \
        --action_recon_weight ${ACTION_RECON_WEIGHT} \
        --seed ${SEED} \
        > logs/projectors/${SUITE}_smolvla_tcn_arw${ACTION_RECON_WEIGHT}_txt_${TEXT_BACKBONE}_gripfix_proj_prob_kl_seed_${SEED}.log 2>&1
done

# ==============================================================================
# SmolVLA + CVAE (z=64, beta=0.001, seed=1)
# ==============================================================================
BETA=0.001
Z_DIM=64
VAE_SEED=1
ARCH="cvae"
VAE_CKPT="./checkpoints/text_tcvae/${VAE_SUITE}/rw${RECON_WEIGHT}_dropout${DROPOUT}_beta${BETA}_z${Z_DIM}_alpha${ALPHA}_chunk${CHUNK_SIZE}_std_text_${TEXT_BACKBONE}_seed_${VAE_SEED}_cyc4_vel0.5_${ARCH}_seed_${VAE_SEED}_step_${STEP}.pt"

for SEED in 1
do
    echo "--------------------------------------------------"
    echo "Running Suite: ${SUITE} (SmolVLA CVAE, gripfix ARW=${ACTION_RECON_WEIGHT}, seed=${SEED})"
    echo "--------------------------------------------------"

    python3 scripts/train_projector.py \
        --suite ${SUITE} \
        --vla_type smolvla \
        --smolvla_model lerobot/smolvla_base \
        --vae_z_dim ${Z_DIM} \
        --projector_type prob \
        --vae_type "text_cvae" \
        --vae_z_dim ${Z_DIM} \
        --loss kl \
        --text_backbone ${TEXT_BACKBONE} \
        --vae_checkpoint ${VAE_CKPT} \
        --projector_arch mlp \
        --emb_noise_std 0.0 \
        --normalize_emb 0 \
        --dropout 0.1 \
        --weight_decay 0.01 \
        --action_recon_weight ${ACTION_RECON_WEIGHT} \
        --seed ${SEED} \
        > logs/projectors/${SUITE}_smolvla_cvae_arw${ACTION_RECON_WEIGHT}_txt_${TEXT_BACKBONE}_gripfix_proj_prob_kl_seed_${SEED}.log 2>&1
done
