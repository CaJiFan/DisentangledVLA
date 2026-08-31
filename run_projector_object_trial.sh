#!/bin/bash
# =============================================================================
# TRIAL PROJECTOR RUN: libero_object with Champion Fixed Beta=0.1 VAE
# =============================================================================

SUITE="libero_object"
VLA_TYPE="pi0"
Z_DIM=128
LOSS="flow"
SEED=1

# 🏆 Champion VAE (78.20% / ~80% Peak):
VAE_CKPT="./checkpoints/new_protocol_cvae/libero_object/rw100_d0.1_beta0.1-0.1_z128_chunk8_protA_cond_prior_SPIRL_state_h1_grip5.0_seed_1_best.pt"

# Fallback if specific file name is slightly different
if [ ! -f "${VAE_CKPT}" ]; then
    VAE_CKPT="./checkpoints/new_protocol_cvae/libero_object/rw100_d0.1_beta0.1_z128_chunk8_protA_cond_prior_SPIRL_state_h1_grip5.0_seed_1_best.pt"
fi

# Shared Transformer Projector Hyperparameters
XFMR_D_MODEL=256
XFMR_HEADS=8
XFMR_LAYERS=3
XFMR_FFN=512

LOG_DIR="logs/projectors/${SUITE}/z${Z_DIM}"
mkdir -p "${LOG_DIR}"

echo "===================================================================="
echo "🚀 STARTING TRIAL PROJECTOR TRAINING"
echo "   Suite:         ${SUITE}"
echo "   VLA Backbone:  ${VLA_TYPE}"
echo "   Teacher VAE:   $(basename ${VAE_CKPT})"
echo "   Loss:          ${LOSS} (Fast Closed-Form Gaussian KL)"
echo "   Latent Dim z:  ${Z_DIM}"
echo "===================================================================="

# Run KL Transformer Projector (or change to 'mlp')
for PROJ_ARCH in "flow_transformer"
do
    LOG_FILE="${LOG_DIR}/${SUITE}_${VLA_TYPE}_condprior_z${Z_DIM}_${PROJ_ARCH}_loss_${LOSS}_seed_${SEED}.log"

    echo ""
    echo "▶ Training: ${PROJ_ARCH} (Seed: ${SEED}) | Log: ${LOG_FILE}"

    python3 scripts/train_projector.py \
        --suite ${SUITE} \
        --vla_type ${VLA_TYPE} \
        --vae_z_dim ${Z_DIM} \
        --projector_type prob \
        --projector_arch ${PROJ_ARCH} \
        --vae_type cond_prior \
        --use_state_cond \
        --loss ${LOSS} \
        --text_backbone clip \
        --vae_checkpoint "${VAE_CKPT}" \
        --xfmr_d_model ${XFMR_D_MODEL} \
        --xfmr_num_heads ${XFMR_HEADS} \
        --xfmr_num_layers ${XFMR_LAYERS} \
        --xfmr_ffn_dim ${XFMR_FFN} \
        --num_fusion_layers 3 \
        --ortho_weight 0.01 \
        --normalize_emb 1 \
        --dropout 0.1 \
        --weight_decay 0.01 \
        --lr 5e-4 \
        --action_recon_weight 0.0 \
        --batch_size 256 \
        --eval_every 10000 \
        --patience 5 \
        --max_steps 2000000 \
        --lr_decay_steps 1000000 \
        --seed ${SEED} \
        2>&1 | tee "${LOG_FILE}"
done

echo ""
echo "===================================================================="
echo "✅ TRIAL PROJECTOR RUN COMPLETE!"
echo "===================================================================="
