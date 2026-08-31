#!/bin/bash

echo "=================================================="
echo "🚀 STARTING PI0 PROJECTOR TRAINING PIPELINE"
echo "   Protocol A: 10-task training, simulator eval only"
echo "=================================================="

# Fix W&B Git "dubious ownership" crash inside Docker
git config --global --add safe.directory '*' || true

export HF_TOKEN="<YOUR_HUGGINGFACE_TOKEN>"

SUITE="libero_spatial"
ACTION_RECON_WEIGHT=0.5

# Shared Transformer Projector Hyperparameters
XFMR_D_MODEL=256
XFMR_HEADS=8
XFMR_LAYERS=3
XFMR_FFN=512

echo ""
echo "===================================================================="
echo " 🏆 THE CHAMPION: CondPrior | z=128 | beta=0.1 | seed=2 "
echo "===================================================================="
Z_DIM=128
VAE_CKPT="./checkpoints/new_protocol_cvae/libero_spatial/rw100_d0.15_beta0.1_z128_chunk8_protA_cyc4_cond_prior_seed_2_step_250000.pt"

VAE_SEED=2
PROJ_SEED=2

VLA_TYPE="pi0"
PROJ_ARCH="flow_transformer"
NUM_FUSION_LAYERS=3

echo "Running Suite: ${SUITE} (${VLA_TYPE} | CondPrior z=${Z_DIM} (Seed ${VAE_SEED}) | ${PROJ_ARCH} Proj | num_fusion_layers=${NUM_FUSION_LAYERS} | proj_seed=${PROJ_SEED})"

LOG_DIR="logs/projectors/protA/pi0/z${Z_DIM}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${SUITE}_${VLA_TYPE}_condprior_z${Z_DIM}_${PROJ_ARCH}_arw${ACTION_RECON_WEIGHT}_fusion${NUM_FUSION_LAYERS}_seed_${PROJ_SEED}.log"

python3 scripts/train_projector.py \
    --suite ${SUITE} \
    --vla_type ${VLA_TYPE} \
    --vae_z_dim ${Z_DIM} \
    --projector_type prob \
    --projector_arch ${PROJ_ARCH} \
    --vae_type "cond_prior" \
    --loss flow \
    --text_backbone "clip" \
    --vae_checkpoint ${VAE_CKPT} \
    --xfmr_d_model ${XFMR_D_MODEL} \
    --xfmr_num_heads ${XFMR_HEADS} \
    --xfmr_num_layers ${XFMR_LAYERS} \
    --xfmr_ffn_dim ${XFMR_FFN} \
    --emb_noise_std 0.0 \
    --normalize_emb 1 \
    --dropout 0.1 \
    --weight_decay 0.01 \
    --action_recon_weight ${ACTION_RECON_WEIGHT} \
    --num_fusion_layers ${NUM_FUSION_LAYERS} \
    --ortho_weight 0.01 \
    --seed ${PROJ_SEED} \
    --eval_every 50000 \
    --patience 3 \
    --max_steps 10000000 \
    --lr_decay_steps 1000000 \
    --lr 5e-4 \
    > ${LOG_FILE} 2>&1

echo "✅ PI0 PROJECTOR RUN COMPLETE"
