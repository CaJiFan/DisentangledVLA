#!/bin/bash

echo "=================================================="
echo "🚀 STARTING VLA PROJECTOR ABLATION PIPELINE"
echo "   Protocol A: 10-task training, simulator eval only"
echo "=================================================="

# Fix W&B Git "dubious ownership" crash inside Docker
git config --global --add safe.directory '*' || true

wait_for_queue() {
    # Block until previous background job finishes.
    while [ $(jobs -p | wc -l) -ge 1 ]; do
        sleep 30
    done
}

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
Z_DIM_1=128
VAE_CKPT_1="./checkpoints/new_protocol_cvae/libero_spatial/rw100_d0.15_beta0.1_z128_chunk8_protA_cyc4_cond_prior_seed_2_step_250000.pt"

echo "Running OCTO Block..."
for VLA_TYPE in "octo"
do
    for LAYER_IDX in -1
    do
        for PROJ_ARCH in "flow_transformer"
        do
            for SEED in 2
            do
                echo "Running Suite: ${SUITE} (${VLA_TYPE} | CondPrior z=${Z_DIM_1} | ${PROJ_ARCH} Proj | layer=${LAYER_IDX} | seed=${SEED})"
                
                LOG_DIR="logs/projectors/protA/z${Z_DIM_1}"
                mkdir -p "${LOG_DIR}"
                LOG_FILE="${LOG_DIR}/${SUITE}_${VLA_TYPE}_condprior_z${Z_DIM_1}_${PROJ_ARCH}_arw${ACTION_RECON_WEIGHT}_layer${LAYER_IDX}_seed_${SEED}.log"

                python3 scripts/train_projector.py \
                    --suite ${SUITE} \
                    --vla_type ${VLA_TYPE} \
                    --vae_z_dim ${Z_DIM_1} \
                    --projector_type prob \
                    --projector_arch ${PROJ_ARCH} \
                    --vae_type "cond_prior" \
                    --loss flow \
                    --text_backbone "clip" \
                    --vae_checkpoint ${VAE_CKPT_1} \
                    --xfmr_d_model ${XFMR_D_MODEL} \
                    --xfmr_num_heads ${XFMR_HEADS} \
                    --xfmr_num_layers ${XFMR_LAYERS} \
                    --xfmr_ffn_dim ${XFMR_FFN} \
                    --emb_noise_std 0.0 \
                    --normalize_emb 1 \
                    --dropout 0.1 \
                    --weight_decay 0.01 \
                    --action_recon_weight ${ACTION_RECON_WEIGHT} \
                    --vla_layer_idx ${LAYER_IDX} \
                    --seed ${SEED} \
                    > ${LOG_FILE} 2>&1
                    
                echo "🧹 Cleaning up any leftover JAX/DataLoader zombie processes to free VRAM..."
                pkill -u $USER -f "multiprocessing.spawn" || true
                sleep 5
                    
                # wait_for_queue
            done
        done
    done
done

echo "Running OPENVLA Block..."
for VLA_TYPE in "openvla"
do
    for LAYER_IDX in 16 -1
    do
        for PROJ_ARCH in "flow_transformer"
        do
            for SEED in 2
            do
                echo "Running Suite: ${SUITE} (${VLA_TYPE} | CondPrior z=${Z_DIM_1} | ${PROJ_ARCH} Proj | layer=${LAYER_IDX} | seed=${SEED})"
                
                LOG_DIR="logs/projectors/protA/z${Z_DIM_1}"
                mkdir -p "${LOG_DIR}"
                LOG_FILE="${LOG_DIR}/${SUITE}_${VLA_TYPE}_condprior_z${Z_DIM_1}_${PROJ_ARCH}_arw${ACTION_RECON_WEIGHT}_layer${LAYER_IDX}_seed_${SEED}.log"

                python3 scripts/train_projector.py \
                    --suite ${SUITE} \
                    --vla_type ${VLA_TYPE} \
                    --vae_z_dim ${Z_DIM_1} \
                    --projector_type prob \
                    --projector_arch ${PROJ_ARCH} \
                    --vae_type "cond_prior" \
                    --loss flow \
                    --text_backbone "clip" \
                    --vae_checkpoint ${VAE_CKPT_1} \
                    --xfmr_d_model ${XFMR_D_MODEL} \
                    --xfmr_num_heads ${XFMR_HEADS} \
                    --xfmr_num_layers ${XFMR_LAYERS} \
                    --xfmr_ffn_dim ${XFMR_FFN} \
                    --emb_noise_std 0.0 \
                    --normalize_emb 1 \
                    --dropout 0.1 \
                    --weight_decay 0.01 \
                    --action_recon_weight ${ACTION_RECON_WEIGHT} \
                    --vla_layer_idx ${LAYER_IDX} \
                    --seed ${SEED} \
                    > ${LOG_FILE} 2>&1
                    
                echo "🧹 Cleaning up any leftover JAX/DataLoader zombie processes to free VRAM..."
                pkill -u $USER -f "multiprocessing.spawn" || true
                sleep 5
                    
                # wait_for_queue
            done
        done
    done
done

# ====================================================================
# COMMENTED OUT: THE CLASSIC BASELINE & THE UNCONDITIONED BASELINE
# ====================================================================
# (Code omitted to focus exclusively on the champion cond_prior)

echo "✅ ALL PROJECTOR RUNS COMPLETE"