#!/bin/bash
# =============================================================================
# EVALUATE TRAINED PROJECTORS WITH TEMPORAL ENSEMBLING
# =============================================================================

SUITE="libero_spatial"
VAE_CKPT="./checkpoints/new_protocol_cvae/libero_spatial/rw100_d0.15_beta0.1_z128_chunk8_protA_cyc4_cond_prior_seed_2_step_250000.pt"

# Fallback to text_tcvae if new_protocol_cvae doesn't exist
if [ ! -f "${VAE_CKPT}" ]; then
    VAE_CKPT="./checkpoints/text_tcvae/libero_spatial/rw100_dropout0.15_beta0.1_z128_alpha1.0_chunk8_std_cyc4_vel0.5_cvae_seed_2_step_100000.pt"
fi

NUM_ROLLOUTS=20
ENSEMBLE_K=0.01

LOG_DIR="eval_logs/projectors/${SUITE}"
mkdir -p "${LOG_DIR}"

echo "=================================================="
echo "🚀 STARTING CLOSED-LOOP PROJECTOR EVALUATION"
echo "   Suite:           ${SUITE}"
echo "   Rollouts/Task:   ${NUM_ROLLOUTS}"
echo "   Temporal Ensemble: ON (k=${ENSEMBLE_K})"
echo "   VAE Checkpoint:  $(basename ${VAE_CKPT})"
echo "=================================================="

# Checkpoints to evaluate:
declare -a PROJ_CKPTS=(
    # 1. KL-based Transformer Projector (200k steps)
    "checkpoints/projectors/openvla/libero_spatial/chunk_8_zdim_128/prob_transformer_loss_kl_seed_1_step_200000.pt"
    # 2. Flow-based Transformer Projector (200k steps)
    "checkpoints/projectors/openvla/libero_spatial/chunk_8_zdim_128/prob_flow_transformer_loss_flow_seed_1_step_200000.pt"
    # 3. KL-based MLP Projector (200k steps)
    "checkpoints/projectors/openvla/libero_spatial/chunk_8_zdim_128/prob_mlp_loss_kl_seed_1_step_200000.pt"
)

for PROJ_CKPT in "${PROJ_CKPTS[@]}"
do
    if [ ! -f "${PROJ_CKPT}" ]; then
        echo "⚠️ Checkpoint not found: ${PROJ_CKPT} — skipping."
        continue
    fi

    CKPT_BASE=$(basename ${PROJ_CKPT} .pt)
    LOG_FILE="${LOG_DIR}/${CKPT_BASE}_temporal_k${ENSEMBLE_K}.log"

    echo ""
    echo "▶ Evaluating: ${CKPT_BASE}"
    echo "  Log: ${LOG_FILE}"

    python3 scripts/eval_projector_suite.py \
        --projector_checkpoint "${PROJ_CKPT}" \
        --vae_checkpoint       "${VAE_CKPT}" \
        --suite                "${SUITE}" \
        --num_rollouts         ${NUM_ROLLOUTS} \
        --temporal_ensemble \
        --ensemble_k           ${ENSEMBLE_K} \
        > "${LOG_FILE}" 2>&1

    # Extract and display result from log
    FINAL_SR=$(grep "FINAL SUCCESS RATE:" "${LOG_FILE}" | tail -n 1)
    echo "  👉 ${FINAL_SR:-"Run completed (see log)"}"
done

echo ""
echo "=================================================="
echo "✅ ALL PROJECTOR EVALUATIONS COMPLETE"
echo "=================================================="
