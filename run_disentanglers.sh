#!/bin/bash

echo "=================================================="
echo "🚀 STARTING DISENTANGLER TRAINING PIPELINE"
echo "   Grid: z_dim ∈ {64, 128} × arch ∈ {TCN-DecOnly, FullCVAE, CondPrior, WAE}"
echo "   Fixed: beta=0.1 (best reconstruction), seeds=2,3"
echo "   8 configs × 2 seeds = 16 runs total, serialised on one GPU."
echo "=================================================="

export HF_TOKEN="<YOUR_HUGGINGFACE_TOKEN>"

# ---------------------------------------------------------------------------
# Shared hyper-parameters
# ---------------------------------------------------------------------------
BETA=0.1          # Fixed: prioritise reconstruction
DROPOUT=0.15      # Classifier-free guidance dropout on text
CHUNK_SIZE=8
RECON_WEIGHT=100
N_CYCLES=4        # Cyclic beta annealing (helps KL not collapse early)
TEXT_BACKBONE="clip"
SUITE="libero_spatial"

# ---------------------------------------------------------------------------
# SWEEP GRID
#   z_dim : 64  — compact latent, faster projector training, easier to generalise
#           128 — richer latent, more capacity for diverse trajectories
#
#   arch  :
#     --use-tcn   → TCN Decoder-Only (text_cond_beta_tcvae)
#                   Text feeds *only* the decoder. Encoder is text-free.
#                   Potentially purer latent space, cleaner for projector.
#
#     --use-cvae  → Full CVAE (text_cvae)
#                   Text feeds encoder+decoder via learned soft gate.
#                   Better conditioned reconstruction, but z is partially
#                   text-entangled (may hurt projector generalisation).
#
#     --use-cond-prior → TCN Decoder-Only with Conditional Prior
#                   Learns p(z|text) instead of N(0,1). Better at capturing
#                   multimodal task distributions cleanly.
#
#     --use-wae   → Wasserstein Autoencoder with TCN Decoder-Only
#                   Uses MMD instead of KL divergence for sharper trajectories.
# ---------------------------------------------------------------------------

for SEED in 1 2 3
do
    for LATENT_DIM in 128
    do

        # # ==============================================================
        # # BLOCK A — TCN Decoder-Only  (--use-tcn)
        # # ==============================================================
        # LOG_DIR="logs/new_protocol_cvae/tcn_deconly/${SUITE}"
        # mkdir -p "${LOG_DIR}"
        # LOG_FILE="${LOG_DIR}/beta${BETA}_z${LATENT_DIM}_rw${RECON_WEIGHT}_seed${SEED}.log"

        # echo ""
        # echo "=========================================================="
        # echo "[TCN Decoder-Only] z=${LATENT_DIM}  β=${BETA}  seed=${SEED}"
        # echo "  Log: ${LOG_FILE}"
        # echo "=========================================================="

        # python3 scripts/train_text_conditioned_disentangler.py \
        #         --suite          ${SUITE}         \
        #         --use-tcn                         \
        #         --text_backbone  ${TEXT_BACKBONE} \
        #         --beta           ${BETA}          \
        #         --latent_dim     ${LATENT_DIM}    \
        #         --chunk_size     ${CHUNK_SIZE}    \
        #         --dropout        ${DROPOUT}       \
        #         --recon_weight   ${RECON_WEIGHT}  \
        #         --n_cycles       ${N_CYCLES}      \
        #         --seed           ${SEED}          \
        #         > ${LOG_FILE} 2>&1

        # # ==============================================================
        # # BLOCK B — Full CVAE  (--use-cvae)
        # # ==============================================================
        # LOG_DIR="logs/new_protocol_cvae/full_cvae/${SUITE}"
        # mkdir -p "${LOG_DIR}"
        # LOG_FILE="${LOG_DIR}/beta${BETA}_z${LATENT_DIM}_rw${RECON_WEIGHT}_seed${SEED}.log"

        # echo ""
        # echo "=========================================================="
        # echo "[Full CVAE] z=${LATENT_DIM}  β=${BETA}  seed=${SEED}"
        # echo "  Log: ${LOG_FILE}"
        # echo "=========================================================="

        # python3 scripts/train_text_conditioned_disentangler.py \
        #         --suite                ${SUITE}         \
        #         --use-cvae                              \
        #         --enc_text_gate_init   0.0              \
        #         --text_backbone        ${TEXT_BACKBONE} \
        #         --beta                 ${BETA}          \
        #         --latent_dim           ${LATENT_DIM}    \
        #         --chunk_size           ${CHUNK_SIZE}    \
        #         --dropout              ${DROPOUT}       \
        #         --recon_weight         ${RECON_WEIGHT}  \
        #         --n_cycles             ${N_CYCLES}      \
        #         --seed                 ${SEED}          \
        #         > ${LOG_FILE} 2>&1

        # ==============================================================
        # BLOCK C — Conditional Prior CVAE  (--use-cond-prior)
        # ==============================================================
        LOG_DIR="logs/new_protocol_cvae/cond_prior/${SUITE}"
        mkdir -p "${LOG_DIR}"
        LOG_FILE="${LOG_DIR}/beta${BETA}_z${LATENT_DIM}_rw${RECON_WEIGHT}_seed${SEED}.log"

        echo ""
        echo "=========================================================="
        echo "[Cond Prior] z=${LATENT_DIM}  β=${BETA}  seed=${SEED}"
        echo "  Log: ${LOG_FILE}"
        echo "=========================================================="

        python3 scripts/train_text_conditioned_disentangler.py \
                --suite                ${SUITE}         \
                --use-cond-prior                        \
                --text_backbone        ${TEXT_BACKBONE} \
                --beta                 ${BETA}          \
                --latent_dim           ${LATENT_DIM}    \
                --chunk_size           ${CHUNK_SIZE}    \
                --dropout              ${DROPOUT}       \
                --recon_weight         ${RECON_WEIGHT}  \
                --n_cycles             ${N_CYCLES}      \
                --seed                 ${SEED}          \
                > ${LOG_FILE} 2>&1

        # # ==============================================================
        # # BLOCK D — Wasserstein AE  (--use-wae)
        # # ==============================================================
        # LOG_DIR="logs/new_protocol_cvae/wae/${SUITE}"
        # mkdir -p "${LOG_DIR}"
        # LOG_FILE="${LOG_DIR}/beta${BETA}_z${LATENT_DIM}_rw${RECON_WEIGHT}_seed${SEED}.log"

        # echo ""
        # echo "=========================================================="
        # echo "[WAE] z=${LATENT_DIM}  β=${BETA}  seed=${SEED}"
        # echo "  Log: ${LOG_FILE}"
        # echo "=========================================================="

        # python3 scripts/train_text_conditioned_disentangler.py \
        #         --suite                ${SUITE}         \
        #         --use-wae                               \
        #         --text_backbone        ${TEXT_BACKBONE} \
        #         --beta                 ${BETA}          \
        #         --latent_dim           ${LATENT_DIM}    \
        #         --chunk_size           ${CHUNK_SIZE}    \
        #         --dropout              ${DROPOUT}       \
        #         --recon_weight         ${RECON_WEIGHT}  \
        #         --n_cycles             ${N_CYCLES}      \
        #         --seed                 ${SEED}          \
        #         > ${LOG_FILE} 2>&1

    done
done

echo ""
echo "=================================================="
echo "✅ ALL DISENTANGLER RUNS COMPLETE"
echo "=================================================="
