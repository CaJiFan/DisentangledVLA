#!/bin/bash
# =============================================================================
# VAE Convergence Sweep — LIBERO_OBJECT
#
# PARALLELISM STRATEGY (single RTX 3090, 24 GB):
#   - All 9 jobs run in parallel during training (~725 MB VRAM each = 6.5 GB)
#   - Launches are staggered by SEED_STAGGER seconds so the temporary CLIP
#     encoder load (~450 MB) never overlaps across processes.
#   - Single `wait` at the end collects all results.
#   - Wall-clock: ~6 h for all 9 to finish simultaneously.
# =============================================================================

echo "=================================================="
echo "🚀 STARTING VAE CONVERGENCE SWEEP ON LIBERO_OBJECT"
echo "   9 jobs in parallel · staggered startup · RTX 3090 24 GB"
echo "   Settings: max_steps=1000000, patience=20, val_freq=10000"
echo "=================================================="

export HF_TOKEN="<YOUR_HUGGINGFACE_TOKEN>"

# ── Fixed hyperparameters ────────────────────────────────────────────────────
BETA=0.1
DROPOUT=0.1        # SupCon handles task separation; high dropout hurts recon quality
CHUNK_SIZE=8
RECON_WEIGHT=100
N_CYCLES=0          # Monotonic warmup (0→β_max over first 10% of steps, then fixed)
TEXT_BACKBONE="clip"
SUITE="libero_object"
MAX_STEPS=2000000
VAL_FREQ=50000
PATIENCE=4
BETA_SCHEDULE="high_to_low"
# BETA_SCHEDULE="warmup"
BETA_HIGH=1.0
WARMUP_RATIO=0.05
GRIPPER_WEIGHT=5.0

# Seconds between each job launch.
# CLIP encoder peaks at ~450 MB during data loading then is freed.
# With 120 s stagger only 1 process loads CLIP at a time → no OOM spike.
STAGGER=120

LOG_DIR="logs/new_protocol_cvae/cond_prior/${SUITE}"
mkdir -p "${LOG_DIR}"

# ── Helper: launch one training job in the background ───────────────────────
launch_job() {
    local LATENT_DIM=$1
    local SEED=$2
    local LOG_FILE=$3
    shift 3
    local EXTRA_FLAGS="$@"

    echo "  ⚡ z=${LATENT_DIM}  seed=${SEED}  →  $(basename ${LOG_FILE})"

    python3 scripts/train_text_conditioned_disentangler.py \
            --suite                ${SUITE}          \
            --use-cond-prior                         \
            --use_state_cond                         \
            --text_backbone        ${TEXT_BACKBONE}  \
            --latent_dim           ${LATENT_DIM}     \
            --chunk_size           ${CHUNK_SIZE}     \
            --dropout              ${DROPOUT}        \
            --recon_weight         ${RECON_WEIGHT}   \
            --gripper_weight       ${GRIPPER_WEIGHT} \
            --beta_schedule        ${BETA_SCHEDULE}  \
            --beta                 ${BETA}           \
            --warmup_ratio         ${WARMUP_RATIO}   \
            --n_cycles             ${N_CYCLES}       \
            --max_steps            ${MAX_STEPS}      \
            --val_freq             ${VAL_FREQ}       \
            --patience             ${PATIENCE}       \
            --seed                 ${SEED}           \
            ${EXTRA_FLAGS}                           \
            > ${LOG_FILE} 2>&1 &
}

# =============================================================================
# LAUNCH ALL 9 JOBS — staggered by STAGGER seconds
# (9 × 120 s = 18 min total launch window, then all train in parallel)
# =============================================================================

JOB=0
LATENT_DIM=128

echo ""
echo "────────────────────────────────────────────────────────────"
echo "📐 LATENT DIM = ${LATENT_DIM}"
echo "────────────────────────────────────────────────────────────"

# # ── A1: CondPrior + SupCon + use_state ───────────────────────────────
# echo "▶  A1: SupCon + use_state"
# for SEED in 1; do
#     JOB=$((JOB + 1))
#     LOG="${LOG_DIR}/beta${BETA}_dr${DROPOUT}_z${LATENT_DIM}_rw${RECON_WEIGHT}_seed${SEED}_supcon_state.log"
#     launch_job "${LATENT_DIM}" "${SEED}" "${LOG}" "--supcon_weight 1.0"
#     [ ${JOB} -lt 9 ] && echo "    ⏳ stagger ${STAGGER}s…" && sleep ${STAGGER}
# done

# # ── A2: CondPrior + use_state only ────────────────────────────────────
# echo "▶  A2: use_state only"
# for SEED in 1; do
#     JOB=$((JOB + 1))
#     LOG="${LOG_DIR}/beta${BETA}_dr${DROPOUT}_z${LATENT_DIM}_rw${RECON_WEIGHT}_seed${SEED}_state_only.log"
#     launch_job "${LATENT_DIM}" "${SEED}" "${LOG}" ""
#     [ ${JOB} -lt 9 ] && echo "    ⏳ stagger ${STAGGER}s…" && sleep ${STAGGER}
# done

# ── A3: CondPrior + SPIRL + use_state ─────────────────────────────────
echo "▶  A3: SPIRL + use_state"
for BETA_HIGH in 1.0 0.5 0.1; do
    for SEED in 1 2 3; do
        JOB=$((JOB + 1))
        LOG="${LOG_DIR}/beta${BETA}-${BETA_HIGH}_dr${DROPOUT}_z${LATENT_DIM}_rw${RECON_WEIGHT}_seed${SEED}_spirl_state.log"
        launch_job "${LATENT_DIM}" "${SEED}" "${LOG}" "--no_text_decoder --beta_high ${BETA_HIGH}"
        [ ${JOB} -lt 9 ] && echo "    ⏳ stagger ${STAGGER}s…" && sleep ${STAGGER}
    done
done

echo ""
echo "🎬 All 9 jobs launched. Training in parallel (~6.5 GB VRAM)."
echo "   Monitor: tail -f ${LOG_DIR}/*_seed1_supcon_state.log"
echo "   GPU:     watch -n30 nvidia-smi"
echo ""
echo "⏳ Waiting for all jobs to finish…"
wait

echo ""
echo "=================================================="
echo "✅ VAE CONVERGENCE SWEEP COMPLETED!"
echo "=================================================="
