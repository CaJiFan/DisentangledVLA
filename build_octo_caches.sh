#!/bin/bash
# =============================================================================
# build_octo_caches.sh
# Rebuilds all Octo embedding caches for every (suite × vae_type) combination.
# Must be run inside the container where JAX / Octo is installed.
#
# Usage:
#   bash build_octo_caches.sh
#
# Logs land in logs/octo_cache/
# =============================================================================

set -e
mkdir -p logs/octo_cache

OCTO_MODEL="hf://rail-berkeley/octo-small-1.5"
OUT_DIR="./checkpoints/projectors/octo"
SUITES=("libero_spatial" "libero_object" "libero_goal")

# --- TCN (text_cond_beta_tcvae, z=128, seed=2) ---
TCN_BETA=0.001
TCN_Z=128
TCN_SEED=2
TCN_VAE_TYPE="text_cond_beta_tcvae"

# --- CVAE (text_cvae, z=64, seed=1) ---
CVAE_BETA=0.001
CVAE_Z=64
CVAE_SEED=1
CVAE_VAE_TYPE="text_cvae"

for SUITE in "${SUITES[@]}"; do
    echo "============================================================"
    echo "  Suite: ${SUITE}"
    echo "============================================================"

    # ── TCN ──────────────────────────────────────────────────────────
    # Run TCN first. build_octo_cache.py saves train_actions (normalised) in the
    # cache. The CVAE run will find it via the fallback scanner and only re-teach
    # teacher targets (~1 min) instead of re-running the full Octo embedding pass.
    echo "[TCN] Building cache for ${SUITE}..."
    python3 scripts/build_octo_cache.py \
        --suite        "${SUITE}" \
        --octo_model   "${OCTO_MODEL}" \
        --vae_type     "${TCN_VAE_TYPE}" \
        --beta         "${TCN_BETA}" \
        --z_dim        "${TCN_Z}" \
        --vae_seed     "${TCN_SEED}" \
        --out_dir      "${OUT_DIR}" \
        2>&1 | tee logs/octo_cache/${SUITE}_tcn.log
    echo "[TCN] Done: ${SUITE}"

    # ── CVAE ──────────────────────────────────────────────────────────
    # Reuse embeddings+actions from the TCN cache above — skips the slow JAX pass.
    TCN_CACHE="${OUT_DIR}/${SUITE}/vla_emb_cache_${TCN_VAE_TYPE}_arch_tcn_beta${TCN_BETA}_z${TCN_Z}.pt"
    echo "[CVAE] Building cache for ${SUITE} (reusing embeddings from TCN cache)..."
    python3 scripts/build_octo_cache.py \
        --suite        "${SUITE}" \
        --octo_model   "${OCTO_MODEL}" \
        --vae_type     "${CVAE_VAE_TYPE}" \
        --beta         "${CVAE_BETA}" \
        --z_dim        "${CVAE_Z}" \
        --vae_seed     "${CVAE_SEED}" \
        --out_dir      "${OUT_DIR}" \
        --emb_cache_from "${TCN_CACHE}" \
        2>&1 | tee logs/octo_cache/${SUITE}_cvae.log
    echo "[CVAE] Done: ${SUITE}"

done

echo ""
echo "============================================================"
echo "  All Octo caches rebuilt. Run tests/test_caches.py next."
echo "============================================================"
