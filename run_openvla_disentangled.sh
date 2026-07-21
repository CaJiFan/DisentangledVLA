#!/bin/bash

# 1. LOAD YOUR ENVIRONMENT
# Update this line to however you activate your virtual environment locally
source /opt/venv/bin/activate 

# 2. APPLY LOCAL STABILITY FIXES
# These are still 100% required on a local machine to prevent Python deadlocks!
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLE_SERVICE=True
export WANDB_START_METHOD=thread

# 3. DEFINE THE EVALUATION MATRIX
SUITES=("libero_10" "libero_goal"  "libero_spatial")
LOSSES=("mse" "nll" "w2")
BETA=0.1
DROPOUT=0.15
Z_DIM=64
CHUNK_SIZE=8
ALPHA=1.0
STEP=80000
SEED=1
RECON_WEIGHT=100
SUITE="libero_spatial"


# Update this to your local checkpoints folder
CHECKPOINT_DIR="/workspace/DisentangledFlow/checkpoints"

echo "=================================================="
echo "🚀 STARTING LOCAL EVALUATION PIPELINE"
echo "=================================================="

# 4. THE SEQUENTIAL LOOP
for CURRENT_LOSS in "nll" "w2"
do
    TCVAE_PATH="${CHECKPOINT_DIR}/text_tcvae/${SUITE}/recon${RECON_WEIGHT}_dropout0.15_beta${BETA}_z${Z_DIM}_alpha${ALPHA}_chunk${CHUNK_SIZE}_std_seed_${SEED}_step_${STEP}.pt"
    
    # Set paths based on the current loss iteration
    if [ "$CURRENT_LOSS" == "nll" ]; then
        PROJ_PATH="${CHECKPOINT_DIR}/projectors/${SUITE}/chunk_${CHUNK_SIZE}_zdim_${Z_DIM}/prob_loss_nll_step_25000.pt"
    elif [ "$CURRENT_LOSS" == "mse" ]; then
        PROJ_PATH="${CHECKPOINT_DIR}/projectors/${SUITE}/chunk_${CHUNK_SIZE}_zdim_${Z_DIM}/mlp_loss_mse_step_25000.pt"
    else
        PROJ_PATH="${CHECKPOINT_DIR}/projectors/${SUITE}/chunk_${CHUNK_SIZE}_zdim_${Z_DIM}/prob_loss_w2_step_25000.pt"
    fi

    echo ""
    echo "--------------------------------------------------"
    echo "▶ RUNNING: Suite: $CURRENT_SUITE | Loss: $CURRENT_LOSS (z=$Z_DIM)"
    echo "--------------------------------------------------"

    # 5. RUN THE PYTHON SCRIPT
    python scripts/evaluate_projector.py \
        --loss_type ${CURRENT_LOSS} \
        --z_dim ${Z_DIM} \
        --projector_weights ${PROJ_PATH} \
        --tcvae_weights ${TCVAE_PATH}
    
    # 6. MEMORY SAFETY CATCH
    # Wait a few seconds between runs to ensure the OS completely 
    # clears the 24GB GPU VRAM before starting the next heavy model load.
    sleep 5 

done

echo ""
echo "=================================================="
echo "✅ ALL EVALUATIONS COMPLETED SUCCESSFULLY"
echo "=================================================="


