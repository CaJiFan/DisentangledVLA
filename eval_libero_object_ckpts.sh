#!/bin/bash

export HF_TOKEN="<YOUR_HUGGINGFACE_TOKEN>"


mkdir -p "eval_logs/libero_object"

# Z_DIM=128
GRIPPER_WEIGHT=5.0
BETA=0.1

echo "🚀 Starting parallel evaluations for seed ${SEED} on libero_object..."

for BETA_HIGH in 1.0 0.5 0.1
do
    for Z_DIM in 128
    do 
        for CONFIG in "SPIRL_state" 
        do
            for SEED in 1 2 3
            do
                CKPT="checkpoints/new_protocol_cvae/libero_object/rw100_d0.1_beta${BETA}-${BETA_HIGH}_z${Z_DIM}_chunk8_protA_cond_prior_${CONFIG}_h1_grip${GRIPPER_WEIGHT}_seed_${SEED}_best.pt"
                LOG="eval_logs/libero_object/z${Z_DIM}_beta${BETA}-${BETA_HIGH}_${CONFIG}_seed${SEED}_g${GRIPPER_WEIGHT}_temporal_k0.01.log"
                
                echo "  → Evaluating CONFIG: ${CONFIG} (Log: ${LOG})"
                
                python3 scripts/eval_disentangler_suite.py \
                    --suite libero_object \
                    --checkpoint "${CKPT}" \
                    --rollouts 50 \
                    --temporal_ensemble \
                    --ensemble_k 0.01 \
                    > "${LOG}" 2>&1 &
            done
        done
    done
done

echo "⏳ Waiting for all evaluation jobs to complete..."
wait
echo "✅ All evaluation jobs finished!"