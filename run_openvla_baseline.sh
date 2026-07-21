#!/bin/bash

export TOKENIZERS_PARALLELISM=false

for CURRENT_SUITE in "libero_spatial" "libero_goal" "libero_10" 
do
    echo "🚀 Starting evaluation for ${CURRENT_SUITE}..."

    # python3 scripts/evaluate_openvla_baseline.py --task_suite ${CURRENT_SUITE} > "logs/${CURRENT_SUITE}_evaluation.log" 2>&1
    # python3 vlas/openvla_oft/experiments/robot/libero/run_libero_eval.py --task_suite_name ${CURRENT_SUITE} > "logs/${CURRENT_SUITE}_evaluation.log" 2>&1
    python3 vlas/openvla/experiments/robot/libero/run_libero_eval.py --task_suite_name ${CURRENT_SUITE} > "logs/openvla_${CURRENT_SUITE}_evaluation.log" 2>&1

    echo "✅ Evaluation for ${CURRENT_SUITE} completed successfully!"
done
