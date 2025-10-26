#!/bin/bash

# Evaluate CoAtNet-0 OVO Ensemble on Test Set (FIXED VOTING)
echo "🧪 Evaluating CoAtNet-0 OVO Ensemble on Test Set (FIXED VOTING)"
echo "================================================================"

python3 ensemble_5class_trainer.py \
    --mode evaluate \
    --base_models coatnet_0_rw_224 \
    --dataset_path ./dataset_eyepacs_5class_balanced_enhanced_v2 \
    --num_classes 5 \
    --output_dir ./coatnet_5class_results \
    --experiment_name 5class_coatnet_0_fixed 2>&1 | tee evaluation_log_coatnet_fixed.txt