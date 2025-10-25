#!/bin/bash

# Evaluate SEResNeXt50 OVO Ensemble on Test Set
echo "🧪 Evaluating SEResNeXt50 OVO Ensemble on Test Set"
echo "===================================================="

python3 ensemble_local_trainer.py \
    --mode evaluate \
    --dataset_path ./dataset_eyepacs_5class_balanced_enhanced_v2 \
    --output_dir ./seresnext50_5class_results \
    --img_size 224 \
    --batch_size 2 \
    --base_models seresnext50_32x4d \
    --num_classes 5 \
    --experiment_name 5class_seresnext50_32x4d_evaluation \
    --seed 42 \
    --freeze_weights false 2>&1 | tee evaluation_log.txt