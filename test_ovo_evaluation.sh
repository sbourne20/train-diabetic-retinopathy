#!/bin/bash

# Test OVO Ensemble Evaluation with Fixed Checkpoint Loading
echo "🧪 Testing OVO Ensemble Evaluation Fix"
echo "======================================"

python ensemble_local_trainer.py \
    --mode evaluate \
    --dataset_path ./dataset6 \
    --output_dir ./ovo_ensemble_results \
    --img_size 299 \
    --base_models mobilenet_v2 inception_v3 densenet121 \
    --experiment_name ovo_evaluation_test \
    --seed 42