#!/bin/bash

# Server OVO Ensemble Evaluation
echo "🧪 Testing OVO Ensemble Evaluation on Server"
echo "============================================="

# First, check what models exist
echo "📁 Checking trained models:"
ls -la ./ovo_ensemble_results/models/ | head -10

echo ""
echo "🔍 Checking for ensemble file:"
ls -la ./ovo_ensemble_results/models/ovo_ensemble_best.pth

echo ""
echo "🚀 Running evaluation:"
python ensemble_local_trainer.py \
    --mode evaluate \
    --dataset_path ./dataset6 \
    --output_dir ./ovo_ensemble_results \
    --img_size 299 \
    --base_models mobilenet_v2 inception_v3 densenet121 \
    --experiment_name ovo_evaluation_test \
    --seed 42