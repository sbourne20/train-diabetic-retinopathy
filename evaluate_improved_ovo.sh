#!/bin/bash

# Improved OVO Ensemble Evaluation Script
echo "🧪 Testing Improved OVO Ensemble Evaluation"
echo "==========================================="

# Check what models exist
echo "📁 Checking improved trained models:"
ls -la ./ovo_ensemble_results_v2/models/ | head -10

echo ""
echo "🔍 Checking for improved ensemble file:"
ls -la ./ovo_ensemble_results_v2/models/ovo_ensemble_best.pth

echo ""
echo "🚀 Running improved evaluation:"
python ensemble_local_trainer.py \
    --mode evaluate \
    --dataset_path ./dataset6 \
    --output_dir ./ovo_ensemble_results_v2 \
    --img_size 299 \
    --base_models mobilenet_v2 inception_v3 densenet121 \
    --experiment_name improved_ovo_evaluation \
    --freeze_weights false \
    --ovo_dropout 0.3 \
    --seed 42

echo ""
echo "📊 Improved evaluation completed!"