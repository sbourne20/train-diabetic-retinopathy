#!/bin/bash

# Improved OVO Ensemble Training Script
echo "🚀 Starting Improved OVO Ensemble Training"
echo "==========================================="

# Create output directory
mkdir -p ./ovo_ensemble_results_v2

echo "📊 Training improved OVO ensemble with:"
echo "  - Fine-tuning enabled (freeze_weights=False)"
echo "  - Enhanced multi-layer classifier heads"
echo "  - Differential learning rates (backbone vs classifier)"
echo "  - Learning rate scheduling"
echo "  - Increased training epochs"
echo "  - Better batch size for stability"

# Train improved OVO ensemble
python ensemble_local_trainer.py \
    --mode train \
    --dataset_path ./dataset6 \
    --output_dir ./ovo_ensemble_results_v2 \
    --img_size 299 \
    --base_models mobilenet_v2 inception_v3 densenet121 \
    --experiment_name improved_ovo_ensemble \
    --epochs 30 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --freeze_weights false \
    --ovo_dropout 0.3 \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --seed 42

echo ""
echo "✅ Improved OVO training completed!"
echo "📁 Results saved to: ./ovo_ensemble_results_v2"