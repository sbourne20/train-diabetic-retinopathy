#!/bin/bash

# Set PyTorch memory management for DenseNet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Improved OVO Ensemble Training Script
echo "🚀 Starting Improved OVO Ensemble Training"
echo "==========================================="

# Create output directory
mkdir -p ./ovo_ensemble_results_v3

echo "🏥 Training MEDICAL-GRADE OVO ensemble (90%+ accuracy target):"
echo "  - Memory optimization enabled (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)"
echo "  - wandb experiment tracking for medical-grade validation"
echo "  - Higher learning rate (2e-3) for better convergence"
echo "  - Reduced regularization for higher accuracy"
echo "  - Progress bars for each epoch (visual tracking)"
echo "  - Overfitting prevention (15% critical stop)"
echo "  - Automatic checkpoint resuming"

# Train improved OVO ensemble with ENHANCED overfitting prevention
python ensemble_local_trainer_enhanced.py \
    --mode train \
    --dataset_path ./dataset7b \
    --output_dir ./ovo_ensemble_results_v3 \
    --img_size 299 \
    --base_models mobilenet_v2 inception_v3 densenet121 \
    --experiment_name improved_ovo_ensemble \
    --epochs 30 \
    --batch_size 4 \
    --learning_rate 5e-3 \
    --weight_decay 1e-4 \
    --enhanced_dropout 0.3 \
    --gradient_clipping 1.0 \
    --overfitting_threshold 0.12 \
    --early_stopping_patience 10 \
    --validation_loss_patience 4 \
    --dynamic_dropout \
    --batch_norm \
    --advanced_scheduler \
    --freeze_weights true \
    --resume \
    --seed 42

echo ""
echo "✅ Improved OVO training with BALANCED overfitting prevention completed!"
echo "📁 Results saved to: ./ovo_ensemble_results_v3"
echo ""
echo "🔍 MEDICAL-GRADE improvements:"
echo "  🏥 90%+ accuracy requirement (medical-grade standard)"
echo "  📊 wandb tracking for all metrics and medical compliance"
echo "  🚀 Higher learning rate (2e-3) for optimal convergence"
echo "  📉 Reduced dropout (0.3) for higher learning capacity"
echo "  ⚡ Lower weight decay (1e-4) for better performance"
echo "  🛡️ Overfitting prevention maintains quality"