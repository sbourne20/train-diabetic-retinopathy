#!/bin/bash

# Set PyTorch memory management for DenseNet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Improved OVO Ensemble Training Script
echo "🚀 Starting Improved OVO Ensemble Training"
echo "==========================================="

# Create output directory
mkdir -p ./ovo_ensemble_results_v3

echo "📊 Training improved OVO ensemble with BALANCED overfitting prevention:"
echo "  - Memory optimization enabled (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)"
echo "  - Progress bars for each epoch (visual tracking)"
echo "  - Balanced overfitting prevention (15% critical stop)"
echo "  - Dynamic dropout adjustment"
echo "  - Gradient clipping for stability"
echo "  - Enhanced learning rate scheduling"
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
    --learning_rate 1e-4 \
    --weight_decay 5e-3 \
    --enhanced_dropout 0.6 \
    --gradient_clipping 1.0 \
    --overfitting_threshold 0.12 \
    --early_stopping_patience 7 \
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
echo "🔍 Key improvements:"
echo "  ✅ Progress bars for each epoch"
echo "  ✅ Balanced overfitting prevention (15% critical stop)"
echo "  ✅ Dynamic dropout adjustment"
echo "  ✅ Enhanced regularization"
echo "  ✅ Better validation accuracy expected: 75-85%"