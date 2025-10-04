#!/bin/bash

# Enhanced OVO Ensemble Training Script with Advanced Overfitting Prevention
echo "🛡️ Starting Enhanced OVO Ensemble Training with Advanced Overfitting Prevention"
echo "==============================================================================="

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create output directory
mkdir -p ./ovo_ensemble_results_enhanced

echo "🛡️ Enhanced overfitting prevention features:"
echo "  - Advanced early stopping with validation loss monitoring"
echo "  - Dynamic dropout adjustment (0.7 → 0.8 when overfitting detected)"
echo "  - Gradient clipping (threshold: 1.0)"
echo "  - Overfitting detection threshold: 15% train-val gap"
echo "  - Enhanced learning rate scheduling"
echo "  - Batch normalization in classifier heads"
echo "  - Higher weight decay (1e-2) for better regularization"
echo "  - Automatic model restoration to best weights"

# Train enhanced OVO ensemble with maximum overfitting prevention
python ensemble_local_trainer_enhanced.py \
    --mode train \
    --dataset_path ./dataset7b \
    --output_dir ./ovo_ensemble_results_enhanced \
    --img_size 299 \
    --base_models mobilenet_v2 \
    --experiment_name enhanced_ovo_with_overfitting_prevention \
    --epochs 30 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --enhanced_dropout 0.7 \
    --gradient_clipping 1.0 \
    --overfitting_threshold 0.15 \
    --early_stopping_patience 5 \
    --validation_loss_patience 3 \
    --dynamic_dropout \
    --batch_norm \
    --advanced_scheduler \
    --freeze_weights true \
    --resume \
    --seed 42

echo ""
echo "✅ Enhanced OVO training with overfitting prevention completed!"
echo "📁 Results saved to: ./ovo_ensemble_results_enhanced"
echo ""
echo "🔍 To analyze results:"
echo "python analyze_ovo_with_metrics.py"
echo ""
echo "📊 Key improvements implemented:"
echo "  ✅ Severe overfitting detection (>25% gap = immediate stop)"
echo "  ✅ Dynamic dropout adjustment based on train-val gap"
echo "  ✅ Advanced early stopping with validation loss monitoring"
echo "  ✅ Gradient clipping for training stability"
echo "  ✅ Enhanced learning rate scheduling"
echo "  ✅ Automatic best model weight restoration"
echo "  ✅ Comprehensive overfitting metrics logging"