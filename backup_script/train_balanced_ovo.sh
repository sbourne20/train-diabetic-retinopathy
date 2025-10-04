#!/bin/bash

# Balanced OVO Ensemble Training Script with Optimized Overfitting Prevention
echo "⚖️ Starting BALANCED OVO Ensemble Training with Optimized Overfitting Prevention"
echo "================================================================================"

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create output directory
mkdir -p ./ovo_ensemble_results_balanced
mkdir -p ./ovo_ensemble_results_balanced/models
mkdir -p ./ovo_ensemble_results_balanced/logs
mkdir -p ./ovo_ensemble_results_balanced/results

echo "⚖️ BALANCED overfitting prevention features:"
echo "  ✅ Progress bars for each epoch (visual training tracking)"
echo "  ✅ Advanced early stopping with validation loss monitoring"
echo "  ✅ Dynamic dropout adjustment (0.6 → 0.8 when needed)"
echo "  ✅ Gradient clipping (threshold: 1.0)"
echo "  ✅ Overfitting detection threshold: 12% train-val gap (BALANCED)"
echo "  ✅ CRITICAL overfitting stop: ≥15% train-val gap (BALANCED)"
echo "  ✅ Enhanced learning rate scheduling"
echo "  ✅ Moderate weight decay (5e-3) for balanced regularization"
echo "  ✅ All 3 base models: MobileNet-v2, Inception-v3, DenseNet121"
echo ""

# Train BALANCED OVO ensemble with optimized overfitting prevention
python ensemble_local_trainer_enhanced.py \
    --mode train \
    --dataset_path ./dataset7b \
    --output_dir ./ovo_ensemble_results_balanced \
    --img_size 299 \
    --base_models mobilenet_v2 inception_v3 densenet121 \
    --experiment_name balanced_ovo_with_optimized_prevention \
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
echo "✅ Balanced OVO training with optimized overfitting prevention completed!"
echo "📁 Results saved to: ./ovo_ensemble_results_balanced"
echo ""
echo "🔍 To analyze results:"
echo "python analyze_ovo_with_metrics.py"
echo ""
echo "📊 BALANCED approach features:"
echo "  ⚖️ CRITICAL overfitting detection (≥15% gap = stop) - BALANCED"
echo "  ⚖️ Warning at 10% gap (gives model room to learn)"
echo "  ⚖️ Progress bars for easy tracking"
echo "  ⚖️ Moderate regularization (not too aggressive)"
echo "  ⚖️ All 3 base models will train properly"
echo "  ⚖️ Better validation accuracy target: 75-85%"