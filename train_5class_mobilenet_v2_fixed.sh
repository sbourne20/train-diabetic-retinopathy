#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# FIXED: Anti-Overfitting Configuration for MobileNet v2
echo "🏥 5-CLASS MobileNetV2 v2 - ANTI-OVERFITTING FIX"
echo "===================================================================="
echo "🎯 Previous issue: 98.78% train vs 89.25% val (9.5% gap - SEVERE)"
echo "🎯 Target: <5% train-val gap with 90-92% validation accuracy"
echo ""

mkdir -p ./mobilenet_5class_v2_fixed_results

echo "🔧 OVERFITTING FIXES APPLIED:"
echo "  1. Learning rate: 5e-4 → 3e-4 (40% reduction - less aggressive)"
echo "  2. Dropout: 0.4 → 0.5 (+25% - more regularization)"
echo "  3. Weight decay: 2e-4 → 4e-4 (+100% - stronger L2 penalty)"
echo "  4. Label smoothing: 0.10 → 0.15 (+50% - reduce overconfidence)"
echo "  5. Early stopping: 17 → 12 epochs (stop faster when overfitting)"
echo "  6. Max grad norm: 1.0 → 0.5 (prevent gradient explosion)"
echo ""

python3 ensemble_5class_trainer.py \
    --mode train \
    --dataset_path ./dataset_eyepacs_5class_balanced \
    --output_dir ./mobilenet_5class_v2_fixed_results \
    --experiment_name "5class_mobilenet_v2_antioverfitting" \
    --base_models mobilenet_v2 \
    --num_classes 5 \
    --img_size 384 \
    --batch_size 16 \
    --epochs 100 \
    --learning_rate 3e-4 \
    --weight_decay 4e-4 \
    --ovo_dropout 0.5 \
    --freeze_weights false \
    --enable_clahe \
    --clahe_clip_limit 2.5 \
    --enable_medical_augmentation \
    --rotation_range 25.0 \
    --brightness_range 0.20 \
    --contrast_range 0.20 \
    --enable_focal_loss \
    --enable_class_weights \
    --class_weight_0 1.0 \
    --class_weight_1 1.0 \
    --class_weight_2 1.0 \
    --class_weight_3 1.0 \
    --class_weight_4 1.0 \
    --focal_loss_alpha 2.5 \
    --focal_loss_gamma 3.0 \
    --label_smoothing 0.15 \
    --scheduler cosine \
    --warmup_epochs 10 \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --patience 22 \
    --early_stopping_patience 12 \
    --target_accuracy 0.92 \
    --max_grad_norm 0.5 \
    --seed 42 \
    --device cuda \
    --no_wandb

echo ""
echo "✅ Training completed!"
echo ""
echo "📊 EXPECTED IMPROVEMENTS:"
echo "  Target train-val gap: <5% (was 9.5%)"
echo "  Target validation: 90-92% (was ~89%)"
echo "  Indication of success: Train and val curves closer together"
echo ""
