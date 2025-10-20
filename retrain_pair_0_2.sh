#!/bin/bash

# Retrain ONLY pair 0-2 with stronger regularization to fix bottleneck
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "🔧 RETRAINING PAIR 0-2 (No DR vs Moderate NPDR)"
echo "=========================================================="
echo "🎯 Current performance: 86.37% validation (causing 92.24% ensemble accuracy)"
echo "🎯 Target: 95%+ validation (should push ensemble to 95%+)"
echo ""
echo "📊 CHANGES FROM ORIGINAL v4:"
echo "  - Dropout: 0.40 → 0.50 (STRONGER regularization)"
echo "  - Weight decay: 5e-4 → 8e-4 (INCREASED L2 penalty)"
echo "  - Learning rate: 5e-5 → 3e-5 (MORE conservative)"
echo "  - Label smoothing: 0.10 → 0.15 (REDUCE overconfidence)"
echo "  - Focal loss gamma: 3.0 → 4.0 (FOCUS on hard examples)"
echo "  - Augmentation: INCREASED (rotation 30°, brightness/contrast 25%)"
echo "  - Gradient clip: 1.0 → 0.5 (PREVENT gradient explosions)"
echo "  - Patience: 25 → 30 (ALLOW more learning time)"
echo ""

# Delete existing 0-2 checkpoint to force retraining
echo "🗑️  Removing old pair 0-2 checkpoint..."
rm -f ./densenet_5class_v4_enhanced_results/models/best_densenet121_0_2.pth
rm -f ./densenet_5class_v4_enhanced_results/models/densenet121_0_2_*.pth
echo ""

python3 ensemble_5class_trainer.py \
    --mode train \
    --dataset_path ./dataset_eyepacs_5class_balanced_enhanced \
    --output_dir ./densenet_5class_v4_enhanced_results \
    --experiment_name "5class_densenet121_v4_gradespec_pair02_retrain" \
    --base_models densenet121 \
    --num_classes 5 \
    --img_size 384 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --epochs 100 \
    --learning_rate 3e-5 \
    --weight_decay 8e-4 \
    --ovo_dropout 0.50 \
    --freeze_weights false \
    --enable_medical_augmentation \
    --rotation_range 30.0 \
    --brightness_range 0.25 \
    --contrast_range 0.25 \
    --enable_focal_loss \
    --enable_class_weights \
    --class_weight_0 1.0 \
    --class_weight_1 1.0 \
    --class_weight_2 1.0 \
    --class_weight_3 1.0 \
    --class_weight_4 1.0 \
    --focal_loss_alpha 2.5 \
    --focal_loss_gamma 4.0 \
    --scheduler cosine \
    --warmup_epochs 10 \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --patience 30 \
    --early_stopping_patience 25 \
    --target_accuracy 0.95 \
    --max_grad_norm 0.5 \
    --label_smoothing 0.15 \
    --seed 42 \
    --resume

echo ""
echo "✅ Pair 0-2 retraining completed!"
echo "📁 New model saved to: ./densenet_5class_v4_enhanced_results/models/best_densenet121_0_2.pth"
echo ""
echo "🔄 NEXT STEPS:"
echo "1. Re-run ensemble evaluation: python analyze_ovo_with_metrics.py"
echo "2. Check if 0-2 accuracy improved to 95%+"
echo "3. Verify ensemble accuracy improved from 92.24% to 95%+"
