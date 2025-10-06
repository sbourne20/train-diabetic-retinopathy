#!/bin/bash

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# EyePACS Balanced + EfficientNetB2 - NO WARMUP VERSION
echo "🏥 EyePACS BALANCED + EfficientNetB2 - NO WARMUP/NO SCHEDULER"
echo "=========================================================="
echo "🎯 Target: 96%+ accuracy"
echo "📊 Dataset: EyePACS Balanced (40,001 training images)"
echo "🏗️ Model: EfficientNetB2 (9M params)"
echo "🔧 FIX: No scheduler, no warmup, constant LR for guaranteed learning"
echo ""

# Create output directory
mkdir -p ./efficientnetb2_eyepacs_no_warmup_results

echo "🔬 NO WARMUP Configuration:"
echo "  - Dataset: EyePACS Balanced - 40,001 samples"
echo "  - Model: EfficientNetB2"
echo "  - Image size: 224x224"
echo "  - Batch size: 32"
echo "  - Learning rate: 3e-4 (CONSTANT - no decay)"
echo "  - Scheduler: NONE (disabled to avoid warmup)"
echo "  - Warmup: 0 (completely disabled)"
echo "  - Epochs: 100"
echo "  - Expected: IMMEDIATE improvement from epoch 1"
echo ""

# Train with NO scheduler to avoid warmup issues
python ensemble_local_trainer.py \
    --mode train \
    --dataset_path ./dataset_eyepacs_ori_balanced_smote \
    --output_dir ./efficientnetb2_eyepacs_no_warmup_results \
    --experiment_name "eyepacs_efficientnetb2_NO_WARMUP" \
    --base_models efficientnetb2 \
    --img_size 224 \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 3e-4 \
    --weight_decay 1e-4 \
    --ovo_dropout 0.3 \
    --freeze_weights false \
    --enable_medical_augmentation \
    --rotation_range 15.0 \
    --brightness_range 0.1 \
    --contrast_range 0.1 \
    --enable_focal_loss \
    --focal_loss_gamma 2.0 \
    --scheduler none \
    --warmup_epochs 0 \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --patience 15 \
    --early_stopping_patience 12 \
    --target_accuracy 0.96 \
    --max_grad_norm 1.0 \
    --label_smoothing 0.1 \
    --seed 42

echo ""
echo "✅ Training completed!"
echo "📁 Results saved to: ./efficientnetb2_eyepacs_no_warmup_results"
echo ""
echo "📊 With NO scheduler/warmup you should see:"
echo "  Epoch 1: LR = 3.0e-04 (constant)"
echo "  Epoch 1: Val Acc = 75-78%"
echo "  Epoch 10: Val Acc = 88-91%"
echo "  Epoch 30: Val Acc = 94-96%"
echo ""
