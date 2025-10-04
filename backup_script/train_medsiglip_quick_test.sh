#!/bin/bash

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Quick test of fixed MedSigLIP configuration (2 epochs only)
echo "🧪 QUICK TEST: Fixed MedSigLIP Configuration (2 epochs)"
echo "====================================================="
echo "🎯 Goal: Test if fix restores 86%+ accuracy from epoch 1-2"
echo "📊 Dataset: EyePACS (5-class DR classification)"
echo "🏗️ Model: MedSigLIP-448 (fixed loading)"
echo "⏱️ Duration: 2 epochs only for quick validation"
echo ""

# Create output directory for quick test
mkdir -p ./medsiglip_quick_test

echo "🔬 Testing MedSigLIP FIXED Configuration:"
echo "  - Dataset: EyePACS (./dataset_eyepacs)"
echo "  - Model: MedSigLIP-448 (FIXED loading with full model)"
echo "  - Image size: 448x448"
echo "  - Batch size: 8 (same as working run)"
echo "  - Learning rate: 1e-4 (same as working run)"
echo "  - Epochs: 2 (quick test)"
echo "  - Expected: Should reach 86%+ by epoch 1-2 if fix works"
echo ""

# Quick test with EXACT working parameters
python ensemble_local_trainer.py \
    --mode train \
    --dataset_path ./dataset_eyepacs \
    --output_dir ./medsiglip_quick_test \
    --experiment_name "quick_test_fixed_medsiglip" \
    --base_models medsiglip_448 \
    --img_size 448 \
    --batch_size 8 \
    --epochs 2 \
    --learning_rate 1e-4 \
    --weight_decay 3e-4 \
    --ovo_dropout 0.2 \
    --freeze_weights false \
    --enable_medical_augmentation \
    --rotation_range 15.0 \
    --brightness_range 0.1 \
    --contrast_range 0.1 \
    --enable_focal_loss \
    --enable_class_weights \
    --class_weight_severe 25.0 \
    --class_weight_pdr 30.0 \
    --focal_loss_alpha 2.5 \
    --focal_loss_gamma 3.5 \
    --scheduler plateau \
    --warmup_epochs 5 \
    --validation_frequency 1 \
    --target_accuracy 0.86 \
    --seed 42

echo ""
echo "✅ Quick test completed!"
echo ""
echo "📊 EXPECTED RESULTS IF FIX WORKS:"
echo "  🎯 Epoch 1: Should show 85-90% accuracy (vs previous 78%)"
echo "  🎯 Epoch 2: Should reach/exceed 86% (matching working baseline)"
echo ""
echo "📋 Analysis:"
echo "  python model_analyzer.py --model ./medsiglip_quick_test/models/best_medsiglip_448_multiclass.pth"