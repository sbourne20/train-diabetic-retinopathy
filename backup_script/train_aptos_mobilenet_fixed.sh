#!/bin/bash

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# APTOS 2019 + MobileNet Anti-Overfitting Training Script
echo "🏥 APTOS 2019 + MobileNet Anti-Overfitting Training"
echo "================================================="
echo "🎯 Target: Reduce overfitting while maintaining accuracy"
echo "📊 Dataset: APTOS 2019 (5-class DR classification)"
echo "🏗️ Model: MobileNet with strong regularization"
echo "🔬 Enhanced regularization and data augmentation"
echo ""

# Create output directory for APTOS results
mkdir -p ./aptos_results_fixed

echo "🔬 APTOS 2019 Anti-Overfitting Configuration:"
echo "  - Dataset: APTOS 2019 (./dataset_aptos)"
echo "  - Model: MobileNet with dropout regularization"
echo "  - Image size: 224x224"
echo "  - Batch size: 16 (reduced for better regularization)"
echo "  - Learning rate: 5e-4 (reduced for stability)"
echo "  - Weight decay: 1e-3 (increased regularization)"
echo "  - Early stopping: patience 5 (aggressive stopping)"
echo "  - Enhanced augmentation: Strong anti-overfitting"
echo "  - Dropout: 0.5 (strong regularization)"
echo ""

# Train APTOS with anti-overfitting hyperparameters
python ensemble_local_trainer.py \
    --mode train \
    --dataset_path ./dataset_aptos \
    --output_dir ./aptos_results_fixed \
    --experiment_name "aptos_2019_mobilenet_antioverfitting" \
    --base_models mobilenet_v2 \
    --img_size 224 \
    --batch_size 16 \
    --epochs 50 \
    --learning_rate 5e-4 \
    --weight_decay 1e-3 \
    --ovo_dropout 0.5 \
    --enable_medical_augmentation \
    --rotation_range 25.0 \
    --brightness_range 0.2 \
    --contrast_range 0.2 \
    --enable_focal_loss \
    --enable_class_weights \
    --scheduler cosine \
    --warmup_epochs 5 \
    --validation_frequency 1 \
    --checkpoint_frequency 3 \
    --patience 5 \
    --target_accuracy 0.90 \
    --early_stopping_patience 5 \
    --seed 42

echo ""
echo "✅ APTOS 2019 anti-overfitting training completed!"
echo "📁 Results saved to: ./aptos_results_fixed"
echo ""
echo "🎯 Anti-Overfitting Improvements:"
echo "  📊 Reduced batch size: 32→16 (better regularization)"
echo "  🎓 Increased weight decay: 1e-4→1e-3 (stronger regularization)"
echo "  🛑 Aggressive early stopping: patience 15→5"
echo "  🔀 Enhanced augmentation: rotation 15°→25°, brightness/contrast 10%→20%"
echo "  💧 Added dropout: 0.5 (prevent memorization)"
echo "  📉 Reduced learning rate: 1e-3→5e-4 (more stable)"
echo ""
echo "📋 Expected Improvements:"
echo "  1. Training-validation gap should be <5%"
echo "  2. Better generalization to new patients"
echo "  3. More stable training curves"
echo "  4. Medical-grade reliability"