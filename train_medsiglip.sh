#!/bin/bash

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# EyePACS + MedSigLIP-448 Medical-Grade Training Script
echo "🏥 EyePACS + MedSigLIP-448 Medical-Grade Training"
echo "================================================="
echo "🎯 Target: 90%+ accuracy with MedSigLIP-448 architecture"
echo "📊 Dataset: EyePACS (5-class DR classification)"
echo "🏗️ Model: MedSigLIP-448 (medical foundation model)"
echo "🔬 Medical-grade architecture with optimized hyperparameters"
echo ""

# Create output directory for MedSigLIP results
mkdir -p ./medsiglip_results

echo "🔬 EyePACS MedSigLIP Medical-Grade Configuration:"
echo "  - Dataset: EyePACS (./dataset_eyepacs)"
echo "  - Model: MedSigLIP-448 (medical foundation model)"
echo "  - Image size: 448x448 (MedSigLIP required size)"
echo "  - Batch size: 8 (optimized for MedSigLIP memory)"
echo "  - Learning rate: 5e-5 (foundation model optimized)"
echo "  - Weight decay: 1e-2 (strong regularization)"
echo "  - Dropout: 0.3 (foundation model regularization)"
echo "  - Epochs: 50 (sufficient for convergence)"
echo "  - Enhanced augmentation + progressive training"
echo "  - Target: 95%+ accuracy (medical production grade)"
echo ""

# Train MedSigLIP with research-validated hyperparameters
python super_ensemble_direct_trainer.py \
    --dataset_path ./dataset_eyepacs \
    --output_dir ./medsiglip_results \
    --experiment_name "eyepacs_medsiglip_medical" \
    --models medsiglip_448 \
    --img_size 448 \
    --batch_size 8 \
    --epochs 50 \
    --learning_rate 5e-5 \
    --weight_decay 1e-2 \
    --dropout 0.3 \
    --medsiglip_lr_multiplier 1.0 \
    --enable_medical_augmentation \
    --rotation_range 15.0 \
    --brightness_range 0.1 \
    --contrast_range 0.1 \
    --enable_focal_loss \
    --enable_class_weights \
    --scheduler cosine \
    --warmup_epochs 5 \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --patience 15 \
    --early_stopping_patience 12 \
    --target_accuracy 0.95 \
    --enable_memory_optimization \
    --gradient_accumulation_steps 2 \
    --seed 42

echo ""
echo "✅ EyePACS MedSigLIP training completed!"
echo "📁 Results saved to: ./medsiglip_results"
echo ""
echo "🎯 Medical-Grade Configuration Applied:"
echo "  🏗️ Architecture: MedSigLIP-448 (medical foundation model)"
echo "  📊 Model capacity: 300M+ parameters (large medical model)"
echo "  🎓 Optimized learning rate: 5e-5 (foundation model optimized)"
echo "  💧 Balanced dropout: 0.3 (prevents overfitting)"
echo "  ⏰ Training epochs: 50 (sufficient convergence)"
echo "  🔀 Enhanced augmentation: Medical imaging optimized settings"
echo ""
echo "📊 Expected Performance:"
echo "  🎯 Target: 95%+ validation accuracy (ambitious medical-grade target)"
echo "  🏥 Medical grade: Should achieve FULL PASS (≥90%)"
echo "  📈 Generalization: Better performance on new patients"
echo "  🔬 EyePACS dataset: Large-scale diabetic retinopathy dataset"
echo ""
echo "📋 Next Steps:"
echo "  1. Analyze results: python model_analyzer.py --model ./medsiglip_results/models/best_medsiglip_448.pth"
echo "  2. Validate medical-grade performance (>90% required)"
echo "  3. Compare with other models for ensemble evaluation"