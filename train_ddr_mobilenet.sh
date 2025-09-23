#!/bin/bash

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# APTOS + MobileNetV2 Medical-Grade Training Script
echo "🏥 APTOS + MobileNetV2 Medical-Grade Training"
echo "============================================="
echo "🎯 Target: 85%+ accuracy with MobileNetV2 architecture"
echo "📊 Dataset: APTOS 2019 (5-class DR classification)"
echo "🏗️ Model: MobileNetV2 (mobile-optimized architecture)"
echo "🔬 Medical-grade architecture with optimized hyperparameters"
echo ""

# Create output directory for DDR results
mkdir -p ./ddr_results

echo "🔬 APTOS MobileNet Medical-Grade Configuration:"
echo "  - Dataset: APTOS 2019 (./dataset_aptos)"
echo "  - Model: MobileNetV2 (mobile-optimized capacity)"
echo "  - Image size: 224x224 (research paper standard)"
echo "  - Batch size: 24 (optimized for MobileNet)"
echo "  - Learning rate: 3e-4 (MobileNet optimized)"
echo "  - Weight decay: 1e-3 (strong regularization)"
echo "  - Dropout: 0.5 (balanced regularization)"
echo "  - Epochs: 60 (sufficient for convergence)"
echo "  - Enhanced augmentation + progressive training"
echo "  - Target: 90%+ accuracy (medical production grade)"
echo ""

# Train DDR with research-validated hyperparameters
python ensemble_local_trainer.py \
    --mode train \
    --dataset_path ./dataset_aptos \
    --output_dir ./ddr_densenet_results \
    --experiment_name "aptos_mobilenet_medical" \
    --base_models mobilenet_v2 \
    --img_size 224 \
    --batch_size 24 \
    --epochs 60 \
    --learning_rate 3e-4 \
    --weight_decay 1e-3 \
    --ovo_dropout 0.5 \
    --enable_medical_augmentation \
    --rotation_range 20.0 \
    --brightness_range 0.15 \
    --contrast_range 0.15 \
    --enable_focal_loss \
    --enable_class_weights \
    --scheduler cosine \
    --warmup_epochs 8 \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --patience 12 \
    --early_stopping_patience 10 \
    --target_accuracy 0.95 \
    --seed 42

echo ""
echo "✅ APTOS MobileNet training completed!"
echo "📁 Results saved to: ./ddr_results"
echo ""
echo "🎯 Medical-Grade Configuration Applied:"
echo "  🏗️ Architecture: MobileNetV2 (mobile-optimized)"
echo "  📊 Model capacity: 3M parameters (efficient)"
echo "  🎓 Optimized learning rate: 3e-4 (MobileNet optimized)"
echo "  💧 Balanced dropout: 0.5 (prevents overfitting)"
echo "  ⏰ Training epochs: 60 (sufficient convergence)"
echo "  🔀 Enhanced augmentation: Medical imaging optimized settings"
echo ""
echo "📊 Expected Performance:"
echo "  🎯 Target: 95%+ validation accuracy (ambitious medical-grade target)"
echo "  🏥 Medical grade: Should achieve FULL PASS (≥90%)"
echo "  📈 Generalization: Better performance on new patients"
echo "  🔬 APTOS dataset: Reliable competition dataset for consistent results"
echo ""
echo "📋 Next Steps:"
echo "  1. Analyze results: python model_analyzer.py --model ./ddr_results/models/best_mobilenet_v2.pth"
echo "  2. Validate medical-grade performance (>90% required)"
echo "  3. Compare with APTOS DenseNet121 (82.51%) for ensemble"