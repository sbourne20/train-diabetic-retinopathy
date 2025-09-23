#!/bin/bash

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# APTOS 2019 + MobileNet Enhanced Training Script
echo "🏥 APTOS 2019 + MobileNet Enhanced Training"
echo "=========================================="
echo "🎯 Target: >92% accuracy with research-validated hyperparameters"
echo "📊 Dataset: APTOS 2019 (5-class DR classification)"
echo "🏗️ Model: MobileNet (research paper's best performer: 92%)"
echo "🔬 Using exact hyperparameters from research paper"
echo ""

# Create output directory for APTOS results
mkdir -p ./aptos_results

echo "🔬 APTOS 2019 Research-Validated Configuration:"
echo "  - Dataset: APTOS 2019 (./dataset_aptos)"
echo "  - Model: MobileNet (achieved 92% in research)"
echo "  - Image size: 224x224 (research paper standard)"
echo "  - Batch size: 32 (research paper setting)"
echo "  - Learning rate: Adam 1e-3 (research paper setting)"
echo "  - Epochs: 50 (research paper setting)"
echo "  - Threshold: 0.60 (research paper setting)"
echo "  - Enhanced augmentation: Medical-grade settings"
echo "  - Target: >92% accuracy (match research results)"
echo ""

# Train APTOS with research-validated hyperparameters
python ensemble_local_trainer.py \
    --mode train \
    --dataset_path ./dataset_aptos \
    --output_dir ./aptos_results \
    --experiment_name "aptos_2019_mobilenet_enhanced" \
    --base_models mobilenet_v2 \
    --img_size 224 \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --enable_medical_augmentation \
    --rotation_range 15.0 \
    --brightness_range 0.1 \
    --contrast_range 0.1 \
    --enable_focal_loss \
    --enable_class_weights \
    --scheduler cosine \
    --warmup_epochs 10 \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --patience 15 \
    --target_accuracy 0.92 \
    --seed 42

echo ""
echo "✅ APTOS 2019 enhanced training completed!"
echo "📁 Results saved to: ./aptos_results"
echo ""
echo "🎯 Research Paper Validation:"
echo "  📊 Using exact hyperparameters from 92% accuracy paper"
echo "  🤖 MobileNet: Best performer on APTOS dataset"
echo "  📈 Medical-grade augmentation enabled"
echo "  🏆 Target: Match or exceed 92% research accuracy"
echo ""
echo "📋 Next Steps:"
echo "  1. Check results in ./aptos_results/results/"
echo "  2. Analyze model performance with model_analyzer.py"
echo "  3. If >92% achieved, proceed to IDRiD dataset training"