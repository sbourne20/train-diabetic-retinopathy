#!/bin/bash

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# APTOS + MobileNetV2 Specialized Training Script
echo "🏥 APTOS + MobileNetV2 Specialized Training"
echo "=========================================="
echo "🎯 Target: Break through (0,2) 77% ceiling with APTOS dataset"
echo "📊 Dataset: APTOS (competition-grade automated classification)"
echo "🏗️ Model: MobileNetV2 (lightweight, efficient)"
echo "🎯 Focus: No DR (0) vs Moderate NPDR (2) classification"
echo ""

# Create specialized output directory
mkdir -p ./ovo_aptos_mobilenet_results

echo "🔬 APTOS-MobileNetV2 Configuration:"
echo "  - Dataset: APTOS competition data (optimized for automation)"
echo "  - Model: MobileNetV2 (proven 98.4% on other pairs)"
echo "  - Image size: 224x224 (optimal for MobileNetV2)"
echo "  - Batch size: 32 (proven effective)"
echo "  - Learning rate: 5e-4 (stable training)"
echo "  - Target pair: (0,2) - No DR vs Moderate NPDR"
echo "  - Expected improvement: 77% → 85%+"
echo ""

# Train APTOS-specialized MobileNetV2
python ensemble_local_trainer.py \
    --mode train \
    --dataset_path ./aptos \
    --output_dir ./ovo_aptos_mobilenet_results \
    --img_size 224 \
    --base_models mobilenet_v2 \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --weight_decay 1e-4 \
    --seed 42

echo ""
echo "✅ APTOS-MobileNetV2 specialized training completed!"
echo "📁 Results saved to: ./ovo_aptos_mobilenet_results"
echo ""
echo "🎯 APTOS Advantages:"
echo "  📊 Competition-grade dataset optimized for automated classification"
echo "  🤖 Better automated distinction between No DR and Moderate NPDR"
echo "  📈 Expected to break through 77% ceiling for (0,2) pair"
echo "  🏆 APTOS winner solutions achieved high accuracy on similar distinctions"