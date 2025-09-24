#!/bin/bash

# MedSigLIP + Other Models Ensemble Script
echo "🔗 MEDSIGLIP ENSEMBLE COMBINATION"
echo "=================================="
echo "🎯 Combining MedSigLIP-448 with other trained models"
echo "📊 Dataset: EyePACS (5-class DR classification)"
echo "🏗️ Ensemble: MedSigLIP-448 + EfficientNets/DenseNet/ResNet"
echo ""

# Check if MedSigLIP model exists
if [ ! -f "./medsiglip_results/models/best_medsiglip_448.pth" ]; then
    echo "❌ MedSigLIP model not found! Please run ./train_medsiglip.sh first"
    exit 1
fi

echo "✅ Found MedSigLIP-448 model: ./medsiglip_results/models/best_medsiglip_448.pth"

# Create ensemble directory
mkdir -p ./ensemble_models
mkdir -p ./ensemble_results

# Copy MedSigLIP model to ensemble directory
cp ./medsiglip_results/models/best_medsiglip_448.pth ./ensemble_models/
echo "✅ Copied MedSigLIP model to ensemble directory"

echo ""
echo "🚀 ENSEMBLE TRAINING OPTIONS:"
echo ""

echo "1️⃣  MedSigLIP + EfficientNet B3/B4:"
echo "python super_ensemble_direct_trainer.py \\"
echo "    --dataset_path ./dataset_eyepacs \\"
echo "    --output_dir ./ensemble_results \\"
echo "    --experiment_name 'medsiglip_efficientnet_ensemble' \\"
echo "    --models medsiglip_448 efficientnet_b3 efficientnet_b4 \\"
echo "    --epochs 30 \\"
echo "    --batch_size 6 \\"
echo "    --enable_memory_optimization"
echo ""

echo "2️⃣  MedSigLIP + Full Super Ensemble (4 models):"
echo "python super_ensemble_direct_trainer.py \\"
echo "    --dataset_path ./dataset_eyepacs \\"
echo "    --output_dir ./ensemble_results \\"
echo "    --experiment_name 'medsiglip_full_super_ensemble' \\"
echo "    --models medsiglip_448 efficientnet_b3 efficientnet_b4 efficientnet_b5 \\"
echo "    --epochs 25 \\"
echo "    --batch_size 4 \\"
echo "    --enable_memory_optimization"
echo ""

echo "3️⃣  Quick Ensemble Test (Debug Mode):"
echo "python super_ensemble_direct_trainer.py \\"
echo "    --dataset_path ./dataset_eyepacs \\"
echo "    --output_dir ./ensemble_test \\"
echo "    --experiment_name 'medsiglip_ensemble_test' \\"
echo "    --models medsiglip_448 efficientnet_b3 \\"
echo "    --debug_mode \\"
echo "    --epochs 3"
echo ""

echo "📊 ANALYSIS AFTER TRAINING:"
echo "# Analyze ensemble results"
echo "python analyze_ovo_with_metrics.py --dataset_path ./ensemble_results"
echo ""
echo "# Compare individual model performance"
echo "python model_analyzer.py --model ./ensemble_results/models/best_medsiglip_448.pth"
echo "python model_analyzer.py --model ./ensemble_results/models/best_efficientnet_b3.pth"
echo ""

echo "🎯 EXPECTED ENSEMBLE PERFORMANCE:"
echo "  🏥 MedSigLIP-448: 92-95% accuracy (medical foundation model)"
echo "  ⚡ EfficientNet-B3: 88-92% accuracy (efficient baseline)"
echo "  🚀 EfficientNet-B4: 90-94% accuracy (optimal balance)"
echo "  💪 Combined Ensemble: 95-97% accuracy (medical-grade target)"
echo ""

echo "✅ Ready for ensemble training! Choose option 1, 2, or 3 above."