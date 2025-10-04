#!/bin/bash

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# IDRID + DenseNet121 Specialized Training Script
echo "🏥 IDRID + DenseNet121 Specialized Training"
echo "==========================================="
echo "🎯 Target: Research-grade (0,2) classification with detailed annotations"
echo "📊 Dataset: IDRID (research quality, detailed lesion annotations)"
echo "🏗️ Model: DenseNet121 (dense feature connectivity, fine detail detection)"
echo "🎯 Focus: No DR (0) vs Moderate NPDR (2) with lesion-level precision"
echo ""

# Create specialized output directory
mkdir -p ./ovo_idrid_densenet_results

echo "🔬 IDRID-DenseNet121 Configuration:"
echo "  - Dataset: IDRID research dataset (detailed lesion annotations)"
echo "  - Model: DenseNet121 (dense feature reuse, excellent for fine details)"
echo "  - Image size: 224x224 (optimal for DenseNet121)"
echo "  - Batch size: 24 (balanced for DenseNet memory usage)"
echo "  - Learning rate: 4e-4 (moderate for research-grade data)"
echo "  - Target pair: (0,2) - No DR vs Moderate NPDR"
echo "  - Expected improvement: Lesion-aware precision → 82%+"
echo ""

# Train IDRID-specialized DenseNet121
python ensemble_local_trainer.py \
    --mode train \
    --dataset_path ./idrid \
    --output_dir ./ovo_idrid_densenet_results \
    --img_size 224 \
    --base_models densenet121 \
    --epochs 50 \
    --batch_size 24 \
    --learning_rate 4e-4 \
    --weight_decay 1e-4 \
    --seed 42

echo ""
echo "✅ IDRID-DenseNet121 specialized training completed!"
echo "📁 Results saved to: ./ovo_idrid_densenet_results"
echo ""
echo "🎯 IDRID Advantages:"
echo "  🔬 Research-grade dataset with detailed lesion-level annotations"
echo "  📊 Precise microaneurysm and hemorrhage labeling"
echo "  🧬 DenseNet121 dense connectivity perfect for subtle feature detection"
echo "  🔍 Fine-grained analysis of early vs moderate NPDR changes"
echo "  📈 Expected research precision for (0,2) boundary detection"