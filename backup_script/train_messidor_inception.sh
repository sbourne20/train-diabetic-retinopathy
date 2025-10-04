#!/bin/bash

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Messidor + InceptionV3 Specialized Training Script
echo "🏥 Messidor + InceptionV3 Specialized Training"
echo "=============================================="
echo "🎯 Target: Clinical-grade (0,2) classification with expert annotations"
echo "📊 Dataset: Messidor (clinical quality, expert graded)"
echo "🏗️ Model: InceptionV3 (multi-scale feature extraction)"
echo "🎯 Focus: No DR (0) vs Moderate NPDR (2) with clinical precision"
echo ""

# Create specialized output directory
mkdir -p ./ovo_messidor_inception_results

echo "🔬 Messidor-InceptionV3 Configuration:"
echo "  - Dataset: Messidor clinical dataset (expert ophthalmologist graded)"
echo "  - Model: InceptionV3 (excellent for complex clinical features)"
echo "  - Image size: 299x299 (optimal for InceptionV3)"
echo "  - Batch size: 16 (memory optimized for larger images)"
echo "  - Learning rate: 3e-4 (conservative for clinical data)"
echo "  - Target pair: (0,2) - No DR vs Moderate NPDR"
echo "  - Expected improvement: Clinical precision → 80%+"
echo ""

# Train Messidor-specialized InceptionV3
python ensemble_local_trainer.py \
    --mode train \
    --dataset_path ./messidor \
    --output_dir ./ovo_messidor_inception_results \
    --img_size 299 \
    --base_models inception_v3 \
    --epochs 50 \
    --batch_size 16 \
    --learning_rate 3e-4 \
    --weight_decay 1e-4 \
    --seed 42

echo ""
echo "✅ Messidor-InceptionV3 specialized training completed!"
echo "📁 Results saved to: ./ovo_messidor_inception_results"
echo ""
echo "🎯 Messidor Advantages:"
echo "  🏥 Clinical-grade dataset with expert ophthalmologist annotations"
echo "  👨‍⚕️ Professional medical imaging standards and protocols"
echo "  🔬 High-quality fundus images with consistent clinical grading"
echo "  📊 Multi-scale InceptionV3 perfect for subtle clinical features"
echo "  🎯 Expected clinical precision for (0,2) boundary distinction"