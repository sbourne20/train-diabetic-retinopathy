#!/bin/bash

# Set PyTorch memory management for DenseNet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Improved OVO Ensemble Training Script
echo "🚀 Starting RESEARCH-VALIDATED OVO Ensemble Training"
echo "====================================================="

# Create output directory
mkdir -p ./ovo_ensemble_results_v3

echo "📊 Training with RESEARCH-VALIDATED parameters from paper achieving 96%+ accuracy:"
echo "  - Image size: 224x224 (exactly as in research)"
echo "  - Batch size: 32 (research specification)"
echo "  - Epochs: 50 (research specification)"
echo "  - Learning rate: 1e-3 Adam (research specification)"
echo "  - Frozen weights: true (research transfer learning approach)"
echo "  - Single output node classifiers (research architecture)"
echo "  - Research-validated augmentation: rotation=45°, shear=0.2, zoom=0.2"
echo "  - Memory optimization enabled (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)"

# Train improved OVO ensemble
python ensemble_local_trainer.py \
    --mode train \
    --dataset_path ./dataset7b \
    --output_dir ./ovo_ensemble_results_v3 \
    --img_size 224 \
    --base_models mobilenet_v2 inception_v3 densenet121 \
    --experiment_name research_validated_ovo_ensemble \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --freeze_weights true \
    --ovo_dropout 0.5 \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --resume \
    --seed 42

echo ""
echo "✅ Improved OVO training completed!"
echo "📁 Results saved to: ./ovo_ensemble_results_v3"