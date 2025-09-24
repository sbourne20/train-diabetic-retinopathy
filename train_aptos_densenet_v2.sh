#!/bin/bash

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# APTOS + DenseNet121 V2 Medical-Grade Training Script
echo "🏥 APTOS + DenseNet121 V2 Medical-Grade Training"
echo "==============================================="
echo "🎯 Target: 85%+ accuracy with enhanced DenseNet121"
echo "📊 Dataset: APTOS 2019 (5-class DR classification)"
echo "🏗️ Model: DenseNet121 V2 (enhanced configuration)"
echo "🔬 Different hyperparameters for ensemble diversity"
echo ""

# Create output directory for APTOS V2 results
mkdir -p ./aptos_densenet_v2_results

echo "🔬 APTOS DenseNet121 V2 Enhanced Configuration:"
echo "  - Dataset: APTOS 2019 (./dataset_aptos)"
echo "  - Model: DenseNet121 (enhanced settings)"
echo "  - Image size: 224x224"
echo "  - Batch size: 12 (smaller for better gradients)"
echo "  - Learning rate: 1e-4 (more aggressive)"
echo "  - Weight decay: 5e-4 (reduced for less constraint)"
echo "  - Dropout: 0.3 (lighter regularization)"
echo "  - Epochs: 100 (more training time)"
echo "  - Enhanced augmentation + different seed"
echo "  - Target: 85%+ accuracy (ensemble model #4)"
echo ""

# Train APTOS with enhanced hyperparameters
python ensemble_local_trainer.py \
    --mode train \
    --dataset_path ./dataset_aptos \
    --output_dir ./aptos_densenet_v2_results \
    --experiment_name "aptos_densenet121_v2_enhanced" \
    --base_models densenet121 \
    --img_size 224 \
    --batch_size 12 \
    --epochs 100 \
    --learning_rate 1e-4 \
    --weight_decay 5e-4 \
    --ovo_dropout 0.3 \
    --enable_medical_augmentation \
    --rotation_range 30.0 \
    --brightness_range 0.2 \
    --contrast_range 0.2 \
    --enable_focal_loss \
    --enable_class_weights \
    --scheduler cosine \
    --warmup_epochs 5 \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --patience 10 \
    --early_stopping_patience 8 \
    --target_accuracy 0.85 \
    --seed 123

echo ""
echo "✅ APTOS DenseNet121 V2 training completed!"
echo "📁 Results saved to: ./aptos_densenet_v2_results"
echo ""
echo "🎯 Enhanced Configuration Applied:"
echo "  🏗️ Architecture: DenseNet121 V2 (enhanced hyperparameters)"
echo "  📊 Model capacity: 8M parameters (medical-grade)"
echo "  🎓 Aggressive learning rate: 1e-4 (faster learning)"
echo "  💧 Lighter dropout: 0.3 (less constraint)"
echo "  ⏰ Extended training: 100 epochs (more convergence time)"
echo "  🔀 Enhanced augmentation: 30° rotation, 20% brightness/contrast"
echo "  🎲 Different seed: 123 (model diversity)"
echo ""
echo "📊 Expected Performance:"
echo "  🎯 Target: 85%+ validation accuracy (enhanced model)"
echo "  🏥 Medical grade: Contribute to 90%+ ensemble"
echo "  📈 Model diversity: Different from original DenseNet"
echo "  🔬 5-model ensemble: Path to 90%+ accuracy"