#!/bin/bash

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# APTOS + MobileNetV2 V2 Medical-Grade Training Script
echo "🏥 APTOS + MobileNetV2 V2 Medical-Grade Training"
echo "==============================================="
echo "🎯 Target: 85%+ accuracy with enhanced MobileNetV2"
echo "📊 Dataset: APTOS 2019 (5-class DR classification)"
echo "🏗️ Model: MobileNetV2 V2 (enhanced configuration)"
echo "🔬 Different hyperparameters for ensemble diversity"
echo ""

# Create output directory for APTOS V2 results
mkdir -p ./aptos_mobilenet_v2_results

echo "🔬 APTOS MobileNetV2 V2 Enhanced Configuration:"
echo "  - Dataset: APTOS 2019 (./dataset_aptos)"
echo "  - Model: MobileNetV2 (enhanced settings)"
echo "  - Image size: 224x224"
echo "  - Batch size: 32 (larger for stable gradients)"
echo "  - Learning rate: 5e-4 (more aggressive)"
echo "  - Weight decay: 2e-3 (stronger regularization)"
echo "  - Dropout: 0.6 (stronger regularization)"
echo "  - Epochs: 80 (sufficient training time)"
echo "  - Enhanced augmentation + different seed"
echo "  - Target: 85%+ accuracy (ensemble model #5)"
echo ""

# Train APTOS with enhanced hyperparameters
python ensemble_local_trainer.py \
    --mode train \
    --dataset_path ./dataset_aptos \
    --output_dir ./aptos_mobilenet_v2_results \
    --experiment_name "aptos_mobilenet_v2_enhanced" \
    --base_models mobilenet_v2 \
    --img_size 224 \
    --batch_size 32 \
    --epochs 80 \
    --learning_rate 5e-4 \
    --weight_decay 2e-3 \
    --ovo_dropout 0.6 \
    --enable_medical_augmentation \
    --rotation_range 25.0 \
    --brightness_range 0.25 \
    --contrast_range 0.25 \
    --enable_focal_loss \
    --enable_class_weights \
    --scheduler cosine \
    --warmup_epochs 5 \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --patience 12 \
    --early_stopping_patience 10 \
    --target_accuracy 0.85 \
    --seed 456

echo ""
echo "✅ APTOS MobileNetV2 V2 training completed!"
echo "📁 Results saved to: ./aptos_mobilenet_v2_results"
echo ""
echo "🎯 Enhanced Configuration Applied:"
echo "  🏗️ Architecture: MobileNetV2 V2 (enhanced hyperparameters)"
echo "  📊 Model capacity: 3M parameters (efficient)"
echo "  🎓 Aggressive learning rate: 5e-4 (faster learning)"
echo "  💧 Strong dropout: 0.6 (prevent overfitting)"
echo "  ⏰ Training epochs: 80 (sufficient convergence)"
echo "  🔀 Enhanced augmentation: 25° rotation, 25% brightness/contrast"
echo "  🎲 Different seed: 456 (model diversity)"
echo ""
echo "📊 Expected Performance:"
echo "  🎯 Target: 85%+ validation accuracy (enhanced model)"
echo "  🏥 Medical grade: Contribute to 90%+ ensemble"
echo "  📈 Model diversity: Different from original MobileNet"
echo "  🔬 5-model ensemble: Path to 90%+ accuracy"