#!/bin/bash

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# APTOS 2019 + EfficientNetB2 Medical-Grade Training Script
echo "🏥 APTOS 2019 + EfficientNetB2 Medical-Grade Training"
echo "===================================================="
echo "🎯 Target: 96%+ accuracy with EfficientNetB2 architecture"
echo "📊 Dataset: APTOS 2019 (5-class DR classification)"
echo "🏗️ Model: EfficientNetB2 (research-proven 96.27% performer)"
echo "🔬 Medical-grade architecture with optimized hyperparameters"
echo ""

# Create output directory for APTOS results
mkdir -p ./aptos_results

echo "🔬 APTOS 2019 Medical-Grade Configuration:"
echo "  - Dataset: APTOS 2019 (./dataset_aptos)"
echo "  - Model: EfficientNetB2 (medical-grade capacity)"
echo "  - Image size: 224x224 (research paper standard)"
echo "  - Batch size: 12 (optimized for EfficientNet)"
echo "  - Learning rate: 3e-4 (EfficientNet optimized)"
echo "  - Weight decay: 1e-3 (strong regularization)"
echo "  - Dropout: 0.3 (balanced regularization)"
echo "  - Epochs: 100 (sufficient for convergence)"
echo "  - Enhanced augmentation + progressive training"
echo "  - Target: 96%+ accuracy (medical production grade)"
echo ""

# Train APTOS with research-validated hyperparameters
python ensemble_local_trainer.py \
    --mode train \
    --dataset_path ./dataset_aptos \
    --output_dir ./aptos_results \
    --experiment_name "aptos_2019_efficientnetb2_medical" \
    --base_models efficientnet_b2 \
    --img_size 224 \
    --batch_size 12 \
    --epochs 100 \
    --learning_rate 3e-4 \
    --weight_decay 1e-3 \
    --ovo_dropout 0.3 \
    --enable_medical_augmentation \
    --rotation_range 20.0 \
    --brightness_range 0.15 \
    --contrast_range 0.15 \
    --enable_focal_loss \
    --enable_class_weights \
    --scheduler cosine \
    --warmup_epochs 10 \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --patience 15 \
    --early_stopping_patience 12 \
    --target_accuracy 0.96 \
    --seed 42

echo ""
echo "✅ APTOS 2019 medical-grade training completed!"
echo "📁 Results saved to: ./aptos_results"
echo ""
echo "🎯 Medical-Grade Improvements Applied:"
echo "  🏗️ Architecture: MobileNet→EfficientNetB2 (96.27% research target)"
echo "  📊 Model capacity: 3M→8M parameters (medical-grade capacity)"
echo "  🎓 Optimized learning rate: 3e-4 (EfficientNet optimized)"
echo "  💧 Balanced dropout: 0.3 (prevents overfitting without hurting performance)"
echo "  ⏰ Extended training: 50→100 epochs (sufficient convergence)"
echo "  🔀 Refined augmentation: Medical imaging optimized settings"
echo ""
echo "📊 Expected Performance:"
echo "  🎯 Target: 96%+ validation accuracy (medical production grade)"
echo "  🏥 Medical grade: Should achieve FULL PASS (≥90%)"
echo "  📈 Generalization: Better performance on new patients"
echo "  🔬 Research validated: EfficientNetB2 achieves 96.27% on DR datasets"
echo ""
echo "📋 Next Steps:"
echo "  1. Analyze results: python model_analyzer.py --model ./aptos_results/models/best_efficientnet_b2.pth"
echo "  2. Validate medical-grade performance (>90% required)"
echo "  3. If successful, proceed to Phase 2: Lesion Detection"