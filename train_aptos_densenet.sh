#!/bin/bash

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# APTOS 2019 + DenseNet121 Medical-Grade Training Script
echo "🏥 APTOS 2019 + DenseNet121 Medical-Grade Training"
echo "================================================="
echo "🎯 Target: 91%+ accuracy with DenseNet121 architecture"
echo "📊 Dataset: APTOS 2019 (5-class DR classification)"
echo "🏗️ Model: DenseNet121 (research-proven 91.21% performer)"
echo "🔬 Medical-grade architecture with optimized hyperparameters"
echo ""

# Create output directory for APTOS results
mkdir -p ./aptos_results

echo "🔬 APTOS 2019 Medical-Grade Configuration:"
echo "  - Dataset: APTOS 2019 (./dataset_aptos)"
echo "  - Model: DenseNet121 (medical-grade capacity)"
echo "  - Image size: 224x224 (research paper standard)"
echo "  - Batch size: 16 (optimized for DenseNet)"
echo "  - Learning rate: 2e-4 (DenseNet optimized)"
echo "  - Weight decay: 1e-3 (strong regularization)"
echo "  - Dropout: 0.4 (balanced regularization)"
echo "  - Epochs: 80 (sufficient for convergence)"
echo "  - Enhanced augmentation + progressive training"
echo "  - Target: 91%+ accuracy (medical production grade)"
echo ""

# Train APTOS with research-validated hyperparameters
python ensemble_local_trainer.py \
    --mode train \
    --dataset_path ./dataset_aptos \
    --output_dir ./aptos_results \
    --experiment_name "aptos_2019_densenet121_medical" \
    --base_models densenet121 \
    --img_size 224 \
    --batch_size 16 \
    --epochs 80 \
    --learning_rate 2e-4 \
    --weight_decay 1e-3 \
    --ovo_dropout 0.4 \
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
    --target_accuracy 0.91 \
    --seed 42

echo ""
echo "✅ APTOS 2019 medical-grade training completed!"
echo "📁 Results saved to: ./aptos_results"
echo ""
echo "🎯 Medical-Grade Improvements Applied:"
echo "  🏗️ Architecture: MobileNet→DenseNet121 (91.21% research target)"
echo "  📊 Model capacity: 3M→8M parameters (medical-grade capacity)"
echo "  🎓 Optimized learning rate: 2e-4 (DenseNet optimized)"
echo "  💧 Balanced dropout: 0.4 (prevents overfitting without hurting performance)"
echo "  ⏰ Extended training: 50→80 epochs (sufficient convergence)"
echo "  🔀 Refined augmentation: Medical imaging optimized settings"
echo ""
echo "📊 Expected Performance:"
echo "  🎯 Target: 91%+ validation accuracy (medical production grade)"
echo "  🏥 Medical grade: Should achieve FULL PASS (≥90%)"
echo "  📈 Generalization: Better performance on new patients"
echo "  🔬 Research validated: DenseNet121 achieves 91.21% on DR datasets"
echo ""
echo "📋 Next Steps:"
echo "  1. Analyze results: python model_analyzer.py --model ./aptos_results/models/best_densenet121.pth"
echo "  2. Validate medical-grade performance (>90% required)"
echo "  3. If successful, proceed to Phase 2: Lesion Detection"