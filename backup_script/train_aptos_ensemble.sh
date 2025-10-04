#!/bin/bash

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# APTOS 2019 Multi-Architecture Ensemble Training Script
echo "🏥 APTOS 2019 Multi-Architecture Ensemble Training"
echo "================================================="
echo "🎯 Target: >96% accuracy with multi-architecture ensemble"
echo "📊 Dataset: APTOS 2019 (5-class DR classification)"
echo "🏗️ Models: EfficientNetB2 + ResNet50 + DenseNet121"
echo "🔬 Research-proven ensemble approach for medical-grade accuracy"
echo ""

# Create output directory for ensemble results
mkdir -p ./aptos_ensemble_results

echo "🔬 APTOS 2019 Multi-Architecture Ensemble Configuration:"
echo "  - Dataset: APTOS 2019 (./dataset_aptos)"
echo "  - Primary: EfficientNetB2 (96.27% individual target)"
echo "  - Secondary: ResNet50 (94.95% individual target)"
echo "  - Tertiary: DenseNet121 (91.21% individual target)"
echo "  - Ensemble: Simple averaging (96.96% combined target)"
echo "  - Image size: 224x224"
echo "  - Batch size: 8 (adjusted for ensemble memory requirements)"
echo "  - Learning rate: 3e-4 (ensemble-optimized)"
echo "  - Weight decay: 1e-3 (strong regularization)"
echo "  - Enhanced augmentation + early stopping"
echo ""

# Train APTOS ensemble with multiple architectures
python ensemble_local_trainer.py \
    --mode train \
    --dataset_path ./dataset_aptos \
    --output_dir ./aptos_ensemble_results \
    --experiment_name "aptos_2019_multiarch_ensemble" \
    --base_models efficientnet_b2,resnet50,densenet121 \
    --img_size 224 \
    --batch_size 8 \
    --epochs 100 \
    --learning_rate 3e-4 \
    --weight_decay 1e-3 \
    --ovo_dropout 0.5 \
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
    --patience 10 \
    --early_stopping_patience 10 \
    --target_accuracy 0.96 \
    --seed 42

echo ""
echo "✅ APTOS 2019 multi-architecture ensemble training completed!"
echo "📁 Results saved to: ./aptos_ensemble_results"
echo ""
echo "🎯 Multi-Architecture Ensemble Approach:"
echo "  🏆 EfficientNetB2: Medical imaging optimized architecture"
echo "  💪 ResNet50: Proven deep residual learning"
echo "  🧠 DenseNet121: Dense connectivity for feature reuse"
echo "  🤝 Ensemble: Combines strengths of all three models"
echo ""
echo "📊 Expected Performance:"
echo "  • Individual models: 90-96% each"
echo "  • Ensemble accuracy: 96%+ (medical-grade)"
echo "  • Reduced overfitting through model diversity"
echo "  • Improved generalization to new patients"
echo ""
echo "📋 Next Steps:"
echo "  1. Analyze ensemble results: python model_analyzer.py --model ./aptos_ensemble_results/models/ensemble_checkpoint.pth"
echo "  2. Validate medical-grade performance (>90% required)"
echo "  3. If successful, proceed to Phase 2: Lesion Detection"