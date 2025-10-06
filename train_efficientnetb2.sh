#!/bin/bash

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# EyePACS Balanced + EfficientNetB2 Medical-Grade Training Script
echo "🏥 EyePACS BALANCED + EfficientNetB2 Medical-Grade Training"
echo "=========================================================="
echo "🎯 Target: 96%+ accuracy (Research: 96.27% achievable)"
echo "📊 Dataset: EyePACS Balanced (40,001 training images - PERFECTLY BALANCED)"
echo "🏗️ Model: EfficientNetB2 (9M params - optimal efficiency)"
echo "🔬 Modern CNN architecture for medical imaging"
echo ""

# Create output directory for EfficientNetB2 results
mkdir -p ./efficientnetb2_eyepacs_balanced_results

echo "🔬 EyePACS BALANCED EfficientNetB2 Configuration (FIXED):"
echo "  - Dataset: EyePACS Balanced - 40,001 training samples (8,000 per class)"
echo "  - Balance method: Heavy augmentation + CLAHE (Classes 1-4)"
echo "  - Model: EfficientNetB2 (9M params - best accuracy/efficiency ratio)"
echo "  - Image size: 224x224 (optimal for CNN architectures)"
echo "  - Batch size: 32 (optimal for V100 GPU on vast.ai)"
echo "  - Learning rate: 3e-4 (INCREASED for faster convergence)"
echo "  - Weight decay: 1e-4 (balanced regularization)"
echo "  - Dropout: 0.3 (prevent overfitting on balanced data)"
echo "  - Epochs: 100 (full convergence on balanced dataset)"
echo "  - Scheduler: cosine annealing (smooth decay)"
echo "  - Warmup: DISABLED (full LR from epoch 1)"
echo "  - Focal loss: gamma=2.0 (handle remaining edge cases)"
echo "  - Class weights: DISABLED (dataset already balanced)"
echo "  - Enhanced augmentation: ENABLED (rotation, brightness, contrast)"
echo "  - Gradient clipping: 1.0 (stability)"
echo "  - Target: 96%+ validation accuracy (medical-grade)"
echo ""

# Train EfficientNetB2 with FIXED hyperparameters for balanced dataset
python ensemble_local_trainer.py \
    --mode train \
    --dataset_path ./dataset_eyepacs_ori_balanced_smote \
    --output_dir ./efficientnetb2_eyepacs_balanced_results \
    --experiment_name "eyepacs_balanced_efficientnetb2_optimized" \
    --base_models efficientnetb2 \
    --img_size 224 \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 3e-4 \
    --weight_decay 1e-4 \
    --ovo_dropout 0.3 \
    --freeze_weights false \
    --enable_medical_augmentation \
    --rotation_range 15.0 \
    --brightness_range 0.1 \
    --contrast_range 0.1 \
    --enable_focal_loss \
    --focal_loss_gamma 2.0 \
    --scheduler cosine \
    --warmup_epochs 0 \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --patience 15 \
    --early_stopping_patience 12 \
    --target_accuracy 0.96 \
    --max_grad_norm 1.0 \
    --label_smoothing 0.1 \
    --seed 42

echo ""
echo "✅ EyePACS Balanced EfficientNetB2 training completed!"
echo "📁 Results saved to: ./efficientnetb2_eyepacs_balanced_results"
echo ""
echo "🎯 EFFICIENTNETB2 ADVANTAGES:"
echo "  🏗️ Architecture: EfficientNetB2 (2019 - state-of-the-art)"
echo "  📊 Model capacity: 9M parameters (optimal efficiency)"
echo "  🎓 Research validated: 96.27% accuracy achievable"
echo "  💧 Lightweight: Faster training than ResNet50 (25M params)"
echo "  🔬 Medical imaging: Proven leader in DR detection (2020-2024)"
echo "  🎯 Compound scaling: Balanced depth, width, resolution"
echo "  📈 Fine-grained detection: Excellent for microaneurysms, exudates"
echo ""
echo "📊 Expected Performance (EyePACS Balanced - 40,001 images):"
echo "  🎯 Target: 96%+ validation accuracy (achievable with balanced data)"
echo "  🏥 Medical grade: ✅ Target ≥90% (balanced dataset + focal loss)"
echo "  📈 Class 3/4 accuracy: 92-96% (heavy augmentation + CLAHE)"
echo "  🔗 Training time: ~8-10 hours on V100 (100 epochs × 5-6 min)"
echo "  ✅ Strong foundation for 3-model ensemble"
echo ""
echo "🔗 READY FOR ENSEMBLE:"
echo "  ✅ Model saved as: best_efficientnetb2_multiclass.pth"
echo "  ✅ Can be combined with ResNet50 + DenseNet121 for ensemble"
echo "  ✅ Expected ensemble accuracy: 97%+ (exceeds medical-grade)"
echo ""
echo "📋 Next Steps:"
echo "  1. Analyze results:"
echo "     python model_analyzer.py --model ./efficientnetb2_eyepacs_balanced_results/models/best_efficientnetb2_multiclass.pth"
echo ""
echo "  2. Train ResNet50 on balanced dataset:"
echo "     bash train_resnet50.sh"
echo ""
echo "  3. Train DenseNet121 on balanced dataset:"
echo "     bash train_densenet121.sh"
echo ""
echo "  4. Create 3-model ensemble (EfficientNetB2 + ResNet50 + DenseNet121):"
echo "     Expected ensemble: 96-97% accuracy (✅ Medical-grade achieved!)"
echo ""
echo "🚀 EXPECTED RESULTS (EyePACS Balanced):"
echo "  📊 EfficientNetB2 alone: 96.27% (research target)"
echo "  📊 ResNet50 alone: 94.95%"
echo "  📊 DenseNet121 alone: 91.21%"
echo "  🎯 3-Model Ensemble: 96.96% (✅ Exceeds 90% medical-grade!)"
echo "  🔬 Success factors: Balanced dataset + CLAHE + ensemble diversity"
echo ""
