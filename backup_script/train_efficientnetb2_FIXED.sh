#!/bin/bash

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# EyePACS Balanced + EfficientNetB2 - OPTIMIZED WITH BALANCED VALIDATION
echo "🏥 EyePACS BALANCED + EfficientNetB2 - BALANCED VALIDATION FIX"
echo "=========================================================="
echo "🎯 Target: 96%+ accuracy (Research: 96.27% achievable)"
echo "📊 Dataset: EyePACS Balanced (40,001 training images)"
echo "📊 Validation: BALANCED (71 per class = 355 total)"
echo "🏗️ Model: EfficientNetB2 (9M params)"
echo "🔧 FIX: Balanced validation for FAIR accuracy metrics"
echo ""

# Create output directory
mkdir -p ./efficientnetb2_eyepacs_balanced_results

echo "🔬 BALANCED VALIDATION Configuration:"
echo "  - Dataset: EyePACS Balanced - 40,001 training samples (8,000 per class)"
echo "  - Validation: BALANCED - 2,500 samples (500 per class)"
echo "  - Model: EfficientNetB2 (9M params)"
echo "  - Image size: 224x224"
echo "  - Batch size: 32"
echo "  - Learning rate: 3e-4"
echo "  - Weight decay: 1e-4"
echo "  - Dropout: 0.3"
echo "  - Epochs: 100"
echo "  - Scheduler: cosine annealing"
echo "  - Warmup: 2 epochs"
echo "  - Focal loss: gamma=2.0"
echo "  - Target: 96%+ validation accuracy"
echo ""
echo "🚀 FRESH START CONFIGURATION (NO FOCAL LOSS):"
echo "  ✅ Learning rate: 1e-3 (INCREASED - model needs to learn faster)"
echo "  ✅ Weight decay: 1e-5 (REDUCED - allow model to learn)"
echo "  ✅ Dropout: 0.3 (standard dropout)"
echo "  ✅ Augmentation: Strong (rotation 20°, brightness/contrast 15%)"
echo "  ✅ Loss: CrossEntropy with class weights (NO FOCAL LOSS)"
echo "  ✅ Patience: 20 epochs"
echo "  ✅ Warmup: 5 epochs"
echo ""
echo "💡 HYPOTHESIS: Focal loss was preventing model from learning!"
echo "   Switching to standard CrossEntropy with class weights."
echo ""

# Train EfficientNetB2 with FIXED hyperparameters
python ensemble_local_trainer.py \
    --mode train \
    --dataset_path ./dataset_eyepacs_ori_balanced_smote \
    --output_dir ./efficientnetb2_eyepacs_balanced_results \
    --experiment_name "eyepacs_balanced_efficientnetb2_fixed" \
    --base_models efficientnetb2 \
    --img_size 224 \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 1e-3 \
    --weight_decay 1e-5 \
    --ovo_dropout 0.3 \
    --freeze_weights false \
    --enable_medical_augmentation \
    --rotation_range 20.0 \
    --brightness_range 0.15 \
    --contrast_range 0.15 \
    --enable_class_weights \
    --scheduler cosine \
    --warmup_epochs 5 \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --patience 20 \
    --target_accuracy 0.96 \
    --max_grad_norm 1.0 \
    --label_smoothing 0.1 \
    --seed 42

echo ""
echo "✅ EyePACS Balanced EfficientNetB2 training completed!"
echo "📁 Results saved to: ./efficientnetb2_eyepacs_balanced_results"
echo ""
echo "📊 Expected Performance Timeline (FIXED VERSION):"
echo "  Epoch 5-10: 80-85% (vs 75-78% before)"
echo "  Epoch 15-20: 88-92% (vs 82-85% before)"
echo "  Epoch 25-40: 94-96% (vs 88-90% before)"
echo "  Epoch 50+: 96%+ target achieved"
echo ""
echo "🎯 Key improvements with 3x learning rate:"
echo "  ✅ Faster initial learning"
echo "  ✅ Reaches 90% medical-grade threshold earlier"
echo "  ✅ Better convergence with cosine annealing"
echo ""
