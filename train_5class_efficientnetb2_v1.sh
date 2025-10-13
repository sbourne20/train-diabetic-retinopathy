#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 5-CLASS DR + EfficientNetB2 Medical-Grade Training Script
echo "🏥 5-CLASS DR + EfficientNetB2 Medical-Grade Training (v1)"
echo "===================================================================="
echo "🎯 Target: 96%+ accuracy (Research target: 96.27% individual)"
echo "📊 Dataset: 5-Class Perfectly Balanced (53,935 images)"
echo "🏗️ Model: EfficientNetB2 (9.2M params - compound scaling)"
echo "🔗 System: V100 16GB GPU optimized"
echo ""

# Create output directory for 5-class EfficientNetB2 results
mkdir -p ./efficientnetb2_5class_results

echo "🔬 5-CLASS EfficientNetB2 OVO ENSEMBLE Configuration (v1):"
echo "  - Dataset: ./dataset_eyepacs_5class_balanced"
echo "  - Classes: 5 (0: No DR, 1: Mild NPDR, 2: Moderate NPDR, 3: Severe NPDR, 4: PDR)"
echo "  - Total images: 53,935 (Train: 37,750 | Val: 8,090 | Test: 8,095)"
echo "  - Perfect balance: 10,787 per class (1.00:1 ratio)"
echo ""
echo "📊 WHY EFFICIENTNETB2?"
echo "  ✅ Research target: 96.27% individual accuracy (vs DenseNet121: 91.21%)"
echo "  ✅ Better regularization: Built-in SE blocks + stochastic depth"
echo "  ✅ Optimal resolution: 260×260 (compound scaling optimized)"
echo "  ✅ Less overfitting: DenseNet achieved 88.46% avg pairs with 64.84% ensemble"
echo "  ✅ Primary model in CLAUDE.md Phase 1 objectives"
echo ""
echo "🎯 v1 STRATEGY - PROVEN CONFIGURATION:"
echo "  - Image size: 260×260 (EfficientNetB2 optimal resolution)"
echo "  - Batch size: 8 (adjusted for larger input size)"
echo "  - Learning rate: 8e-5 (lower than DenseNet due to better convergence)"
echo "  - Weight decay: 2.5e-4 (less than DenseNet - EfficientNet has built-in regularization)"
echo "  - Dropout: 0.28 (lower than DenseNet 0.32 - SE blocks provide regularization)"
echo "  - Label smoothing: 0.10 (standard for medical imaging)"
echo "  - CLAHE: ✅ ENABLED (essential for retinal vessel enhancement)"
echo "  - Focal loss: ✅ ENABLED (alpha=2.5, gamma=3.0 for class balance)"
echo "  - Augmentation: MEDICAL-GRADE (25° rotation, 20% brightness/contrast)"
echo "  - Scheduler: Cosine with 10-epoch warmup"
echo "  - Patience: 22 epochs (allow sufficient learning)"
echo ""

# Train 5-Class with EfficientNetB2
python3 ensemble_5class_trainer.py \
    --mode train \
    --dataset_path ./dataset_eyepacs_5class_balanced \
    --output_dir ./efficientnetb2_5class_results \
    --experiment_name "5class_efficientnetb2_v1_ovo" \
    --base_models efficientnetb2 \
    --num_classes 5 \
    --img_size 260 \
    --batch_size 8 \
    --epochs 100 \
    --learning_rate 8e-5 \
    --weight_decay 2.5e-4 \
    --ovo_dropout 0.28 \
    --freeze_weights false \
    --enable_clahe \
    --clahe_clip_limit 2.5 \
    --enable_medical_augmentation \
    --rotation_range 25.0 \
    --brightness_range 0.20 \
    --contrast_range 0.20 \
    --enable_focal_loss \
    --enable_class_weights \
    --class_weight_0 1.0 \
    --class_weight_1 1.0 \
    --class_weight_2 1.0 \
    --class_weight_3 1.0 \
    --class_weight_4 1.0 \
    --focal_loss_alpha 2.5 \
    --focal_loss_gamma 3.0 \
    --scheduler cosine \
    --warmup_epochs 10 \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --patience 22 \
    --early_stopping_patience 17 \
    --target_accuracy 0.95 \
    --max_grad_norm 1.0 \
    --label_smoothing 0.10 \
    --seed 42

echo ""
echo "✅ 5-CLASS EfficientNetB2 OVO ENSEMBLE training completed!"
echo ""
echo "📊 EFFICIENTNETB2 vs DENSENET121 COMPARISON:"
echo ""
echo "  Parameter          | DenseNet121 (v3) | EfficientNetB2 (v1) | Rationale"
echo "  -------------------|------------------|---------------------|------------------"
echo "  Architecture       | Dense Blocks     | Compound Scaling    | Better efficiency"
echo "  Parameters         | 8.0M             | 9.2M                | Slightly larger"
echo "  Image Size         | 299×299          | 260×260             | Optimal for EffNet"
echo "  Batch Size         | 10               | 8                   | Memory adjusted"
echo "  Learning Rate      | 9e-5             | 8e-5                | Faster convergence"
echo "  Dropout            | 0.32             | 0.28                | Built-in SE blocks"
echo "  Weight Decay       | 3.5e-4           | 2.5e-4              | Less regularization needed"
echo "  Label Smoothing    | 0.11             | 0.10                | Standard value"
echo ""
echo "🎯 EXPECTED RESULTS (Based on Research):"
echo "  ✅ Individual pairs: 93-97% accuracy (vs DenseNet: 77-99%)"
echo "  ✅ Average pair accuracy: 95-96% (vs DenseNet: 88.46%)"
echo "  ✅ Ensemble accuracy: 95-96% (vs DenseNet: 64.84%)"
echo "  ✅ Weak pairs (1-2, 2-3, 3-4): Expected 90-94% (vs DenseNet: 77-83%)"
echo "  ✅ Strong pairs (0-3, 0-4): Expected 98-99% (maintained)"
echo ""
echo "📈 KEY ADVANTAGES OF EFFICIENTNETB2:"
echo "  1. Compound scaling: Balanced depth/width/resolution"
echo "  2. Squeeze-and-Excitation blocks: Channel-wise attention"
echo "  3. Stochastic depth: Regularization during training"
echo "  4. Better feature extraction: Mobile Inverted Bottleneck Conv"
echo "  5. Proven medical imaging performance: 96.27% target"
echo ""
echo "⚠️ MONITORING CHECKPOINTS:"
echo "  1. Pair (1,2): DenseNet 77.13% → Target >90%"
echo "  2. Pair (2,3): DenseNet 83.03% → Target >92%"
echo "  3. Pair (3,4): DenseNet 78.71% → Target >90%"
echo "  4. Ensemble: DenseNet 64.84% → Target >95%"
echo ""
echo "🔧 NEXT STEPS IF RESULTS SUBOPTIMAL:"
echo "  If accuracy <94%:"
echo "    • Try EfficientNetB3 (12.3M params, 300×300 input)"
echo "    • Multi-architecture ensemble: EfficientNetB2 + ResNet50"
echo "    • Increase learning rate to 9e-5"
echo ""
echo "  If overfitting >3% gap:"
echo "    • Increase dropout to 0.30"
echo "    • Increase weight decay to 3e-4"
echo "    • Add more augmentation"
echo ""
echo "🚀 Training started at: $(date)"
echo "📁 Results directory: ./efficientnetb2_5class_results/"
echo "📊 Monitor progress: tail -f ./efficientnetb2_5class_results/logs/*.log"
echo ""
