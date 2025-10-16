#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 5-CLASS DR + DenseNet121 Medical-Grade Training Script
echo "🏥 5-CLASS DR + DenseNet121 Medical-Grade Training (v3 - FINE-TUNED)"
echo "===================================================================="
echo "🎯 Target: 95%+ accuracy with minimal overfitting (<2% gap)"
echo "📊 Dataset: 5-Class Perfectly Balanced (53,935 images)"
echo "🏗️ Model: DenseNet121 (8M params - dense connectivity)"
echo "🔗 System: V100 16GB GPU optimized"
echo ""

# Create output directory for 5-class DenseNet results
mkdir -p ./densenet_5class_results

echo "🔬 5-CLASS DenseNet121 OVO ENSEMBLE Configuration (v3 - FINE-TUNED):"
echo "  - Dataset: ./dataset_eyepacs_5class_balanced"
echo "  - Classes: 5 (0: No DR, 1: Mild NPDR, 2: Moderate NPDR, 3: Severe NPDR, 4: PDR)"
echo "  - Total images: 53,935 (Train: 37,750 | Val: 8,090 | Test: 8,095)"
echo "  - Perfect balance: 10,787 per class (1.00:1 ratio)"
echo ""
echo "📊 CURRENT RESULTS ANALYSIS:"
echo "  Pair 0-1: Best 88.57% (gap ~0.7%) ✅ No overfitting BUT ❌ Below 95% target"
echo "  Pair 0-2: Current 85.32% (gap ~1.8%) ✅ No overfitting BUT ❌ Below target"
echo "  Problem: Regularization TOO aggressive → good generalization but low accuracy"
echo ""
echo "🎯 v3 STRATEGY - FINE-TUNED BALANCE:"
echo "  Goal: Recover accuracy to 92-95% while keeping overfitting gap <2%"
echo "  - CLAHE: ✅ ENABLED (essential for accuracy)"
echo "  - Learning rate: 9e-5 (SLIGHTLY HIGHER than 8e-5, still below 1e-4)"
echo "  - Weight decay: 3.5e-4 (REDUCED from 4e-4, closer to proven 3e-4)"
echo "  - Dropout: 0.32 (REDUCED from 0.35, closer to proven 0.3)"
echo "  - Label smoothing: 0.11 (REDUCED from 0.12, closer to proven 0.1)"
echo "  - Focal loss: ✅ ENABLED (alpha=2.5, gamma=3.0)"
echo "  - Patience: 22 (INCREASED from 20 - allow more learning)"
echo "  - Augmentation: MODERATE (25° rotation, 20% brightness/contrast)"
echo "  - Scheduler: Cosine with warm restarts"
echo ""

# Train 5-Class with FINE-TUNED hyperparameters
python3 ensemble_5class_trainer.py \
    --mode train \
    --dataset_path ./dataset_eyepacs_5class_balanced \
    --output_dir ./densenet_5class_results \
    --experiment_name "5class_densenet121_v3_finetuned" \
    --base_models densenet121 \
    --num_classes 5 \
    --img_size 299 \
    --batch_size 10 \
    --epochs 100 \
    --learning_rate 9e-5 \
    --weight_decay 3.5e-4 \
    --ovo_dropout 0.32 \
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
    --label_smoothing 0.11 \
    --seed 42

echo ""
echo "✅ 5-CLASS DenseNet121 OVO ENSEMBLE training completed!"
echo ""
echo "📊 v3 HYPERPARAMETER COMPARISON:"
echo ""
echo "  Parameter          | Original | Balanced | v3 (Fine-tuned)"
echo "  -------------------|----------|----------|----------------"
echo "  CLAHE              | ✅ Yes   | ✅ Yes   | ✅ Yes"
echo "  Learning Rate      | 1e-4     | 8e-5     | 9e-5 ⬆️"
echo "  Dropout            | 0.30     | 0.35     | 0.32 ⬇️"
echo "  Weight Decay       | 3e-4     | 4e-4     | 3.5e-4 ⬇️"
echo "  Label Smoothing    | 0.10     | 0.12     | 0.11 ⬇️"
echo "  Patience           | 25       | 20       | 22 ⬆️"
echo "  Focal Loss         | ✅ Yes   | ✅ Yes   | ✅ Yes"
echo ""
echo "🎯 RATIONALE:"
echo "  The 'Balanced' version prevented overfitting successfully (gap <2%)"
echo "  But accuracy was too low (88.57% vs 95% target)"
echo "  v3 loosens regularization slightly to recover accuracy:"
echo "    • Higher LR (9e-5): Learn faster without overfitting"
echo "    • Lower dropout (0.32): More model capacity"
echo "    • Lower weight decay (3.5e-4): Less L2 penalty"
echo "    • Lower label smoothing (0.11): Less forced uncertainty"
echo "    • Higher patience (22): Allow more learning epochs"
echo ""
echo "📈 EXPECTED RESULTS:"
echo "  Target: 92-95% validation accuracy"
echo "  Overfitting gap: <2% (train - val)"
echo "  Strategy: Slightly closer to proven config while keeping anti-overfitting"
echo ""
echo "⚠️ IF RESULTS STILL SHOW:"
echo "  1. Accuracy <90% BUT no overfitting → further reduce regularization"
echo "  2. Accuracy >93% BUT overfitting >3% → slightly increase regularization"
echo "  3. Both accuracy <90% AND overfitting → dataset/architecture issue"
echo ""
echo "🔧 NEXT ADJUSTMENTS IF NEEDED:"
echo "  To increase accuracy further (if still <92%):"
echo "    • LR: 9e-5 → 9.5e-5 or 1e-4"
echo "    • Dropout: 0.32 → 0.31 or 0.30"
echo "    • Weight decay: 3.5e-4 → 3.2e-4 or 3e-4"
echo ""
echo "  To reduce overfitting (if gap >2%):"
echo "    • Dropout: 0.32 → 0.33 or 0.35"
echo "    • Weight decay: 3.5e-4 → 3.8e-4 or 4e-4"
echo "    • Patience: 22 → 20 or 18"
echo ""
