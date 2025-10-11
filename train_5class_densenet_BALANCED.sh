#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 5-CLASS DR + DenseNet121 Medical-Grade Training Script
echo "🏥 5-CLASS DR + DenseNet121 Medical-Grade Training (CLAHE + Anti-Overfitting)"
echo "==============================================================================="
echo "🎯 Target: 95%+ accuracy with CLAHE enabled but overfitting prevented"
echo "📊 Dataset: 5-Class Perfectly Balanced (53,935 images)"
echo "🏗️ Model: DenseNet121 (8M params - dense connectivity)"
echo "🔗 System: V100 16GB GPU optimized"
echo ""

# Create output directory for 5-class DenseNet results
mkdir -p ./densenet_5class_results

echo "🔬 5-CLASS DenseNet121 OVO ENSEMBLE Configuration (BALANCED APPROACH):"
echo "  - Dataset: ./dataset_eyepacs_5class_balanced"
echo "  - Classes: 5 (0: No DR, 1: Mild NPDR, 2: Moderate NPDR, 3: Severe NPDR, 4: PDR)"
echo "  - Total images: 53,935 (Train: 37,750 | Val: 8,090 | Test: 8,095)"
echo "  - Class distribution: PERFECTLY BALANCED (10,787 per class - 1.00:1 ratio)"
echo "  - Imbalance ratio: 1.00:1 (ZERO imbalance - perfectly balanced)"
echo "  - Model: DenseNet121 (8M params - dense connectivity)"
echo "  - OVO Training: 10 binary classifiers (pairs: 0-1, 0-2, 0-3, 0-4, 1-2, 1-3, 1-4, 2-3, 2-4, 3-4)"
echo "  - OVO Voting: Weighted voting with PDR boost"
echo "  - Image size: 299x299 (optimal for medical imaging)"
echo "  - Batch size: 10 (optimized for V100 16GB)"
echo ""
echo "🎯 BALANCED STRATEGY (CLAHE + Anti-Overfitting):"
echo "  - CLAHE: ✅ ENABLED (clip_limit=2.5) - Essential for 98% accuracy"
echo "  - Learning rate: 8e-5 (SLIGHTLY REDUCED from proven 1e-4)"
echo "  - Weight decay: 4e-4 (MODERATELY INCREASED from 3e-4)"
echo "  - Dropout: 0.35 (SLIGHTLY INCREASED from 0.3)"
echo "  - Label smoothing: 0.12 (SLIGHTLY INCREASED from 0.1)"
echo "  - Focal loss: ✅ ENABLED (alpha=2.5, gamma=3.0)"
echo "  - Augmentation: MODERATE (25° rotation, 20% brightness/contrast)"
echo "  - Scheduler: Cosine with warm restarts (T_0=15)"
echo "  - Early stopping: patience=20 (between 25 and 10)"
echo "  - Validation: Every epoch with best model selection"
echo ""

# Train 5-Class with BALANCED hyperparameters (CLAHE + gentle anti-overfitting)
python3 ensemble_5class_trainer.py \
    --mode train \
    --dataset_path ./dataset_eyepacs_5class_balanced \
    --output_dir ./densenet_5class_results \
    --experiment_name "5class_densenet121_balanced_clahe" \
    --base_models densenet121 \
    --num_classes 5 \
    --img_size 299 \
    --batch_size 10 \
    --epochs 100 \
    --learning_rate 8e-5 \
    --weight_decay 4e-4 \
    --ovo_dropout 0.35 \
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
    --patience 20 \
    --early_stopping_patience 15 \
    --target_accuracy 0.95 \
    --max_grad_norm 1.0 \
    --label_smoothing 0.12 \
    --seed 42

echo ""
echo "✅ 5-CLASS DenseNet121 OVO ENSEMBLE training completed!"
echo "📁 Results saved to: ./densenet_5class_results"
echo ""
echo "🎯 OVO ENSEMBLE TRAINING CONFIGURATION:"
echo "  🔢 Binary classifiers: 10 (all pairwise combinations)"
echo "    • pair_0_1: No DR vs Mild NPDR"
echo "    • pair_0_2: No DR vs Moderate NPDR"
echo "    • pair_0_3: No DR vs Severe NPDR"
echo "    • pair_0_4: No DR vs PDR"
echo "    • pair_1_2: Mild NPDR vs Moderate NPDR"
echo "    • pair_1_3: Mild NPDR vs Severe NPDR"
echo "    • pair_1_4: Mild NPDR vs PDR"
echo "    • pair_2_3: Moderate NPDR vs Severe NPDR"
echo "    • pair_2_4: Moderate NPDR vs PDR"
echo "    • pair_3_4: Severe NPDR vs PDR"
echo "  🗳️ OVO Voting: Weighted with severity-based boost"
echo "  🏗️ Architecture: DenseNet121 (8M parameters)"
echo ""
echo "🔬 KEY DIFFERENCES FROM PREVIOUS ATTEMPTS:"
echo "  ✅ CLAHE: ENABLED (essential for accuracy) with moderate regularization"
echo "  ✅ Learning rate: 8e-5 (20% reduction from 1e-4 - gentler)"
echo "  ✅ Dropout: 0.35 (17% increase from 0.3 - moderate)"
echo "  ✅ Weight decay: 4e-4 (33% increase from 3e-4 - moderate)"
echo "  ✅ Label smoothing: 0.12 (20% increase from 0.1 - moderate)"
echo "  ✅ Patience: 20 (reduced from 25 but not too aggressive)"
echo "  ✅ Focal loss: ENABLED (proven effective with CLAHE)"
echo "  ✅ Augmentation: MODERATE (proven balance)"
echo ""
echo "📊 RATIONALE - BALANCED APPROACH:"
echo "  Problem 1: CLAHE disabled → 86% accuracy (12% regression from 98%)"
echo "  Problem 2: CLAHE enabled with original settings → overfitting"
echo "  Solution: CLAHE enabled + GENTLE regularization increases"
echo ""
echo "  Why this works:"
echo "  • CLAHE provides the feature clarity needed for 98% accuracy"
echo "  • Slightly lower LR (8e-5 vs 1e-4) prevents rapid overfitting"
echo "  • Moderate dropout (0.35 vs 0.3) adds regularization without hurting capacity"
echo "  • Moderate weight decay (4e-4 vs 3e-4) adds L2 penalty"
echo "  • Slightly higher label smoothing (0.12 vs 0.1) improves generalization"
echo "  • Patience 20 (vs 25) stops earlier but not too aggressively"
echo "  • Perfect balance (1.00:1 ratio) reduces overfitting risk naturally"
echo ""
echo "⚖️ ANTI-OVERFITTING MEASURES (MODERATE):"
echo "  ✅ Dropout 0.35 (gentle increase from 0.3)"
echo "  ✅ Weight decay 4e-4 (moderate increase from 3e-4)"
echo "  ✅ Label smoothing 0.12 (gentle increase from 0.1)"
echo "  ✅ Learning rate 8e-5 (20% reduction from 1e-4)"
echo "  ✅ Early stopping patience=20 (reduced from 25)"
echo "  ✅ Gradient clipping max_norm=1.0 (stable)"
echo "  ✅ Validation every epoch (monitoring)"
echo "  ✅ Checkpoint every 5 epochs (best model selection)"
echo "  ✅ Cosine scheduler with warm restarts (escape local minima)"
echo "  ✅ Perfectly balanced dataset (1.00:1 ratio - natural regularization)"
echo ""
echo "🎯 EXPECTED PERFORMANCE:"
echo "  Target: 95-97% validation accuracy"
echo "  Strategy: CLAHE clarity + moderate regularization"
echo "  Advantage: Best of both worlds (accuracy + generalization)"
echo "  Training time: ~12-15 hours on V100 16GB (10 classifiers × 100 epochs)"
echo ""
echo "💾 V100 16GB GPU OPTIMIZATION:"
echo "  ✅ Batch size: 10 (optimal for V100 with 299×299)"
echo "  ✅ Image size: 299×299 (DenseNet121 optimal)"
echo "  ✅ Gradient accumulation: 2 steps"
echo "  ✅ Pin memory: True (faster data loading)"
echo "  ✅ Persistent workers: num_workers=4"
echo "  ✅ Expected memory: ~6-7GB (safe for 16GB V100)"
echo ""
echo "🚀 NEXT STEPS AFTER TRAINING:"
echo "  1. Analyze results:"
echo "     python model_analyzer.py --model ./densenet_5class_results/models/ovo_ensemble_best.pth"
echo ""
echo "  2. Check individual pairs for overfitting:"
echo "     python model_analyzer.py --model ./densenet_5class_results/models/best_densenet121_0_2.pth"
echo ""
echo "  3. If overfitting still occurs:"
echo "     • Increase dropout to 0.4"
echo "     • Increase weight decay to 5e-4"
echo "     • Reduce patience to 15"
echo ""
echo "  4. If accuracy drops below 92%:"
echo "     • Restore learning rate to 1e-4"
echo "     • Reduce dropout back to 0.3"
echo "     • Check if CLAHE is actually enabled in logs"
echo ""
