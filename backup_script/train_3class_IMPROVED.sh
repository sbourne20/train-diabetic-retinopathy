#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 3-CLASS DR + DenseNet121 IMPROVED Training for Critical NORMAL vs NPDR
echo "🏥 3-CLASS DR + DenseNet121 IMPROVED - OPTIMIZED FOR NORMAL vs NPDR"
echo "=================================================================="
echo "🚨 CRITICAL: Optimized for NORMAL vs NPDR detection (clinical importance)"
echo "📊 Dataset: 3-Class Balanced (39,850 images - NORMAL, NPDR, PDR)"
echo "🏗️ Model: DenseNet121 with STRONGER anti-overfitting"
echo "🔗 System: V100 16GB GPU optimized"
echo ""

# Create output directory for improved 3-class DenseNet results
mkdir -p ./densenet_3class_improved_results

echo "🔬 IMPROVED 3-CLASS Configuration - NORMAL vs NPDR CRITICAL:"
echo "  - Dataset: ./dataset_eyepacs_3class_balanced"
echo "  - Classes: 3 (NORMAL, NPDR merged 1-3, PDR)"
echo "  - Focus: Maximize pair_0_1 (NORMAL vs NPDR) accuracy"
echo "  - Clinical rationale: Early detection = treatment window"
echo "  - Model: DenseNet121 (8M params)"
echo "  - Image size: 299x299"
echo "  - Batch size: 8 (smaller for more updates)"
echo "  - Learning rate: 3e-5 (LOWER for fine-grained learning)"
echo "  - Weight decay: 5e-4 (STRONGER regularization)"
echo "  - Dropout: 0.5 (MUCH STRONGER anti-overfitting)"
echo "  - Epochs: 150 (more time to converge slowly)"
echo "  - CLAHE: ENABLED (clip_limit=3.0 - stronger contrast)"
echo "  - Focal loss: alpha=3.0, gamma=4.0 (STRONGER focus on hard cases)"
echo "  - Label smoothing: 0.2 (STRONGER to prevent overconfidence)"
echo "  - Augmentation: STRONGER (35° rotation, 30% brightness/contrast)"
echo "  - Early stopping: patience=30 (give more time)"
echo ""

# Train 3-Class with IMPROVED hyperparameters specifically for NORMAL vs NPDR
python3 ensemble_3class_trainer.py \
    --mode train \
    --dataset_path ./dataset_eyepacs_3class_balanced \
    --output_dir ./densenet_3class_improved_results \
    --experiment_name "3class_densenet121_IMPROVED_NPDR_FOCUS" \
    --base_models densenet121 \
    --num_classes 3 \
    --img_size 299 \
    --batch_size 8 \
    --epochs 150 \
    --learning_rate 3e-5 \
    --weight_decay 5e-4 \
    --ovo_dropout 0.5 \
    --freeze_weights false \
    --enable_clahe \
    --clahe_clip_limit 3.0 \
    --enable_medical_augmentation \
    --rotation_range 35.0 \
    --brightness_range 0.30 \
    --contrast_range 0.30 \
    --enable_focal_loss \
    --enable_class_weights \
    --class_weight_normal 0.515 \
    --class_weight_npdr 1.323 \
    --class_weight_pdr 3.321 \
    --focal_loss_alpha 3.0 \
    --focal_loss_gamma 4.0 \
    --scheduler cosine \
    --warmup_epochs 15 \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --patience 30 \
    --early_stopping_patience 30 \
    --target_accuracy 0.95 \
    --max_grad_norm 0.5 \
    --label_smoothing 0.2 \
    --seed 42

echo ""
echo "✅ IMPROVED 3-CLASS DenseNet121 training completed!"
echo "📁 Results saved to: ./densenet_3class_improved_results"
echo ""
echo "🎯 KEY IMPROVEMENTS FOR NORMAL vs NPDR:"
echo "  🔬 Stronger CLAHE (3.0 vs 2.5) - better microaneurysm visibility"
echo "  🎓 Lower learning rate (3e-5 vs 1e-4) - finer feature learning"
echo "  💧 Higher dropout (0.5 vs 0.3) - stronger anti-overfitting"
echo "  ⚖️ Stronger weight decay (5e-4 vs 3e-4) - better regularization"
echo "  🔀 Stronger augmentation (35° vs 25°, 30% vs 20%) - better generalization"
echo "  🎯 Stronger focal loss (gamma=4.0 vs 3.0) - focus on hard cases"
echo "  🏷️ Stronger label smoothing (0.2 vs 0.1) - prevent overconfidence"
echo "  📊 Smaller batch size (8 vs 10) - more gradient updates"
echo "  ⏰ More epochs (150 vs 100) - more time for slow convergence"
echo "  🛑 More patience (30 vs 20) - don't stop too early"
echo ""
echo "📊 EXPECTED IMPROVEMENT:"
echo "  Previous pair_0_1: 84.56% (overfitting)"
echo "  Target pair_0_1: 88-92% (better generalization)"
echo "  Clinical impact: Better NORMAL vs NPDR detection"
echo ""
