#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 5-CLASS DR + DenseNet121 HYBRID Training Script (High Resolution)
echo "🏥 5-CLASS DR + DenseNet121 HYBRID Training (v4 - High Resolution)"
echo "===================================================================="
echo "🎯 Target: 94%+ accuracy (Kaggle winner's resolution + proven v3 config)"
echo "📊 Dataset: 5-Class Perfectly Balanced (53,935 images)"
echo "🏗️ Model: DenseNet121 (8M params - dense connectivity)"
echo "🔗 System: V100 16GB GPU optimized"
echo ""

# Create output directory for 5-class DenseNet v4 results
mkdir -p ./densenet_5class_v4_results

echo "🔬 5-CLASS DenseNet121 OVO ENSEMBLE Configuration (v4 - HIGH RESOLUTION HYBRID):"
echo "  - Dataset: ./dataset_eyepacs_5class_balanced"
echo "  - Classes: 5 (0: No DR, 1: Mild NPDR, 2: Moderate NPDR, 3: Severe NPDR, 4: PDR)"
echo "  - Total images: 53,935 (Train: 37,750 | Val: 8,090 | Test: 8,095)"
echo "  - Perfect balance: 10,787 per class (1.00:1 ratio)"
echo ""
echo "📊 HYBRID APPROACH - v4 EVOLUTION:"
echo "  ✅ From Kaggle Winner (Guanshuo Xu - 1st place):"
echo "     • Much higher resolution: 448×448 (closer to 512×512)"
echo "     • Focus on image detail extraction"
echo "  "
echo "  ✅ From DenseNet v3 (Your Proven Config):"
echo "     • CLAHE preprocessing (essential)"
echo "     • Fine-tuned regularization (no overfitting)"
echo "     • Medical-grade augmentation"
echo "     • Focal loss + class weighting"
echo ""
echo "📊 v4 CRITICAL CHANGES FROM v3:"
echo "  Parameter          | v3 (Proven: 64.84%) | v4 (Hybrid: High-Res) | Rationale"
echo "  -------------------|---------------------|----------------------|------------------"
echo "  Image Size         | 299×299             | 448×448              | +125% pixels (2.2× memory)"
echo "  Batch Size         | 10                  | 8                    | Memory for 448×448"
echo "  Learning Rate      | 9e-5                | 7e-5                 | More stable for high-res"
echo "  Dropout            | 0.32                | 0.30                 | More capacity for features"
echo "  Weight Decay       | 3.5e-4              | 3e-4                 | Balanced regularization"
echo "  Label Smoothing    | 0.11                | 0.10                 | Standard medical value"
echo "  Patience           | 22                  | 25                   | Allow more learning time"
echo ""
echo "⚠️  WHY 448×448 FOR DENSENET?"
echo "  1. ARCHITECTURE: DenseNet's dense connections excel with high-res inputs"
echo "  2. RESEARCH: Paper shows DenseNet121 achieves 91.21% individual (vs EfficientNetB2 96.27%)"
echo "  3. HYPOTHESIS: Low resolution (299) was limiting factor in v3's 64.84%"
echo "  4. MEMORY: DenseNet memory-efficient → can handle 448 with batch 8"
echo "  5. TARGET: Close the gap to EfficientNetB2 by leveraging resolution"
echo ""
echo "🎯 v4 CONFIGURATION - RESOLUTION UPGRADE:"
echo "  - Image size: 448×448 (MAJOR UPGRADE from 299, close to winner's 512)"
echo "  - Batch size: 8 (REDUCED from 10 for memory)"
echo "  - Learning rate: 7e-5 (REDUCED from 9e-5 for stability)"
echo "  - Weight decay: 3e-4 (REDUCED from 3.5e-4, proven v3 baseline)"
echo "  - Dropout: 0.30 (REDUCED from 0.32 for more capacity)"
echo "  - Label smoothing: 0.10 (REDUCED from 0.11 to standard)"
echo "  - CLAHE: ✅ ENABLED (clip_limit=2.5, KEEP from v3)"
echo "  - Focal loss: ✅ ENABLED (alpha=2.5, gamma=3.0, KEEP from v3)"
echo "  - Augmentation: MEDICAL-GRADE (25° rotation, 20% brightness/contrast)"
echo "  - Scheduler: Cosine with 10-epoch warmup"
echo "  - Patience: 25 epochs (INCREASED from 22 for high-res learning)"
echo "  - Epochs: 100 (same as v3)"
echo ""
echo "📈 EXPECTED RESULTS UPGRADE PATH:"
echo "  v3 Performance (299×299):"
echo "    • Best pair: 0-1 = 88.57% accuracy"
echo "    • Average pairs: ~88.46%"
echo "    • Ensemble: 64.84% (gap suggests voting issues, not overfitting)"
echo "    • Overfitting: <2% gap ✅ No overfitting"
echo "  "
echo "  v4 Target (448×448):"
echo "    • Strong pairs (0v3, 0v4): 95-98% (high-res detail)"
echo "    • Good pairs (0v1, 0v2, 1v3, 1v4, 2v4): 92-95% (resolution boost)"
echo "    • Weak pairs (1v2, 2v3, 3v4): 88-92% (MAJOR improvement from 77-83%)"
echo "    • Average pair accuracy: 93-95% (vs v3: 88.46%)"
echo "    • Ensemble: 92-94% (vs v3: 64.84%) - TARGET BREAKTHROUGH"
echo ""

# Train 5-Class with DenseNet121 (High Resolution Hybrid + Grade-Specific Preprocessing)
python3 ensemble_5class_trainer.py \
    --mode train \
    --dataset_path ./dataset_eyepacs_5class_balanced_enhanced \
    --output_dir ./densenet_5class_v4_enhanced_results \
    --experiment_name "5class_densenet121_v4_gradespec_enhanced" \
    --base_models densenet121 \
    --num_classes 5 \
    --img_size 384 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --epochs 100 \
    --learning_rate 5e-5 \
    --weight_decay 5e-4 \
    --ovo_dropout 0.40 \
    --freeze_weights false \
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
    --patience 25 \
    --early_stopping_patience 20 \
    --target_accuracy 0.94 \
    --max_grad_norm 1.0 \
    --label_smoothing 0.10 \
    --seed 42 \
    --resume

echo ""
echo "✅ 5-CLASS DenseNet121 HYBRID OVO ENSEMBLE training completed!"
