#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 5-CLASS DR + ConvNeXt-Tiny Training Script (Modern CNN Architecture)
echo "🏥 5-CLASS DR + ConvNeXt-Tiny OVO Training (v1 - Modern CNN)"
echo "====================================================================="
echo "🎯 Target: 95%+ accuracy (Modern CNN with Transformer-like design)"
echo "📊 Dataset: 5-Class Perfectly Balanced (53,935 images)"
echo "🏗️ Model: ConvNeXt-Tiny (28M params - SOTA pure CNN architecture)"
echo "🔗 System: V100 16GB GPU optimized"
echo ""

# Create output directory for 5-class ConvNeXt v1 results
mkdir -p ./convnext_5class_v1_results

echo "🔬 5-CLASS ConvNeXt-Tiny OVO ENSEMBLE Configuration (v1):"
echo "  - Dataset: ./dataset_eyepacs_5class_balanced_enhanced_v2"
echo "  - Classes: 5 (0: No DR, 1: Mild NPDR, 2: Moderate NPDR, 3: Severe NPDR, 4: PDR)"
echo "  - Total images: 53,935 (Train: 37,750 | Val: 8,090 | Test: 8,095)"
echo "  - Perfect balance: 10,787 per class (1.00:1 ratio)"
echo ""
echo "🏗️ ConvNeXt-Tiny ARCHITECTURE ADVANTAGES:"
echo "  ✅ Modern CNN Design (2022):"
echo "     • Modernized ResNet with 2020s techniques"
echo "     • Depthwise convolutions (like transformers)"
echo "     • Layer normalization (transformer-inspired)"
echo "     • GELU activation (modern choice)"
echo "  "
echo "  ✅ Medical Imaging Performance:"
echo "     • Proven SOTA on ImageNet (82.1% top-1)"
echo "     • Excellent feature extraction for fundus images"
echo "     • Better than ResNet/DenseNet on high-res inputs"
echo "     • Memory-efficient with gradient checkpointing"
echo ""
echo "📊 v1 CONFIGURATION - OPTIMIZED FOR CONVNEXT:"
echo "  Parameter          | Value              | Rationale"
echo "  -------------------|--------------------|-----------------------------------------"
echo "  Image Size         | 384×384            | Optimal for ConvNeXt (divisible by 32)"
echo "  Batch Size         | 6                  | Balance memory & training stability"
echo "  Gradient Accum.    | 2                  | Effective batch = 12"
echo "  Learning Rate      | 5e-5               | Conservative for pre-trained ConvNeXt"
echo "  Weight Decay       | 5e-4               | Strong regularization (ConvNeXt standard)"
echo "  Dropout            | 0.40               | Prevent overfitting on medical data"
echo "  Label Smoothing    | 0.10               | Standard medical value"
echo "  Patience           | 25                 | Allow convergence for complex architecture"
echo "  Scheduler          | Cosine             | Smooth learning rate decay"
echo "  Warmup Epochs      | 10                 | Gradual warm-up for stability"
echo ""
echo "🎯 PREPROCESSING & AUGMENTATION:"
echo "  - CLAHE: ✅ ENABLED (clip_limit=2.5, medical-grade contrast)"
echo "  - Focal loss: ✅ ENABLED (alpha=2.5, gamma=3.0, handle class imbalance)"
echo "  - Medical Augmentation:"
echo "    • Rotation: ±25° (preserve retinal anatomy)"
echo "    • Brightness/Contrast: ±20% (camera variation)"
echo "    • Horizontal flip: 50% probability"
echo "  - Normalization: ImageNet statistics (transfer learning)"
echo ""
echo "🎯 TRAINING STRATEGY:"
echo "  - OVO Binary Classifiers: 10 pairs (C(5,2) = 10)"
echo "  - Freeze Strategy: First 2 stages frozen, fine-tune later stages"
echo "  - Mixed Precision: FP16 (memory optimization)"
echo "  - Gradient Clipping: Max norm = 1.0 (stability)"
echo "  - Early Stopping: 20 epochs patience"
echo ""
echo "📈 EXPECTED RESULTS:"
echo "  Individual Binary Classifiers:"
echo "    • Strong pairs (0v3, 0v4): 96-98% (clear separation)"
echo "    • Good pairs (0v1, 0v2, 1v3, 1v4, 2v4): 93-96%"
echo "    • Challenging pairs (1v2, 2v3): 90-93% (adjacent classes)"
echo "    • Average pair accuracy: 94-96%"
echo "  "
echo "  Ensemble Performance:"
echo "    • Target: 95%+ overall accuracy"
echo "    • Per-class sensitivity: >92%"
echo "    • Per-class specificity: >96%"
echo ""
echo "🚀 Starting ConvNeXt-Tiny Training..."
echo ""

# Train 5-Class with ConvNeXt-Tiny
python3 ensemble_5class_trainer.py \
    --mode train \
    --dataset_path ./dataset_eyepacs_5class_balanced_enhanced_v2 \
    --output_dir ./convnext_5class_v1_results \
    --experiment_name "5class_convnext_tiny_v1_ovo" \
    --base_models convnext_tiny \
    --num_classes 5 \
    --img_size 384 \
    --batch_size 6 \
    --gradient_accumulation_steps 2 \
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
    --target_accuracy 0.95 \
    --max_grad_norm 1.0 \
    --label_smoothing 0.10 \
    --seed 42 \
    --resume

echo ""
echo "✅ 5-CLASS ConvNeXt-Tiny OVO ENSEMBLE training completed!"
echo ""
echo "📊 Next Steps:"
echo "  1. Analyze results: python model_analyzer.py --model ./convnext_5class_v1_results/models"
echo "  2. Evaluate ensemble: ./test_ovo_evaluation.sh (update to use ConvNeXt results)"
echo "  3. Compare with other models: DenseNet, EfficientNet, ResNet, MedSigLIP"
echo "  4. Check per-class performance and confusion matrix"
echo ""
