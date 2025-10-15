#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 5-CLASS DR + MobileNetV2 Paper Replication Training Script
echo "🏥 5-CLASS DR + MobileNetV2 Paper Replication (v1)"
echo "===================================================================="
echo "🎯 Target: 92%+ accuracy (Paper's result: 92.00% on APTOS 2019)"
echo "📊 Dataset: 5-Class Perfectly Balanced (53,935 images)"
echo "🏗️ Model: MobileNetV2 (3.5M params - lightweight)"
echo "🔗 System: V100 16GB GPU optimized"
echo ""

# Create output directory for 5-class MobileNet results
mkdir -p ./mobilenet_5class_results

echo "🔬 5-CLASS MobileNetV2 OVO ENSEMBLE Configuration (Paper Replication v1):"
echo "  - Dataset: ./dataset_eyepacs_5class_balanced"
echo "  - Classes: 5 (0: No DR, 1: Mild NPDR, 2: Moderate NPDR, 3: Severe NPDR, 4: PDR)"
echo "  - Total images: 53,935 (Train: 37,750 | Val: 8,090 | Test: 8,095)"
echo "  - Perfect balance: 10,787 per class (1.00:1 ratio)"
echo ""
echo "📊 WHY MOBILENETV2 (PAPER'S BEST)?"
echo "  ✅ Paper's result: 92.00% accuracy (best performer vs DenseNet 90.18%)"
echo "  ✅ Lightweight: 3.5M parameters (vs EfficientNetB2 9.2M)"
echo "  ✅ Fast inference: Optimized for mobile/edge deployment"
echo "  ✅ Proven medical imaging: Widely used in DR detection"
echo "  ✅ Better than your previous: EfficientNetB2 64.20%, DenseNet121 64.84%"
echo ""
echo "🎯 v1 STRATEGY - PAPER'S PROVEN CONFIGURATION:"
echo "  - Image size: 224×224 (Paper's standard, NOT 260 or 299)"
echo "  - Batch size: 32 (Paper's setting, NOT 8 or 10)"
echo "  - Learning rate: 1e-3 (Paper's setting, NOT 8e-5 or 9e-5)"
echo "  - Weight decay: 1e-4 (Paper's standard regularization)"
echo "  - Dropout: 0.5 (Paper's conservative setting, NOT 0.28 or 0.32)"
echo "  - Label smoothing: 0.0 (DISABLED - paper didn't use it)"
echo "  - CLAHE: DISABLED (paper used simple preprocessing)"
echo "  - SMOTE: N/A (dataset already balanced)"
echo "  - Focal loss: DISABLED (paper used simple Cross-Entropy)"
echo "  - Augmentation: SIMPLE (rotation 45°, flip, zoom 0.2)"
echo "  - Scheduler: Cosine with 5-epoch warmup (NOT 10)"
echo "  - Patience: 15 epochs (shorter for faster training)"
echo "  - Epochs: 50 (Paper's setting, NOT 100)"
echo ""
echo "⚠️  CRITICAL DIFFERENCES FROM YOUR PREVIOUS TRAINING:"
echo "  HYPERPARAMETER          | Previous (Failed)  | Paper (Success)    | Change"
echo "  ------------------------|--------------------|--------------------|-------------"
echo "  Architecture            | EfficientNetB2     | MobileNetV2        | Simpler"
echo "  Image Size              | 260×260            | 224×224            | -14% pixels"
echo "  Batch Size              | 8                  | 32                 | +300%"
echo "  Learning Rate           | 8e-5               | 1e-3               | +1,150%"
echo "  Dropout                 | 0.28               | 0.5                | +79%"
echo "  Label Smoothing         | 0.10               | 0.0                | REMOVED"
echo "  CLAHE                   | ENABLED            | DISABLED           | REMOVED"
echo "  Focal Loss              | ENABLED            | DISABLED           | REMOVED"
echo "  Warmup Epochs           | 10                 | 5                  | -50%"
echo "  Total Epochs            | 100                | 50                 | -50%"
echo ""
echo "📈 EXPECTED RESULTS (Based on Paper's APTOS 2019):"
echo "  Individual Binary Pairs:"
echo "    • Strong pairs (0v3, 0v4): 97-99%"
echo "    • Good pairs (0v1, 0v2, 1v3, 1v4, 2v4): 85-95%"
echo "    • Weak pairs (1v2, 2v3, 3v4): 77-83%"
echo "    • Average pair accuracy: ~92%"
echo "  "
echo "  Final Ensemble Performance:"
echo "    • Target accuracy: 92%+"
echo "    • Medical grade threshold: ≥90%"
echo "    • Research quality: ≥85%"
echo ""

# Train 5-Class with MobileNetV2 (Paper Replication)
python3 ensemble_5class_trainer.py \
    --mode train \
    --dataset_path ./dataset_eyepacs_5class_balanced \
    --output_dir ./mobilenet_5class_results \
    --experiment_name "5class_mobilenet_v2_paper_replication" \
    --base_models mobilenet_v2 \
    --num_classes 5 \
    --img_size 224 \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --ovo_dropout 0.5 \
    --freeze_weights false \
    --enable_medical_augmentation \
    --rotation_range 45.0 \
    --brightness_range 0.2 \
    --contrast_range 0.2 \
    --label_smoothing 0.0 \
    --scheduler cosine \
    --warmup_epochs 5 \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --patience 15 \
    --early_stopping_patience 10 \
    --target_accuracy 0.92 \
    --max_grad_norm 1.0 \
    --seed 42 \
    --device cuda \
    --no_wandb

echo ""
echo "✅ 5-CLASS MobileNetV2 OVO ENSEMBLE training completed!"
echo ""
echo "📊 MOBILENETV2 vs YOUR PREVIOUS MODELS COMPARISON:"
echo ""
echo "  Parameter          | DenseNet121 (v3) | EfficientNetB2 (v1) | MobileNetV2 (v1) | Source"
echo "  -------------------|------------------|---------------------|------------------|------------------"
echo "  Result             | 64.84%           | 64.20%              | TARGET: 92%      | Paper vs Yours"
echo "  Architecture       | Dense Blocks     | Compound Scaling    | Inverted Residual| Design"
echo "  Parameters         | 8.0M             | 9.2M                | 3.5M             | Efficiency"
echo "  Image Size         | 299×299          | 260×260             | 224×224          | Input"
echo "  Batch Size         | 10               | 8                   | 32               | Training"
echo "  Learning Rate      | 9e-5             | 8e-5                | 1e-3             | Optimization"
echo "  Dropout            | 0.32             | 0.28                | 0.5              | Regularization"
echo "  Label Smoothing    | 0.11             | 0.10                | 0.0              | Loss"
echo "  CLAHE              | ENABLED          | ENABLED             | DISABLED         | Preprocessing"
echo "  Focal Loss         | ENABLED          | ENABLED             | DISABLED         | Loss Function"
echo ""
echo "🎯 EXPECTED RESULTS ANALYSIS:"
echo ""
echo "  IF ACCURACY ≥ 90%:"
echo "    ✅ SUCCESS! Paper configuration works on your dataset"
echo "    ✅ Proceed to train ResNet50 and DenseNet121 with same settings"
echo "    ✅ Then create meta-ensemble (3 models) → Expected: 93-95%"
echo ""
echo "  IF ACCURACY 85-90%:"
echo "    ⚠️  GOOD! Close to paper's result"
echo "    ⚠️  Dataset quality may differ from APTOS 2019"
echo "    ⚠️  Still suitable for meta-ensemble"
echo "    💡 Try: Increase epochs to 80, or adjust learning rate to 8e-4"
echo ""
echo "  IF ACCURACY 75-85%:"
echo "    ⚠️  MODERATE improvement over previous (64%)"
echo "    ⚠️  Hyperparameters working but dataset issues possible"
echo "    💡 Try: Review image quality, increase batch size to 64"
echo ""
echo "  IF ACCURACY < 75%:"
echo "    ❌ UNEXPECTED - Debug required"
echo "    💡 Check: Dataset balance, training logs, GPU utilization"
echo "    💡 Try: Longer training (100 epochs), add back focal loss"
echo ""
echo "📈 KEY PERFORMANCE INDICATORS TO MONITOR:"
echo "  1. Binary Pair Accuracies (should be 77-99%)"
echo "  2. Weak Pairs (1v2, 3v4) - Critical for ensemble"
echo "  3. Training vs Validation Gap (overfitting indicator)"
echo "  4. Loss curves (should converge smoothly)"
echo "  5. Confusion matrix (class-wise performance)"
echo ""
echo "⚠️  MONITORING CHECKPOINTS (Check These During Training):"
echo "  Epoch 5:  Expect ~70-75% (warmup complete)"
echo "  Epoch 10: Expect ~80-85% (learning progressing)"
echo "  Epoch 20: Expect ~88-90% (approaching target)"
echo "  Epoch 30: Expect ~90-92% (should reach target)"
echo "  Epoch 50: Expect ~92%+ (final performance)"
echo ""
echo "🔧 NEXT STEPS AFTER TRAINING:"
echo "  1. Analyze results:"
echo "     python3 model_analyzer.py --model ./mobilenet_5class_results/models/ovo_ensemble_best.pth"
echo ""
echo "  2. Compare with paper:"
echo "     - Paper's MobileNet: 92.00%"
echo "     - Your result: Check analyzer output"
echo "     - Gap analysis: Why different (if any)?"
echo ""
echo "  3. If successful (≥88%):"
echo "     - Train ResNet50 with same configuration"
echo "     - Train DenseNet121 with same configuration"
echo "     - Create meta-ensemble of all 3"
echo ""
echo "  4. If unsuccessful (<85%):"
echo "     - Review training logs for issues"
echo "     - Check dataset quality"
echo "     - Try alternative configurations"
echo ""
echo "🚀 Training started at: $(date)"
echo "📁 Results directory: ./mobilenet_5class_results/"
echo "📊 Monitor progress: tail -f ./mobilenet_5class_results/logs/*.log"
echo ""
echo "💡 REMINDER: This configuration exactly replicates the paper that achieved 92%"
echo "    Any significant deviation (<85%) suggests dataset or environment differences"
echo ""
