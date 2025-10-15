#!/bin/bash

# PAPER REPLICATION: "A lightweight transfer learning based ensemble approach for diabetic retinopathy detection"
# Target: 92% accuracy (as achieved by paper on APTOS 2019)

source venv/bin/activate

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "📄 PAPER REPLICATION: MobileNetV2 OVO Ensemble"
echo "========================================================================"
echo "🎯 Target: 92%+ accuracy (Paper's result on APTOS 2019)"
echo "📊 Dataset: 5-Class Perfectly Balanced EyePACS"
echo "🏗️ Model: MobileNetV2 (Paper's best performer)"
echo "🔗 System: V100 16GB GPU optimized"
echo ""
echo "📋 PAPER'S EXACT CONFIGURATION:"
echo "  Architecture: MobileNetV2 (lightweight, 3.5M params)"
echo "  Image Size: 224×224 (Paper's standard)"
echo "  Batch Size: 32 (Paper's setting - 4x your previous)"
echo "  Learning Rate: 1e-3 = 0.001 (Paper's setting - 12.5x your previous)"
echo "  Epochs: 50 (Paper's setting)"
echo "  Dropout: 0.5 (Paper's conservative setting)"
echo "  Preprocessing: SIMPLE (no CLAHE, no SMOTE, no label smoothing)"
echo "  Augmentation: Basic rotation (45°), flip, zoom (0.2)"
echo "  Scheduler: Cosine annealing (Paper's approach)"
echo "  Warmup: 5 epochs (shorter than your previous 10)"
echo ""
echo "⚠️  KEY DIFFERENCES FROM YOUR PREVIOUS TRAINING:"
echo "  ✅ MobileNetV2 instead of EfficientNetB2 (Paper's best)"
echo "  ✅ Learning Rate: 1e-3 (was 8e-5) → 12.5x HIGHER"
echo "  ✅ Batch Size: 32 (was 8) → 4x LARGER"
echo "  ✅ Image Size: 224 (was 260) → SMALLER, faster"
echo "  ✅ NO CLAHE enhancement (was enabled)"
echo "  ✅ NO Label Smoothing (was 0.10)"
echo "  ✅ Dropout: 0.5 (was 0.28) → MORE regularization"
echo "  ✅ Epochs: 50 (was 100) → FASTER training"
echo ""
echo "🔬 EXPECTED RESULTS (Based on Paper):"
echo "  Binary Pair Accuracies: 72-99%"
echo "  Average Pair Accuracy: ~92%"
echo "  Ensemble Accuracy: 92%+"
echo "  Weak Pairs (1v2, 3v4): 77-79% (Paper's results)"
echo "  Strong Pairs (0v3, 0v4): 97-99% (Paper's results)"
echo ""
echo "📈 TRACKING IMPROVEMENTS:"
echo "  ✅ Train/Val losses saved (overfitting detection)"
echo "  ✅ Test accuracy tracked"
echo "  ✅ Confusion matrix saved"
echo "  ✅ Per-class metrics recorded"
echo ""

# Create output directory
mkdir -p ./mobilenet_paper_replication_results

echo "🚀 Starting Paper Replication Training..."
echo "========================================================================"
echo ""

# Train with PAPER'S EXACT CONFIGURATION
python3 ensemble_5class_trainer.py \
    --mode train \
    --dataset_path ./dataset_eyepacs_5class_balanced \
    --output_dir ./mobilenet_paper_replication_results \
    --experiment_name "paper_replication_mobilenet_v2_ovo" \
    \
    `# Model Configuration (Paper's Best)` \
    --base_models mobilenet_v2 \
    --num_classes 5 \
    --freeze_weights false \
    --ovo_dropout 0.5 \
    \
    `# Image & Batch (Paper's Settings)` \
    --img_size 224 \
    --batch_size 32 \
    \
    `# Training Hyperparameters (Paper's Exact Values)` \
    --epochs 50 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --patience 15 \
    --early_stopping_patience 10 \
    \
    `# Preprocessing (Paper's Simple Approach)` \
    --rotation_range 45.0 \
    --brightness_range 0.2 \
    --contrast_range 0.2 \
    --zoom_range 0.2 \
    --horizontal_flip \
    --vertical_flip \
    \
    `# Loss Function (Paper used simple Cross-Entropy)` \
    --enable_focal_loss false \
    --enable_class_weights false \
    --label_smoothing 0.0 \
    \
    `# Scheduler (Paper used Cosine)` \
    --scheduler cosine \
    --warmup_epochs 5 \
    \
    `# Validation & Checkpointing` \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --target_accuracy 0.92 \
    \
    `# System` \
    --seed 42 \
    --device cuda \
    --no_wandb

echo ""
echo "✅ Paper Replication Training Completed!"
echo "========================================================================"
echo ""
echo "📊 NEXT STEPS:"
echo "  1. Check results in: ./mobilenet_paper_replication_results/"
echo "  2. Analyze with: python3 model_analyzer.py --model ./mobilenet_paper_replication_results/models/ovo_ensemble_best.pth"
echo "  3. Compare performance:"
echo "     - Paper's MobileNet: 92.00%"
echo "     - Your EfficientNetB2: 64.20%"
echo "     - Your DenseNet121: 64.84%"
echo "     - Expected This Run: 88-92%"
echo ""
echo "📈 PERFORMANCE INDICATORS:"
echo "  • If achieving 88-92%: ✅ Paper replication SUCCESSFUL"
echo "  • If achieving 80-88%: ⚠️  Good progress, tune hyperparameters"
echo "  • If achieving 70-80%: ⚠️  Dataset quality or training issues"
echo "  • If achieving <70%: ❌ Debug required (check logs)"
echo ""
echo "🔍 TROUBLESHOOTING IF ACCURACY < 85%:"
echo "  1. Check dataset quality (gradable images)"
echo "  2. Verify class balance (should be perfect 1:1:1:1:1)"
echo "  3. Monitor training logs for early stopping"
echo "  4. Try with longer training (--epochs 80)"
echo "  5. Check GPU utilization (should be >90%)"
echo ""
echo "💡 THEORY vs PRACTICE:"
echo "  Paper used APTOS 2019 (clinical grade, 3,662 images)"
echo "  You're using EyePACS (mixed quality, 37,750 images)"
echo "  Expected: 88-92% (may differ due to dataset characteristics)"
echo ""
echo "🎯 SUCCESS CRITERIA:"
echo "  ✅ Individual pairs: >85% average"
echo "  ✅ Weak pairs (1v2, 3v4): >75%"
echo "  ✅ Strong pairs (0v3, 0v4): >95%"
echo "  ✅ Ensemble: >90% (medical grade)"
echo ""
echo "Training completed at: $(date)"
echo "========================================================================"
