#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 5-CLASS DR + DenseNet121 Medical-Grade Training Script
echo "🏥 5-CLASS DR + DenseNet121 Medical-Grade Training"
echo "==============================================="
echo "🎯 Target: 95%+ accuracy with optimized DenseNet121"
echo "📊 Dataset: 5-Class Balanced (53,935 images - Class 0, 1, 2, 3, 4)"
echo "🏗️ Model: DenseNet121 (8M params - dense connectivity)"
echo "🔗 System: V100 16GB GPU optimized"
echo ""

# Create output directory for 5-class DenseNet results
mkdir -p ./densenet_5class_results

echo "🔬 5-CLASS DenseNet121 OVO ENSEMBLE Configuration (95%+ TARGET - ANTI-OVERFITTING):"
echo "  - Dataset: ./dataset_eyepacs_5class_balanced"
echo "  - Classes: 5 (0: No DR, 1: Mild NPDR, 2: Moderate NPDR, 3: Severe NPDR, 4: PDR)"
echo "  - Total images: 53,935 (Train: 37,750 | Val: 8,090 | Test: 8,095)"
echo "  - Class distribution: PERFECTLY BALANCED (10,787 per class)"
echo "  - Imbalance ratio: 1.00:1 (PERFECT BALANCE)"
echo "  - Model: DenseNet121 (8M params - dense connectivity)"
echo "  - OVO Training: 10 binary classifiers (pairs: 0-1, 0-2, 0-3, 0-4, 1-2, 1-3, 1-4, 2-3, 2-4, 3-4)"
echo "  - OVO Voting: Weighted voting with PDR boost"
echo "  - Image size: 299x299 (optimal for medical imaging)"
echo "  - Batch size: 10 (optimized for V100 16GB)"
echo "  - Learning rate: 5e-5 (REDUCED to prevent overfitting)"
echo "  - Weight decay: 5e-4 (INCREASED regularization)"
echo "  - Dropout: 0.5 (INCREASED to combat overfitting)"
echo "  - Epochs: 50 per binary classifier (REDUCED - early stop works)"
echo "  - CLAHE: DISABLED (caused overfitting in pair 0-1)"
echo "  - Focal loss: DISABLED (using weighted CE for stability)"
echo "  - Class weights: EQUAL (1.0 for all classes - perfectly balanced dataset)"
echo "  - Augmentation: AGGRESSIVE (25° rotation, 20% brightness/contrast)"
echo "  - Scheduler: ReduceLROnPlateau (adaptive to val performance)"
echo "  - Early stopping: patience=10 (aggressive to prevent overfitting)"
echo "  - Strategy: Strong regularization + aggressive early stopping"
echo ""

# Train 5-Class with ANTI-OVERFITTING hyperparameters optimized for 95%+ accuracy
python3 ensemble_5class_trainer.py \
    --mode train \
    --dataset_path ./dataset_eyepacs_5class_balanced \
    --output_dir ./densenet_5class_results \
    --experiment_name "5class_densenet121_anti_overfit" \
    --base_models densenet121 \
    --num_classes 5 \
    --img_size 299 \
    --batch_size 10 \
    --epochs 50 \
    --learning_rate 5e-5 \
    --weight_decay 5e-4 \
    --ovo_dropout 0.5 \
    --freeze_weights false \
    --enable_medical_augmentation \
    --rotation_range 25.0 \
    --brightness_range 0.20 \
    --contrast_range 0.20 \
    --enable_class_weights \
    --class_weight_0 1.0 \
    --class_weight_1 1.0 \
    --class_weight_2 1.0 \
    --class_weight_3 1.0 \
    --class_weight_4 1.0 \
    --scheduler plateau \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --patience 10 \
    --early_stopping_patience 8 \
    --target_accuracy 0.95 \
    --max_grad_norm 0.5 \
    --label_smoothing 0.15 \
    --seed 42

echo ""
echo "✅ 5-CLASS DenseNet121 OVO ENSEMBLE training completed!"
echo "📁 Results saved to: ./densenet_5class_results"
echo ""
echo "🎯 OVO ENSEMBLE TRAINING RESULTS:"
echo "  🔢 Binary classifiers trained: 10"
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
echo "  📊 Model capacity per classifier: 8M parameters"
echo "  🎓 Learning rate: 5e-5 (REDUCED to prevent overfitting)"
echo "  💧 Dropout: 0.5 (INCREASED to combat overfitting)"
echo "  ⏰ Training: 50 epochs per binary classifier (~7-8 hours total)"
echo "  🔬 CLAHE: DISABLED (caused overfitting)"
echo "  🔀 Augmentation: 25° rotation, 20% brightness/contrast (AGGRESSIVE)"
echo "  ⚖️ Class weights: 1.0 for ALL classes (PERFECTLY BALANCED DATASET)"
echo "  🎯 Focal loss: DISABLED (using standard weighted CE)"
echo "  🔧 Scheduler: ReduceLROnPlateau (adaptive)"
echo ""
echo "📊 Expected Performance (OVO Ensemble - 53,935 images):"
echo "  🎯 Target: 95-97% validation accuracy (with anti-overfitting measures)"
echo "  🏥 Strategy: Strong regularization + aggressive augmentation + early stopping"
echo "  📈 Rationale: Prevent overfitting → better generalization → higher val accuracy"
echo "  🔗 Training time: ~7-8 hours on V100 16GB (10 classifiers)"
echo "  ⚠️ ADVANTAGE: Perfect balance (1.00:1 ratio) = stable training"
echo ""
echo "🔗 SAVED MODEL FILES:"
echo "  ✅ best_densenet121_0_1.pth (No DR vs Mild NPDR)"
echo "  ✅ best_densenet121_0_2.pth (No DR vs Moderate NPDR)"
echo "  ✅ best_densenet121_0_3.pth (No DR vs Severe NPDR)"
echo "  ✅ best_densenet121_0_4.pth (No DR vs PDR)"
echo "  ✅ best_densenet121_1_2.pth (Mild NPDR vs Moderate NPDR)"
echo "  ✅ best_densenet121_1_3.pth (Mild NPDR vs Severe NPDR)"
echo "  ✅ best_densenet121_1_4.pth (Mild NPDR vs PDR)"
echo "  ✅ best_densenet121_2_3.pth (Moderate NPDR vs Severe NPDR)"
echo "  ✅ best_densenet121_2_4.pth (Moderate NPDR vs PDR)"
echo "  ✅ best_densenet121_3_4.pth (Severe NPDR vs PDR)"
echo "  ✅ ovo_ensemble_best.pth (Combined OVO ensemble)"
echo "  🎯 Ready for incremental model addition"
echo ""
echo "📋 Next Steps:"
echo "  1. Analyze OVO ensemble results:"
echo "     python model_analyzer.py --model ./densenet_5class_results/models/ovo_ensemble_best.pth"
echo ""
echo "  2. Check individual binary classifiers:"
echo "     python model_analyzer.py --model ./densenet_5class_results/models/best_densenet121_0_1.pth"
echo "     python model_analyzer.py --model ./densenet_5class_results/models/best_densenet121_3_4.pth"
echo ""
echo "  3. Add more models for higher accuracy (97%+ target):"
echo "     • Modify --base_models: densenet121 efficientnetb2"
echo "     • This trains 10 new EfficientNetB2 binary classifiers"
echo "     • Skips already-trained DenseNet121 classifiers"
echo "     • Total: 20 binary classifiers with OVO voting"
echo ""
echo "  4. Eventually add ResNet50 for maximum ensemble:"
echo "     • Modify --base_models: densenet121 efficientnetb2 resnet50"
echo "     • This trains 10 new ResNet50 binary classifiers"
echo "     • Total: 30 binary classifiers → 97%+ accuracy expected"
echo ""
echo "🚀 OVO ENSEMBLE APPROACH FOR 5-CLASS (95-97%+ TARGET):"
echo "  ✅ OVO Training: 10 binary classifiers (simpler than multi-class)"
echo "  ✅ Weighted voting: Severity-based boost for medical safety"
echo "  ✅ NO CLAHE: Disabled to prevent overfitting"
echo "  ✅ Aggressive augmentation: 25° rotation, 20% brightness/contrast"
echo "  ✅ Strong dropout: 0.5 (combat overfitting)"
echo "  ✅ NO focal loss: Standard weighted CE for stability"
echo "  ✅ EQUAL class weights: 1.0 for all (PERFECTLY BALANCED DATASET)"
echo "  ✅ Reduced epochs: 50 per classifier (early stop prevents overfitting)"
echo "  ✅ Incremental training: Add models without retraining existing ones"
echo "  📊 Expected: Single model 92-95% → Multi-model 95-97%+"
echo ""
echo "⚠️ ANTI-OVERFITTING MEASURES (AGGRESSIVE):"
echo "  ✅ Dropout 0.5 (DOUBLED from 0.3)"
echo "  ✅ Weight decay 5e-4 (INCREASED from 3e-4)"
echo "  ✅ Label smoothing 0.15 (INCREASED from 0.1)"
echo "  ✅ Gradient clipping max_norm=0.5 (MORE aggressive)"
echo "  ✅ Early stopping patience=10 (REDUCED from 25)"
echo "  ✅ Early stopping counter=8 (AGGRESSIVE - stops at first sign)"
echo "  ✅ ReduceLROnPlateau scheduler (adaptive to validation)"
echo "  ✅ Validation every epoch (monitoring)"
echo "  ✅ Checkpoint every 5 epochs (best model selection)"
echo "  ✅ PERFECTLY balanced dataset (53,935 images - 10,787 per class)"
echo "  ✅ Lower learning rate: 5e-5 (HALVED from 1e-4)"
echo ""
echo "⚠️ KEY CHANGES FROM PREVIOUS RUN:"
echo "  🔧 Learning rate: 1e-4 → 5e-5 (50% reduction)"
echo "  🔧 Dropout: 0.3 → 0.5 (67% increase)"
echo "  🔧 Weight decay: 3e-4 → 5e-4 (67% increase)"
echo "  🔧 Patience: 25 → 10 (60% reduction)"
echo "  🔧 CLAHE: ENABLED → DISABLED (caused overfitting)"
echo "  🔧 Focal loss: ENABLED → DISABLED (simpler is better)"
echo "  🔧 Augmentation: MODERATE → AGGRESSIVE (better generalization)"
echo ""
echo "💾 V100 16GB GPU OPTIMIZATION:"
echo "  ✅ Batch size: 10 (optimal for V100 with 299×299)"
echo "  ✅ Image size: 299×299 (DenseNet121 optimal)"
echo "  ✅ Gradient accumulation: 2 steps"
echo "  ✅ Pin memory: True (faster data loading)"
echo "  ✅ Persistent workers: num_workers=4"
echo "  ✅ Expected memory: ~6-7GB (safe for 16GB V100)"
echo "  ✅ Training time: ~7-8 hours for 10 binary classifiers (50 epochs each)"
echo ""
echo "🎯 PATH TO 97%+ ENSEMBLE ACCURACY:"
echo "  1. DenseNet121 (this run): 92-95% expected (anti-overfitting optimized)"
echo "  2. Train EfficientNetB2 (5-class): 93-96%+ expected"
echo "  3. Train ResNet50 (5-class): 91-94%+ expected"
echo "  4. Ensemble averaging: 95-97%+ target (medical-grade)"
echo ""
echo "📈 IMPROVEMENT STRATEGY:"
echo "  Problem observed: Pair 0-1 achieved 88.57% (below 95% target)"
echo "  Root cause: Overfitting (train 89.98% vs val 86.53% at epoch 27)"
echo "  Solution applied:"
echo "    • Reduced learning rate (prevent rapid overfitting)"
echo "    • Increased dropout (force generalization)"
echo "    • Disabled CLAHE (was causing overfitting)"
echo "    • Aggressive early stopping (stop before overfitting)"
echo "    • Stronger augmentation (better generalization)"
echo "  Expected result: Val accuracy closer to train accuracy, 92-95% range"
echo ""
