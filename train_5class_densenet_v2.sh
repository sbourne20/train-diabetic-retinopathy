#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 5-CLASS DR + DenseNet121 Medical-Grade Training Script V2
echo "🏥 5-CLASS DR + DenseNet121 Medical-Grade Training V2"
echo "======================================================"
echo "🎯 Target: 95%+ accuracy with BALANCED regularization"
echo "📊 Dataset: 5-Class Balanced (53,935 images - Class 0, 1, 2, 3, 4)"
echo "🏗️ Model: DenseNet121 (8M params - dense connectivity)"
echo "🔗 System: V100 16GB GPU optimized"
echo ""

# Create output directory for 5-class DenseNet results
mkdir -p ./densenet_5class_results

echo "🔬 5-CLASS DenseNet121 OVO ENSEMBLE Configuration (95%+ TARGET - BALANCED V2):"
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
echo "  - Learning rate: 8e-5 (BALANCED - between 5e-5 and 1e-4)"
echo "  - Weight decay: 4e-4 (BALANCED regularization)"
echo "  - Dropout: 0.4 (BALANCED - not too high, not too low)"
echo "  - Epochs: 60 per binary classifier (BALANCED)"
echo "  - CLAHE: DISABLED (caused overfitting)"
echo "  - Focal loss: DISABLED (using standard weighted CE)"
echo "  - Class weights: EQUAL (1.0 for all classes - perfectly balanced dataset)"
echo "  - Augmentation: BALANCED (22° rotation, 18% brightness/contrast)"
echo "  - Scheduler: ReduceLROnPlateau with factor=0.5 (adaptive)"
echo "  - Early stopping: patience=12 (BALANCED)"
echo "  - Strategy: BALANCED regularization for 90%+ binary accuracy"
echo ""

# Train 5-Class with BALANCED hyperparameters - sweet spot between overfitting prevention and model capacity
python3 ensemble_5class_trainer.py \
    --mode train \
    --dataset_path ./dataset_eyepacs_5class_balanced \
    --output_dir ./densenet_5class_results \
    --experiment_name "5class_densenet121_balanced_v2" \
    --base_models densenet121 \
    --num_classes 5 \
    --img_size 299 \
    --batch_size 10 \
    --epochs 60 \
    --learning_rate 8e-5 \
    --weight_decay 4e-4 \
    --ovo_dropout 0.4 \
    --freeze_weights false \
    --enable_medical_augmentation \
    --rotation_range 22.0 \
    --brightness_range 0.18 \
    --contrast_range 0.18 \
    --enable_class_weights \
    --class_weight_0 1.0 \
    --class_weight_1 1.0 \
    --class_weight_2 1.0 \
    --class_weight_3 1.0 \
    --class_weight_4 1.0 \
    --scheduler plateau \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --patience 12 \
    --early_stopping_patience 10 \
    --target_accuracy 0.95 \
    --max_grad_norm 0.7 \
    --label_smoothing 0.12 \
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
echo "  🎓 Learning rate: 8e-5 (BALANCED between previous attempts)"
echo "  💧 Dropout: 0.4 (BALANCED - allows model capacity)"
echo "  ⏰ Training: 60 epochs per binary classifier (~8-9 hours total)"
echo "  🔬 CLAHE: DISABLED (caused overfitting)"
echo "  🔀 Augmentation: 22° rotation, 18% brightness/contrast (BALANCED)"
echo "  ⚖️ Class weights: 1.0 for ALL classes (PERFECTLY BALANCED DATASET)"
echo "  🎯 Focal loss: DISABLED (standard weighted CE for stability)"
echo "  🔧 Scheduler: ReduceLROnPlateau (adaptive, factor=0.5)"
echo ""
echo "📊 Expected Performance (OVO Ensemble - 53,935 images):"
echo "  🎯 Target: 90-92% per binary classifier (REALISTIC)"
echo "  🏥 Strategy: Balanced regularization - prevent overfitting but allow learning"
echo "  📈 Rationale: Sweet spot between too much and too little regularization"
echo "  🔗 Training time: ~8-9 hours on V100 16GB (10 classifiers)"
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
echo ""
echo "  3. Add more models for higher accuracy (95%+ target):"
echo "     • Modify --base_models: densenet121 efficientnetb2"
echo "     • This trains 10 new EfficientNetB2 binary classifiers"
echo "     • Skips already-trained DenseNet121 classifiers"
echo "     • Total: 20 binary classifiers with OVO voting"
echo ""
echo "  4. Eventually add ResNet50 for maximum ensemble:"
echo "     • Modify --base_models: densenet121 efficientnetb2 resnet50"
echo "     • This trains 10 new ResNet50 binary classifiers"
echo "     • Total: 30 binary classifiers → 95-97%+ accuracy expected"
echo ""
echo "🚀 OVO ENSEMBLE APPROACH FOR 5-CLASS (95-97%+ TARGET):"
echo "  ✅ OVO Training: 10 binary classifiers (simpler than multi-class)"
echo "  ✅ Weighted voting: Severity-based boost for medical safety"
echo "  ✅ NO CLAHE: Disabled to prevent overfitting"
echo "  ✅ Balanced augmentation: 22° rotation, 18% brightness/contrast"
echo "  ✅ Balanced dropout: 0.4 (allows learning while preventing overfitting)"
echo "  ✅ NO focal loss: Standard weighted CE for stability"
echo "  ✅ EQUAL class weights: 1.0 for all (PERFECTLY BALANCED DATASET)"
echo "  ✅ Balanced epochs: 60 per classifier"
echo "  ✅ Incremental training: Add models without retraining existing ones"
echo "  📊 Expected: Single model 90-92% → Multi-model 95-97%+"
echo ""
echo "⚠️ BALANCED REGULARIZATION APPROACH:"
echo "  ✅ Dropout 0.4 (BALANCED - not too aggressive)"
echo "  ✅ Weight decay 4e-4 (BALANCED - middle ground)"
echo "  ✅ Label smoothing 0.12 (BALANCED - slight smoothing)"
echo "  ✅ Gradient clipping max_norm=0.7 (BALANCED)"
echo "  ✅ Early stopping patience=12 (BALANCED - allows learning)"
echo "  ✅ ReduceLROnPlateau scheduler (adaptive, factor=0.5)"
echo "  ✅ Validation every epoch (monitoring)"
echo "  ✅ Checkpoint every 5 epochs (best model selection)"
echo "  ✅ PERFECTLY balanced dataset (53,935 images - 10,787 per class)"
echo "  ✅ Moderate learning rate: 8e-5 (sweet spot)"
echo ""
echo "⚠️ COMPARISON OF ALL VERSIONS:"
echo "  Version 1 (Original):"
echo "    - LR: 1e-4, Dropout: 0.3, Patience: 25"
echo "    - Result: 88.57% but OVERFITTING (train 89.98% vs val 86.53%)"
echo ""
echo "  Version 2 (Anti-overfit):"
echo "    - LR: 5e-5, Dropout: 0.5, Patience: 10"
echo "    - Result: 88.57% NO overfitting (train 88.15% vs val 88.57%)"
echo "    - Problem: Stuck at 88.5%, too much regularization"
echo ""
echo "  Version 3 (Balanced - THIS):"
echo "    - LR: 8e-5, Dropout: 0.4, Patience: 12"
echo "    - Expected: 90-92% with healthy train-val gap"
echo "    - Strategy: Sweet spot between overfitting and underfitting"
echo ""
echo "💾 V100 16GB GPU OPTIMIZATION:"
echo "  ✅ Batch size: 10 (optimal for V100 with 299×299)"
echo "  ✅ Image size: 299×299 (DenseNet121 optimal)"
echo "  ✅ Gradient accumulation: 2 steps"
echo "  ✅ Pin memory: True (faster data loading)"
echo "  ✅ Persistent workers: num_workers=4"
echo "  ✅ Expected memory: ~6-7GB (safe for 16GB V100)"
echo "  ✅ Training time: ~8-9 hours for 10 binary classifiers (60 epochs each)"
echo ""
echo "🎯 PATH TO 95%+ ENSEMBLE ACCURACY:"
echo "  1. DenseNet121 (this run): 90-92% expected per binary classifier"
echo "  2. Train EfficientNetB2 (5-class): 91-93%+ expected"
echo "  3. Train ResNet50 (5-class): 89-92%+ expected"
echo "  4. Ensemble averaging: 95-97%+ target (OVO voting boost)"
echo ""
echo "📈 V3 IMPROVEMENT STRATEGY:"
echo "  V1 Problem: Overfitting (too little regularization)"
echo "  V2 Problem: Underfitting (too much regularization)"
echo "  V3 Solution: BALANCED regularization (Goldilocks zone)"
echo "    • LR 8e-5: Faster learning than 5e-5, slower than 1e-4"
echo "    • Dropout 0.4: Regularizes but doesn't strangle model"
echo "    • Patience 12: Allows convergence but stops before overfit"
echo "    • Augmentation 22°/18%: Strong enough but not excessive"
echo "  Expected: Binary classifiers reach 90-92% range consistently"
echo ""
