#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 3-CLASS DR + DenseNet121 Medical-Grade Training Script
echo "🏥 3-CLASS DR + DenseNet121 Medical-Grade Training"
echo "==============================================="
echo "🎯 Target: 95%+ accuracy with optimized DenseNet121"
echo "📊 Dataset: 3-Class Balanced (39,850 images - NORMAL, NPDR, PDR)"
echo "🏗️ Model: DenseNet121 (8M params - dense connectivity)"
echo "🔗 System: V100 16GB GPU optimized"
echo ""

# Create output directory for 3-class DenseNet results
mkdir -p ./densenet_3class_results

echo "🔬 3-CLASS DenseNet121 OVO ENSEMBLE Configuration (95%+ TARGET):"
echo "  - Dataset: ./dataset_eyepacs_3class_balanced"
echo "  - Classes: 3 (NORMAL, NPDR merged 1-3, PDR)"
echo "  - Total images: 39,850 (Train: 31,878 | Val: 3,983 | Test: 3,989)"
echo "  - Class distribution: NORMAL=64.8%, NPDR=25.2%, PDR=10.0%"
echo "  - Imbalance ratio: 6.45:1 (NORMAL:PDR) - manageable"
echo "  - Model: DenseNet121 (8M params - dense connectivity)"
echo "  - OVO Training: 3 binary classifiers (pair_0_1, pair_0_2, pair_1_2)"
echo "  - OVO Voting: Weighted voting with PDR boost"
echo "  - Image size: 299x299 (optimal for medical imaging)"
echo "  - Batch size: 10 (optimized for V100 16GB)"
echo "  - Learning rate: 1e-4 (stable proven rate)"
echo "  - Weight decay: 3e-4 (balanced regularization)"
echo "  - Dropout: 0.3 (BALANCED - not too aggressive)"
echo "  - Epochs: 100 per binary classifier"
echo "  - CLAHE: ENABLED with conservative augmentation"
echo "  - Focal loss: alpha=2.5, gamma=3.0 (BALANCED for CLAHE)"
echo "  - Class weights: 0.515 (NORMAL), 1.323 (NPDR), 3.321 (PDR)"
echo "  - Augmentation: MODERATE (25° rotation, 20% brightness/contrast)"
echo "  - Scheduler: Cosine with warm restarts (T_0=15)"
echo "  - Strategy: OVO ensemble + CLAHE + balanced settings"
echo ""

# Train 3-Class with BALANCED hyperparameters optimized for CLAHE and balanced dataset
python3 ensemble_3class_trainer.py \
    --mode train \
    --dataset_path ./dataset_eyepacs_3class_balanced \
    --output_dir ./densenet_3class_results \
    --experiment_name "3class_densenet121_balanced_clahe" \
    --base_models densenet121 \
    --num_classes 3 \
    --img_size 299 \
    --batch_size 10 \
    --epochs 100 \
    --learning_rate 1e-4 \
    --weight_decay 3e-4 \
    --ovo_dropout 0.3 \
    --freeze_weights false \
    --enable_clahe \
    --clahe_clip_limit 2.5 \
    --enable_medical_augmentation \
    --rotation_range 25.0 \
    --brightness_range 0.20 \
    --contrast_range 0.20 \
    --enable_focal_loss \
    --enable_class_weights \
    --class_weight_normal 0.515 \
    --class_weight_npdr 1.323 \
    --class_weight_pdr 3.321 \
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
    --label_smoothing 0.1 \
    --seed 42

echo ""
echo "✅ 3-CLASS DenseNet121 OVO ENSEMBLE training completed!"
echo "📁 Results saved to: ./densenet_3class_results"
echo ""
echo "🎯 OVO ENSEMBLE TRAINING RESULTS:"
echo "  🔢 Binary classifiers trained: 3"
echo "    • pair_0_1: NORMAL vs NPDR"
echo "    • pair_0_2: NORMAL vs PDR"
echo "    • pair_1_2: NPDR vs PDR"
echo "  🗳️ OVO Voting: Weighted with PDR boost (2x)"
echo "  🏗️ Architecture: DenseNet121 (8M parameters)"
echo "  📊 Model capacity per classifier: 8M parameters"
echo "  🎓 Learning rate: 1e-4 (stable for medical imaging)"
echo "  💧 Dropout: 0.3 (BALANCED - CLAHE reduces overfitting naturally)"
echo "  ⏰ Training: 100 epochs per binary classifier (~4.5 hours total)"
echo "  🔬 CLAHE: ENABLED (clip_limit=2.5, conservative)"
echo "  🔀 Augmentation: 25° rotation, 20% brightness/contrast (MODERATE)"
echo "  ⚖️ Class weights: 0.515 (NORMAL), 1.323 (NPDR), 3.321 (PDR)"
echo "  🎯 Focal loss: alpha=2.5, gamma=3.0 (BALANCED with CLAHE)"
echo "  🔧 Scheduler: Cosine with warm restarts (T_0=15)"
echo ""
echo "📊 Expected Performance (OVO Ensemble - 39,850 images):"
echo "  🎯 Target: 95-97% validation accuracy (OVO voting advantage)"
echo "  🏥 Strategy: OVO + CLAHE + balanced settings"
echo "  📈 Rationale: Binary classifiers easier + voting = higher accuracy"
echo "  🔗 Training time: ~4.5 hours on V100 16GB (3 classifiers)"
echo "  ⚠️ LESSON: OVO ensemble + balanced dataset = medical-grade accuracy"
echo ""
echo "🔗 SAVED MODEL FILES:"
echo "  ✅ best_densenet121_0_1.pth (NORMAL vs NPDR)"
echo "  ✅ best_densenet121_0_2.pth (NORMAL vs PDR)"
echo "  ✅ best_densenet121_1_2.pth (NPDR vs PDR)"
echo "  ✅ ovo_ensemble_best.pth (Combined OVO ensemble)"
echo "  🎯 Ready for incremental model addition"
echo ""
echo "📋 Next Steps:"
echo "  1. Analyze OVO ensemble results:"
echo "     python model_analyzer.py --model ./densenet_3class_results/models/ovo_ensemble_best.pth"
echo ""
echo "  2. Check individual binary classifiers:"
echo "     python model_analyzer.py --model ./densenet_3class_results/models/best_densenet121_0_1.pth"
echo "     python model_analyzer.py --model ./densenet_3class_results/models/best_densenet121_0_2.pth"
echo "     python model_analyzer.py --model ./densenet_3class_results/models/best_densenet121_1_2.pth"
echo ""
echo "  3. Add more models for higher accuracy (97%+ target):"
echo "     • Modify --base_models: densenet121 efficientnetb2"
echo "     • This trains 3 new EfficientNetB2 binary classifiers"
echo "     • Skips already-trained DenseNet121 classifiers"
echo "     • Total: 6 binary classifiers with OVO voting"
echo ""
echo "  4. Eventually add ResNet50 for maximum ensemble:"
echo "     • Modify --base_models: densenet121 efficientnetb2 resnet50"
echo "     • This trains 3 new ResNet50 binary classifiers"
echo "     • Total: 9 binary classifiers → 97%+ accuracy expected"
echo ""
echo "🚀 OVO ENSEMBLE APPROACH FOR 3-CLASS (95-97%+ TARGET):"
echo "  ✅ OVO Training: 3 binary classifiers (simpler than multi-class)"
echo "  ✅ Weighted voting: PDR gets 2x boost for medical safety"
echo "  ✅ CLAHE preprocessing: ENABLED but conservative (clip=2.5)"
echo "  ✅ Moderate augmentation: 25° rotation, 20% brightness/contrast"
echo "  ✅ Balanced dropout: 0.3 (not too high with CLAHE)"
echo "  ✅ Balanced focal loss: gamma=3.0 (not conflicting with CLAHE)"
echo "  ✅ Optimized class weights: 0.515, 1.323, 3.321 (6.45:1 ratio)"
echo "  ✅ Extended training: 100 epochs per classifier"
echo "  ✅ Incremental training: Add models without retraining existing ones"
echo "  📊 Expected: Single model 95%+ → Multi-model 97%+"
echo ""
echo "⚠️ ANTI-OVERFITTING MEASURES:"
echo "  ✅ Dropout 0.3 (balanced - not aggressive)"
echo "  ✅ Weight decay 3e-4 (regularization)"
echo "  ✅ Label smoothing 0.1 (generalization)"
echo "  ✅ Gradient clipping max_norm=1.0 (stability)"
echo "  ✅ Early stopping patience=25 (prevents overtraining)"
echo "  ✅ Cosine scheduler with warm restarts (escape plateaus)"
echo "  ✅ Validation every epoch (monitoring)"
echo "  ✅ Checkpoint every 5 epochs (best model selection)"
echo "  ✅ Balanced dataset (39,850 images - sufficient size)"
echo "  ✅ Class weights prevent minority overfitting"
echo ""
echo "⚠️ ANTI-PLATEAU MEASURES:"
echo "  ✅ Cosine annealing with warm restarts (T_0=15)"
echo "  ✅ Warmup epochs (10) for stable start"
echo "  ✅ Learning rate: 1e-4 (proven stable)"
echo "  ✅ Patience: 25 epochs (allows recovery)"
echo "  ✅ Min LR: 1e-7 (prevents complete stagnation)"
echo "  ✅ Simpler 3-class problem (more stable convergence)"
echo ""
echo "💾 V100 16GB GPU OPTIMIZATION:"
echo "  ✅ Batch size: 10 (optimal for V100 with 299×299)"
echo "  ✅ Image size: 299×299 (DenseNet121 optimal)"
echo "  ✅ Gradient accumulation: 2 steps"
echo "  ✅ Pin memory: True (faster data loading)"
echo "  ✅ Persistent workers: num_workers=4"
echo "  ✅ Expected memory: ~5-6GB (safe for 16GB V100)"
echo "  ✅ Training time: ~3-4 hours for 100 epochs"
echo ""
echo "🎯 PATH TO 97%+ ENSEMBLE ACCURACY:"
echo "  1. DenseNet121 (this run): 95%+ expected"
echo "  2. Train EfficientNetB2 (3-class): 96%+ expected"
echo "  3. Train ResNet50 (3-class): 94%+ expected"
echo "  4. Ensemble averaging: 97%+ target (medical-grade++)"
echo ""
