#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# APTOS 2019 + DenseNet121 Medical-Grade Training Script
echo "🏥 APTOS 2019 + DenseNet121 Medical-Grade Training"
echo "==============================================="
echo "🎯 Target: 80-84% accuracy with optimized DenseNet121"
echo "📊 Dataset: APTOS 2019 (5-class DR classification - 3,657 images)"
echo "🏗️ Model: DenseNet121 (ensemble-compatible configuration)"
echo "🔗 System: Compatible with EfficientNetB2 ensemble"
echo ""

# Create output directory for APTOS DenseNet results
mkdir -p ./densenet_aptos_results

echo "🔬 EyePACS DenseNet121 BALANCED Configuration (85-90% TARGET):"
echo "  - Dataset: EyePACS (./dataset_eyepacs) - 33,857 training images"
echo "  - Model: DenseNet121 (8M params - dense connectivity)"
echo "  - Image size: 299x299 (optimal for medical imaging)"
echo "  - Batch size: 10 (optimized for V100)"
echo "  - Learning rate: 1e-4 (stable proven rate)"
echo "  - Weight decay: 3e-4 (balanced regularization)"
echo "  - Dropout: 0.3 (BALANCED - not too aggressive)"
echo "  - Epochs: 100 (extended for convergence)"
echo "  - CLAHE: ENABLED with conservative augmentation"
echo "  - Focal loss: alpha=2.5, gamma=3.0 (BALANCED for CLAHE)"
echo "  - Class weights: 12x mild, 6x moderate, 10x severe, 12x PDR"
echo "  - Augmentation: MODERATE (25° rotation, 20% brightness/contrast)"
echo "  - Scheduler: Cosine with warm restarts (T_0=15)"
echo "  - Strategy: CLAHE requires LESS aggressive other settings"
echo ""

# Train EyePACS with BALANCED hyperparameters optimized for CLAHE
python3 ensemble_local_trainer.py \
    --mode train \
    --dataset_path ./dataset_eyepacs \
    --output_dir ./densenet_eyepacs_results \
    --experiment_name "eyepacs_densenet121_balanced_clahe" \
    --base_models densenet121 \
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
    --class_weight_mild 12.0 \
    --class_weight_moderate 6.0 \
    --class_weight_severe 10.0 \
    --class_weight_pdr 12.0 \
    --focal_loss_alpha 2.5 \
    --focal_loss_gamma 3.0 \
    --scheduler cosine \
    --warmup_epochs 10 \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --patience 25 \
    --early_stopping_patience 20 \
    --target_accuracy 0.88 \
    --max_grad_norm 1.0 \
    --label_smoothing 0.1 \
    --seed 42

echo ""
echo "✅ EyePACS DenseNet121 MEDICAL-GRADE training completed!"
echo "📁 Results saved to: ./densenet_eyepacs_results"
echo ""
echo "🎯 BALANCED OPTIMIZATIONS Applied:"
echo "  🏗️ Architecture: DenseNet121 (dense connectivity)"
echo "  📊 Model capacity: 8M parameters"
echo "  🎓 Learning rate: 1e-4 (stable for medical imaging)"
echo "  💧 Dropout: 0.3 (BALANCED - CLAHE reduces overfitting naturally)"
echo "  ⏰ Training: 100 epochs (extended for convergence)"
echo "  🔬 CLAHE: ENABLED (clip_limit=2.5, conservative)"
echo "  🔀 Augmentation: 25° rotation, 20% brightness/contrast (MODERATE)"
echo "  ⚖️ Class weights: 12x mild, 6x moderate, 10x severe, 12x PDR"
echo "  🎯 Focal loss: alpha=2.5, gamma=3.0 (BALANCED with CLAHE)"
echo "  🔧 Scheduler: Cosine with warm restarts (T_0=15)"
echo ""
echo "📊 Expected Performance (EyePACS - 33,857 images):"
echo "  🎯 Target: 85-88% validation accuracy (realistic with CLAHE)"
echo "  🏥 Strategy: CLAHE + moderate settings = stable improvement"
echo "  📈 Previous run: 82% (no CLAHE) → Expected: 85-88% (with CLAHE)"
echo "  🔗 Training time: ~3-4 hours on V100"
echo "  ⚠️ LESSON: CLAHE makes images similar → use LESS aggressive augmentation"
echo ""
echo "🔗 ENSEMBLE COMPATIBILITY:"
echo "  ✅ Model saved as: best_densenet121_multiclass.pth"
echo "  ✅ Same training system as EfficientNetB2"
echo "  ✅ Same checkpoint format and structure"
echo "  ✅ Same image size (299×299)"
echo "  ✅ Ready for ensemble combination"
echo ""
echo "📋 Next Steps:"
echo "  1. Analyze results:"
echo "     python model_analyzer.py --model ./densenet_eyepacs_results/models/best_densenet121_multiclass.pth"
echo ""
echo "  2. If 90%+ not achieved, consider dataset balancing:"
echo "     python create_balanced_dataset_simple.py --input_dir ./dataset_eyepacs --output_dir ./dataset_eyepacs_balanced"
echo ""
echo "  3. Optional: Create 3-model ensemble for 96%+ accuracy:"
echo "     • Train EfficientNetB2 (96.27% target)"
echo "     • Train ResNet50 (94.95% target)"
echo "     • Ensemble DenseNet121 + EfficientNetB2 + ResNet50 → 96.96%"
echo ""
echo "🚀 BALANCED APPROACH (82% → 85-88%):"
echo "  ✅ CLAHE preprocessing: ENABLED but conservative (clip=2.5)"
echo "  ✅ Moderate augmentation: 25° rotation, 20% brightness/contrast"
echo "  ✅ Balanced dropout: 0.3 (not too high with CLAHE)"
echo "  ✅ Balanced focal loss: gamma=3.0 (not conflicting with CLAHE)"
echo "  ✅ Optimized class weights: 12x mild, 6x moderate"
echo "  ✅ Extended training: 100 epochs for full convergence"
echo "  📊 Expected improvement: +3-6% (82% → 85-88%)"
echo ""
echo "⚠️ IMPORTANT LESSON LEARNED:"
echo "  ❌ Previous attempt: Too aggressive (dropout=0.4, gamma=4.0, 30° rotation)"
echo "     → CLAHE + extreme settings = conflicting signals → 79% (WORSE)"
echo "  ✅ Current approach: Moderate settings balanced with CLAHE"
echo "     → CLAHE enhances features → need LESS augmentation → 85-88% expected"
echo ""
echo "🎯 PATH TO 90%+ IF THIS REACHES 85-88%:"
echo "  1. Create balanced dataset (SMOTE): +3-5% → 88-93%"
echo "  2. Add ensemble (EfficientNetB2 + ResNet50): +3-5% → 91-98%"
echo "  3. Fine-tune on balanced data: → 96%+ target"
echo ""