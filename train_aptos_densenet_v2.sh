#!/bin/bash

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

echo "🔬 APTOS 2019 DenseNet121 OPTIMIZED Configuration:"
echo "  - Dataset: APTOS 2019 (./dataset_aptos2019) - IMBALANCED"
echo "  - Model: DenseNet121 (8M params - dense connectivity)"
echo "  - Image size: 299x299 (same as EfficientNetB2)"
echo "  - Batch size: 10 (optimized for V100)"
echo "  - Learning rate: 1e-4 (stable proven rate)"
echo "  - Weight decay: 3e-4 (balanced regularization)"
echo "  - Dropout: 0.2 (consistent with EfficientNetB2)"
echo "  - Epochs: 80 (early stopping enabled)"
echo "  - Focal loss: alpha=2.5, gamma=3.0 (moderate - like EfficientNetB2)"
echo "  - Class weights: 10x severe, 12x PDR (balanced for APTOS)"
echo "  - Scheduler: Cosine with warm restarts (T_0=15)"
echo "  - Target: 80-84% accuracy (ensemble component)"
echo ""

# Train APTOS 2019 with optimized hyperparameters for class imbalance
python ensemble_local_trainer.py \
    --mode train \
    --dataset_path ./dataset_aptos2019 \
    --output_dir ./densenet_aptos_results \
    --experiment_name "aptos2019_densenet121_optimized" \
    --base_models densenet121 \
    --img_size 299 \
    --batch_size 10 \
    --epochs 80 \
    --learning_rate 1e-4 \
    --weight_decay 3e-4 \
    --ovo_dropout 0.2 \
    --freeze_weights false \
    --enable_medical_augmentation \
    --rotation_range 20.0 \
    --brightness_range 0.15 \
    --contrast_range 0.15 \
    --enable_focal_loss \
    --enable_class_weights \
    --class_weight_severe 10.0 \
    --class_weight_pdr 12.0 \
    --focal_loss_alpha 2.5 \
    --focal_loss_gamma 3.0 \
    --scheduler cosine \
    --warmup_epochs 8 \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --patience 20 \
    --early_stopping_patience 15 \
    --target_accuracy 0.84 \
    --max_grad_norm 1.0 \
    --label_smoothing 0.1 \
    --seed 42

echo ""
echo "✅ APTOS 2019 DenseNet121 training completed!"
echo "📁 Results saved to: ./densenet_aptos_results"
echo ""
echo "🎯 APTOS 2019 OPTIMIZATION Applied:"
echo "  🏗️ Architecture: DenseNet121 (dense connectivity)"
echo "  📊 Model capacity: 8M parameters"
echo "  🎓 Learning rate: 1e-4 (same as EfficientNetB2)"
echo "  💧 Dropout: 0.2 (consistent across models)"
echo "  ⏰ Training: 80 epochs (early stopping enabled)"
echo "  🔀 Augmentation: 20° rotation, 15% brightness/contrast"
echo "  ⚖️ Class weights: 10x severe, 12x PDR (moderate for APTOS)"
echo "  🎯 Focal loss: alpha=2.5, gamma=3.0 (moderate)"
echo "  🔧 Scheduler: Cosine with warm restarts"
echo ""
echo "📊 Expected Performance (APTOS 2019 - 3,657 images):"
echo "  🎯 Target: 80-84% validation accuracy"
echo "  🏥 Medical grade: ⚠️ Below 90% (but good for ensemble)"
echo "  📈 Dense connectivity: Helps with feature reuse"
echo "  🔗 Training time: ~2-3 hours on V100"
echo "  ✅ Ensemble diversity: Different architecture from EfficientNetB2"
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
echo "     python model_analyzer.py --model ./densenet_aptos_results/models/best_densenet121_multiclass.pth"
echo ""
echo "  2. Create 2-model CNN ensemble (if both ready):"
echo "     • EfficientNetB2: 82.05%"
echo "     • DenseNet121: ~81-83% (expected)"
echo "     • 2-model ensemble: ~83-85%"
echo ""
echo "  3. RECOMMENDED: Add MedSigLIP-448 for 90% target:"
echo "     • 3-model ensemble: 87-90%"
echo ""
echo "🚀 ENSEMBLE STRATEGY:"
echo "  📊 Current: EfficientNetB2 (82.05%)"
echo "  📊 After this: + DenseNet121 (~81-83%)"
echo "  ⭐ Next: + MedSigLIP-448 (~84-86%) → 90% goal!"
echo ""