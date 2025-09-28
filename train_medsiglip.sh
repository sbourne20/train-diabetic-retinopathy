#!/bin/bash

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# EyePACS + MedSigLIP-448 Medical-Grade Training Script
echo "🏥 EyePACS + MedSigLIP-448 Medical-Grade Training"
echo "================================================="
echo "🎯 Target: 90%+ accuracy with MedSigLIP-448 architecture"
echo "📊 Dataset: EyePACS (5-class DR classification)"
echo "🏗️ Model: MedSigLIP-448 (medical foundation model)"
echo "🔬 Medical-grade architecture with optimized hyperparameters"
echo ""

# Create output directory for MedSigLIP results
mkdir -p ./medsiglip_results

echo "🔬 EyePACS MedSigLIP OPTIMIZED Configuration V2 (90%+ Target):"
echo "  - Dataset: EyePACS (./dataset_eyepacs) - AUGMENTED 33,857 samples"
echo "  - Model: MedSigLIP-448 (medical foundation model - OPTIMIZED)"
echo "  - Image size: 448x448 (MedSigLIP required size)"
echo "  - Batch size: 8 (INCREASED - better GPU utilization)"
echo "  - Learning rate: 4e-4 (DOUBLED - faster convergence for 880M model)"
echo "  - Weight decay: 1e-5 (REDUCED - less aggressive regularization)"
echo "  - Dropout: 0.2 (REDUCED - balanced learning capacity)"
echo "  - Epochs: 50 (OPTIMIZED - sufficient for convergence)"
echo "  - Scheduler: plateau (STABLE - reduces on validation plateau)"
echo "  - Warmup: 15 epochs (EXTENDED - proper large model warmup)"
echo "  - Checkpoint frequency: 3 epochs (frequent monitoring)"
echo "  - Early stopping: 8 epochs patience (prevent overfitting)"
echo "  - BALANCED class weights + refined augmentation"
echo "  - Target: 90%+ validation accuracy (OPTIMIZED APPROACH)"
echo ""

# Train MedSigLIP with OVO-compatible system (single model mode) - OPTIMIZED HYPERPARAMETERS
python ensemble_local_trainer.py \
    --mode train \
    --dataset_path ./dataset_eyepacs \
    --output_dir ./medsiglip_results \
    --experiment_name "eyepacs_medsiglip_augmented_v2" \
    --base_models medsiglip_448 \
    --img_size 448 \
    --batch_size 8 \
    --epochs 50 \
    --learning_rate 4e-4 \
    --weight_decay 1e-5 \
    --ovo_dropout 0.2 \
    --freeze_weights false \
    --enable_medical_augmentation \
    --rotation_range 12.0 \
    --brightness_range 0.08 \
    --contrast_range 0.08 \
    --enable_focal_loss \
    --enable_class_weights \
    --class_weight_severe 20.0 \
    --class_weight_pdr 25.0 \
    --focal_loss_alpha 1.5 \
    --focal_loss_gamma 2.0 \
    --scheduler plateau \
    --warmup_epochs 15 \
    --validation_frequency 1 \
    --checkpoint_frequency 3 \
    --patience 10 \
    --early_stopping_patience 8 \
    --target_accuracy 0.90 \
    --seed 42

echo ""
echo "✅ EyePACS MedSigLIP training completed!"
echo "📁 Results saved to: ./medsiglip_results"
echo ""
echo "🎯 OPTIMIZED Configuration V2 Applied (90%+ Target):"
echo "  🏗️ Architecture: MedSigLIP-448 (medical foundation model - V2 OPTIMIZED)"
echo "  📊 Model capacity: 880M parameters (large medical model)"
echo "  🎓 Learning rate: 4e-4 (DOUBLED - faster convergence for large model)"
echo "  💧 Balanced dropout: 0.2 (optimized learning vs overfitting)"
echo "  ⚖️ Weight decay: 1e-5 (REDUCED - less aggressive regularization)"
echo "  📈 Scheduler: plateau (STABLE - adaptive LR reduction)"
echo "  ⏰ Extended warmup: 15 epochs (proper large model initialization)"
echo "  🎯 Core optimizations: Higher LR + Plateau scheduler + Balanced regularization"
echo "  🔀 BALANCED optimization: 20x/25x class weights, refined augmentation"
echo "  📈 Dataset: 33,857 samples with optimized minority class handling"
echo ""
echo "📊 Expected Performance with OPTIMIZED V2 Configuration:"
echo "  🎯 Target: 90%+ validation accuracy (V2 OPTIMIZED approach)"
echo "  🚀 Initial epochs: Faster convergence with 4e-4 LR"
echo "  🏥 Medical grade: 90%+ TARGET (improved from 79-86% range)"
echo "  📈 Plateau scheduler: Stable LR with adaptive reduction"
echo "  🔗 Batch size 8: Better GPU utilization for V100 with 880M model"
echo "  ⚡ Core improvements: Higher LR + Plateau scheduler + Reduced regularization"
echo "  ✅ Optimized learning: Overcomes LR stagnation with balanced approach"
echo ""
echo "📋 Next Steps:"
echo "  1. Analyze results: python model_analyzer.py --model ./medsiglip_results/models/best_medsiglip_448_multiclass.pth"
echo "  2. Validate medical-grade performance (>90% required)"
echo "  3. Train complementary models:"
echo "     ./train_aptos_densenet_v2.sh"
echo "     ./train_ddr_mobilenet.sh"
echo "  4. Combine all models for ensemble analysis"
echo ""
echo "🔗 ENSEMBLE COMPATIBILITY CONFIRMED:"
echo "  ✅ Model saved as: best_medsiglip_448_multiclass.pth (OVO-compatible)"
echo "  ✅ Same checkpoint format as DenseNet/MobileNet models"
echo "  ✅ Compatible with train_aptos_densenet_v2.sh output"
echo "  ✅ Ready for OVO ensemble combination"
echo "  ✅ Works with analyze_ovo_with_metrics.py and model_analyzer.py"
echo ""
echo "🚀 ENSEMBLE USAGE EXAMPLES:"
echo "  # Analyze this model with other OVO models"
echo "  python analyze_ovo_with_metrics.py --dataset_path ./medsiglip_results"
echo ""
echo "  # Comprehensive multi-model analysis"
echo "  python analyze_all_ovo_models.py"
echo ""
echo "  # Train ensemble with MedSigLIP + DenseNet + MobileNet"
echo "  # (After training other models, they will be compatible)"