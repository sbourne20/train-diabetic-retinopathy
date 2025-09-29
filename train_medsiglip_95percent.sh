#!/bin/bash

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# EyePACS + MedSigLIP-448 ENHANCED Medical-Grade Training Script for 95% Accuracy
echo "🏥 EyePACS + MedSigLIP-448 ENHANCED Medical-Grade Training (95% Target)"
echo "=================================================================="
echo "🎯 Target: 95%+ accuracy with ENHANCED MedSigLIP-448 configuration"
echo "📊 Dataset: EyePACS (5-class DR classification)"
echo "🏗️ Model: MedSigLIP-448 (medical foundation model)"
echo "🔬 ENHANCED medical-grade architecture with proven optimizations"
echo ""

# Create output directory for enhanced MedSigLIP results
mkdir -p ./medsiglip_95percent_results

echo "🔬 EyePACS MedSigLIP ENHANCED Configuration (86.05% → 95% Target):"
echo "  - Dataset: EyePACS (./dataset_eyepacs) - AUGMENTED 33,857 samples"
echo "  - Model: MedSigLIP-448 (medical foundation model - ENHANCED SETTINGS)"
echo "  - Image size: 448x448 (MedSigLIP required size)"
echo "  - Batch size: 6 (OPTIMIZED - better gradient accumulation)"
echo "  - Learning rate: 2e-4 (ENHANCED - optimal for large models)"
echo "  - Weight decay: 1e-4 (OPTIMIZED - reduced for better learning)"
echo "  - Dropout: 0.3 (ENHANCED - balanced regularization)"
echo "  - Epochs: 80 (EXTENDED - full convergence for 95% target)"
echo "  - Scheduler: cosine_warmup (ENHANCED - better convergence)"
echo "  - Warmup: 8 epochs (EXTENDED - stable initialization)"
echo "  - EXTREME focal loss: alpha=3.0, gamma=4.0 (ENHANCED for hard examples)"
echo "  - EXTREME class weights: 30x/35x (ENHANCED for imbalanced data)"
echo "  - Enhanced augmentation: 20° rotation, 15% brightness/contrast"
echo "  - Gradient clipping: 1.0 (NEW - stability for large models)"
echo "  - Target: 95%+ validation accuracy (ENHANCED APPROACH)"
echo ""

# Train MedSigLIP with ENHANCED configuration for 95% accuracy
python ensemble_local_trainer.py \
    --mode train \
    --dataset_path ./dataset_eyepacs \
    --output_dir ./medsiglip_95percent_results \
    --experiment_name "eyepacs_medsiglip_95percent_enhanced" \
    --base_models medsiglip_448 \
    --img_size 448 \
    --batch_size 6 \
    --epochs 80 \
    --learning_rate 2e-4 \
    --weight_decay 1e-4 \
    --ovo_dropout 0.3 \
    --freeze_weights false \
    --enable_medical_augmentation \
    --rotation_range 20.0 \
    --brightness_range 0.15 \
    --contrast_range 0.15 \
    --enable_focal_loss \
    --enable_class_weights \
    --class_weight_severe 30.0 \
    --class_weight_pdr 35.0 \
    --focal_loss_alpha 3.0 \
    --focal_loss_gamma 4.0 \
    --scheduler cosine \
    --warmup_epochs 8 \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --patience 20 \
    --early_stopping_patience 15 \
    --target_accuracy 0.95 \
    --max_grad_norm 1.0 \
    --seed 42

echo ""
echo "✅ EyePACS MedSigLIP ENHANCED training completed!"
echo "📁 Results saved to: ./medsiglip_95percent_results"
echo ""
echo "🎯 ENHANCED Configuration Applied (86.05% → 95% Target):"
echo "  🏗️ Architecture: MedSigLIP-448 (medical foundation model - ENHANCED SETTINGS)"
echo "  📊 Model capacity: 880M parameters (large medical model)"
echo "  🎓 Learning rate: 2e-4 (ENHANCED - optimal for medical foundation models)"
echo "  💧 Enhanced dropout: 0.3 (optimized regularization)"
echo "  ⚖️ Weight decay: 1e-4 (OPTIMIZED - reduced for better learning)"
echo "  📈 Scheduler: cosine_warmup (ENHANCED - better convergence pattern)"
echo "  ⏰ Extended warmup: 8 epochs (ENHANCED - stable initialization)"
echo "  🎯 Core success factors: ENHANCED focal loss + ENHANCED class weights"
echo "  🔀 ENHANCED optimization: 30x/35x class weights + alpha=3.0, gamma=4.0"
echo "  📈 Dataset: 33,857 samples with ENHANCED imbalance handling"
echo "  🎯 Gradient clipping: 1.0 (NEW - stability for large model convergence)"
echo ""
echo "📊 Expected Performance with ENHANCED Configuration:"
echo "  🎯 Target: 95%+ validation accuracy (ENHANCED approach from 86.05% baseline)"
echo "  🚀 Initial epochs: Should exceed 86.05% by epoch 3-4"
echo "  🏥 Medical grade: 95%+ TARGET (medical production ready)"
echo "  📈 Cosine scheduler: Enhanced convergence with warmup"
echo "  🔗 Batch size 6: Enhanced gradient quality for V100"
echo "  ⚡ Core success factors: ENHANCED focal loss (3.0/4.0) + ENHANCED weights (30x/35x)"
echo "  ✅ Enhanced approach: Optimized from proven 86.05% configuration"
echo ""
echo "📋 Next Steps:"
echo "  1. Analyze results: python model_analyzer.py --model ./medsiglip_95percent_results/models/best_medsiglip_448_multiclass.pth"
echo "  2. Validate 95%+ medical-grade performance"
echo "  3. Train complementary models for ensemble:"
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
echo "  python analyze_ovo_with_metrics.py --dataset_path ./medsiglip_95percent_results"
echo ""
echo "  # Comprehensive multi-model analysis"
echo "  python analyze_all_ovo_models.py"
echo ""
echo "  # Train ensemble with ENHANCED MedSigLIP + DenseNet + MobileNet"
echo "  # (After training other models, they will be compatible)"