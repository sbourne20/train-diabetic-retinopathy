#!/bin/bash

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# EyePACS + MedSigLIP-448 CONSERVATIVE Medical-Grade Training Script
echo "🏥 EyePACS + MedSigLIP-448 CONSERVATIVE Training (86%+ → 90%+ Conservative)"
echo "================================================================="
echo "🎯 Target: 90%+ accuracy with CONSERVATIVE stable approach"
echo "📊 Dataset: EyePACS (5-class DR classification)"
echo "🏗️ Model: MedSigLIP-448 (medical foundation model)"
echo "🔬 CONSERVATIVE approach focusing on stability and reproduction"
echo ""

# Create output directory for conservative MedSigLIP results
mkdir -p ./medsiglip_conservative_results

echo "🔬 EyePACS MedSigLIP CONSERVATIVE Configuration (Stable 86%+ → 90%+):"
echo "  - Dataset: EyePACS (./dataset_eyepacs) - AUGMENTED 33,857 samples"
echo "  - Model: MedSigLIP-448 (medical foundation model - CONSERVATIVE SETTINGS)"
echo "  - Image size: 448x448 (MedSigLIP required size)"
echo "  - Batch size: 8 (PROVEN - same as 86.05% run)"
echo "  - Learning rate: 5e-5 (CONSERVATIVE - lower than proven to avoid overshoot)"
echo "  - Weight decay: 3e-4 (PROVEN - exact from 86.05% run)"
echo "  - Dropout: 0.15 (CONSERVATIVE - lower than proven 0.2 for stability)"
echo "  - Epochs: 100 (EXTENDED - allow more time for convergence)"
echo "  - Scheduler: plateau (PROVEN - same as 86.05% run)"
echo "  - Warmup: 5 epochs (PROVEN - exact from 86.05% run)"
echo "  - Focal loss: alpha=2.0, gamma=3.0 (CONSERVATIVE - slightly reduced)"
echo "  - Class weights: 20x/25x (CONSERVATIVE - reduced for stability)"
echo "  - Conservative augmentation: 10° rotation, 8% brightness/contrast"
echo "  - Target: 90%+ validation accuracy (CONSERVATIVE STABLE APPROACH)"
echo ""

# Train MedSigLIP with CONSERVATIVE configuration
python ensemble_local_trainer.py \
    --mode train \
    --dataset_path ./dataset_eyepacs \
    --output_dir ./medsiglip_conservative_results \
    --experiment_name "eyepacs_medsiglip_conservative_stable" \
    --base_models medsiglip_448 \
    --img_size 448 \
    --batch_size 8 \
    --epochs 100 \
    --learning_rate 5e-5 \
    --weight_decay 3e-4 \
    --ovo_dropout 0.15 \
    --freeze_weights false \
    --enable_medical_augmentation \
    --rotation_range 10.0 \
    --brightness_range 0.08 \
    --contrast_range 0.08 \
    --enable_focal_loss \
    --enable_class_weights \
    --class_weight_severe 20.0 \
    --class_weight_pdr 25.0 \
    --focal_loss_alpha 2.0 \
    --focal_loss_gamma 3.0 \
    --scheduler plateau \
    --warmup_epochs 5 \
    --validation_frequency 1 \
    --checkpoint_frequency 3 \
    --patience 25 \
    --early_stopping_patience 20 \
    --target_accuracy 0.90 \
    --max_grad_norm 0.5 \
    --seed 42

echo ""
echo "✅ EyePACS MedSigLIP CONSERVATIVE training completed!"
echo "📁 Results saved to: ./medsiglip_conservative_results"
echo ""
echo "🎯 CONSERVATIVE Configuration Applied (Focus on Stability):"
echo "  🏗️ Architecture: MedSigLIP-448 (medical foundation model - CONSERVATIVE SETTINGS)"
echo "  📊 Model capacity: 880M parameters (large medical model)"
echo "  🎓 Learning rate: 5e-5 (CONSERVATIVE - lower for stability)"
echo "  💧 Conservative dropout: 0.15 (reduced from proven 0.2 for stability)"
echo "  ⚖️ Weight decay: 3e-4 (PROVEN - exact from 86.05% configuration)"
echo "  📈 Scheduler: plateau (PROVEN - exact from 86.05% configuration)"
echo "  ⏰ Warmup: 5 epochs (PROVEN - exact from 86.05% configuration)"
echo "  🎯 Conservative focal loss: alpha=2.0, gamma=3.0 (reduced for stability)"
echo "  🔀 Conservative optimization: 20x/25x class weights (reduced for stability)"
echo "  📈 Dataset: 33,857 samples with CONSERVATIVE imbalance handling"
echo "  🎯 Gradient clipping: 0.5 (CONSERVATIVE - stability first)"
echo ""
echo "📊 Expected Performance with CONSERVATIVE Configuration:"
echo "  🎯 Target: 90%+ validation accuracy (CONSERVATIVE stable approach)"
echo "  🚀 Stability: Should maintain steady improvement from epoch 1"
echo "  🏥 Medical grade: 90%+ TARGET (conservative but reliable)"
echo "  📈 Plateau scheduler: Proven stable LR progression"
echo "  🔗 Batch size 8: Proven optimal from 86.05% run"
echo "  ⚡ Core success factors: CONSERVATIVE focal loss + CONSERVATIVE weights"
echo "  ✅ Conservative approach: Focused on reproducing and improving 86.05%"
echo ""
echo "📋 Next Steps:"
echo "  1. Monitor early epochs for consistency with 86.05% baseline"
echo "  2. Analyze results: python model_analyzer.py --model ./medsiglip_conservative_results/models/best_medsiglip_448_multiclass.pth"
echo "  3. If stable 90%+, then try more aggressive configurations"
echo "  4. Use as reliable baseline for ensemble combination"