#!/bin/bash
# MLX Local Training Script for Mac M4
# Balanced anti-overfitting parameters for medical-grade performance

echo "🏥 MLX DIABETIC RETINOPATHY TRAINING - MAC M4"
echo "Foundation Model: google/medsiglip-448 with FULL parameter training"
echo ""
echo "💻 LOCAL TRAINING ADVANTAGES:"
echo "  ✅ Complete parameter control (no checkpoint confusion)"
echo "  ✅ Immediate debugging and monitoring"
echo "  ✅ Cost-free training (no cloud charges)"
echo "  ✅ MLX optimization for Apple Silicon"
echo ""
echo "🔧 BALANCED ANTI-OVERFITTING CONFIGURATION:"
echo "  ✅ Learning Rate: 1e-4 (balanced, not nuclear)"
echo "  ✅ Weight Decay: 5e-3 (moderate regularization)"
echo "  ✅ Dropout: 0.3 (allows model to use learned features)"
echo "  ✅ Scheduler: polynomial (gentle LR decay)"
echo "  ✅ Batch Size: 2 with gradient accumulation (effective batch 8)"
echo "  ✅ Early Stopping: patience=15 (more time to improve)"
echo ""
echo "🎯 MEDICAL-GRADE TARGET:"
echo "  • Overall Accuracy: 90-95% (MINIMUM 90% required)"
echo "  • Referable DR Sensitivity: >92%"
echo "  • Sight-threatening DR Sensitivity: >95%"
echo "  • FDA/CE Medical Device Compliance"
echo ""

# Check if dataset exists
if [ ! -d "dataset3_augmented_resized" ]; then
    echo "❌ Error: dataset3_augmented_resized folder not found!"
    echo "Please ensure the dataset is in the current directory."
    exit 1
fi

# Check if MLX is installed
python3 -c "import mlx.core" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: MLX not installed!"
    echo "Install with: pip install mlx"
    exit 1
fi

# Check for HuggingFace token
if [ -z "$HUGGINGFACE_TOKEN" ] && [ ! -f ".env" ]; then
    echo "❌ Error: HUGGINGFACE_TOKEN not found!"
    echo "Create .env file with: HUGGINGFACE_TOKEN=hf_your_token_here"
    exit 1
fi

echo "💾 MODEL CACHING INFO:"
echo "  • First run: Downloads MedSigLIP-448 (3.5GB) to results/models/"
echo "  • Subsequent runs: Loads from cache instantly"
echo "  • To pre-download: python3 download_model.py"
echo ""
echo "🚀 STARTING LOCAL MLX TRAINING..."
echo ""

# Run MLX training with balanced parameters
python3 mlx_ai_trainer.py \
  --dataset-path dataset3_augmented_resized \
  --results-dir results \
  --num-epochs 150 \
  --batch-size 2 \
  --learning-rate 1e-4 \
  --weight-decay 5e-3 \
  --dropout 0.3 \
  --scheduler polynomial \
  --gradient-accumulation-steps 4 \
  --warmup-epochs 3 \
  --validation-frequency 1 \
  --checkpoint-frequency 2 \
  --patience 15 \
  --min-delta 0.01 \
  --enable-focal-loss \
  --enable-class-weights \
  --enable-medical-grade \
  --gradient-clip-norm 1.0 \
  --experiment-name "mlx_medsiglip_balanced_local"

echo ""
echo "⏱️ TRAINING COMPLETED!"
echo "📊 Results saved to: results/"
echo "💾 Checkpoints saved to: results/checkpoints/"
echo "📈 Outputs and plots saved to: results/outputs/"
echo ""
echo "🎯 CHECK MEDICAL-GRADE PERFORMANCE:"
echo "  • Review training_results.json for final accuracy"
echo "  • Check training_plots.png for learning curves"  
echo "  • Validation accuracy should reach 90%+ for medical grade"
echo ""