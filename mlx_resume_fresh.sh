#!/bin/bash
# MLX Resume Training with Fresh Optimizer
# Load model weights but use fresh optimizer (Option 1 implementation)

echo "💰 MLX INVESTMENT PRESERVATION TRAINING"
echo "Loading your trained model weights + fresh optimizer with balanced parameters"
echo ""
echo "🔄 OPTION 1 IMPLEMENTATION:"
echo "  ✅ Load model weights from checkpoint (preserves your investment)"
echo "  ✅ Fresh optimizer with new parameters (fixes overfitting)"
echo "  ✅ Balanced anti-overfitting (not nuclear, not under-learning)"
echo ""

# Check if best_model.pth exists locally or needs download
if [ ! -f "results/checkpoints/best_model.pth" ]; then
    echo "📥 Downloading best_model.pth from GCS..."
    mkdir -p results/checkpoints
    
    # Download from GCS (you'll need to replace this with your actual download)
    # gsutil cp gs://dr-data-2/checkpoints/best_model.pth results/checkpoints/
    echo "❌ Please download gs://dr-data-2/checkpoints/best_model.pth to results/checkpoints/"
    echo "   Run: gsutil cp gs://dr-data-2/checkpoints/best_model.pth results/checkpoints/"
    exit 1
fi

echo "✅ Found local best_model.pth checkpoint"
echo ""

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
  --resume-from-checkpoint results/checkpoints/best_model.pth \
  --fresh-optimizer \
  --experiment-name "mlx_resume_fresh_optimizer"

echo ""
echo "💰 INVESTMENT PRESERVATION COMPLETE!"
echo "Your model knowledge has been preserved while fixing overfitting."