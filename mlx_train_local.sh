#!/bin/bash
# MLX MEDICAL-GRADE LOCAL TRAINING - MAC M4 NUCLEAR FOCAL LOSS FOR IMBALANCED DATASET3
# Optimized for severe class imbalance with aggressive focal loss parameters

echo "🍎 MLX LOCAL TRAINING: NUCLEAR FOCAL LOSS FOR IMBALANCED DATASET3 - MAC M4"
echo "Foundation Model: google/medsiglip-448 - AGGRESSIVE PARAMETERS FOR SEVERE IMBALANCE"
echo ""
echo "🚀 NUCLEAR FOCAL LOSS: Aggressive parameters for severe class imbalance"
echo "  🎯 TARGET: Handle 5.4x imbalance → 85%+ → 92% medical-grade accuracy"
echo "  ⚠️  Imbalanced Dataset3: 118k samples with 5.4:1 imbalance ratio"
echo "  ✅ Hardware: Apple Silicon M4 (MLX optimized for efficiency)"
echo "  ✅ Memory Optimized: LoRA r=16 for Apple Silicon efficiency"
echo ""
echo "🎯 NUCLEAR FOCAL LOSS CONFIGURATION:"
echo "  ✅ LoRA Rank (r): 16 (memory efficient configuration)"
echo "  ✅ LoRA Alpha: 32 (proven effective configuration)"
echo "  🎯 Learning Rate: 2e-5 (stable rate for imbalanced training)"
echo "  🔥 Class Weights: 8.0/8.0 (NUCLEAR: extreme priority for Classes 3&4)"
echo "  🚀 Scheduler: none (STABLE: fixed LR throughout training)"
echo "  ✅ Medical Warmup: 20 epochs (gradual ramp-up for stability)"
echo "  🎯 Batch Size: 2 (MLX OPTIMIZED: with gradient accumulation)"
echo "  ✅ Dropout: 0.3 (regularization for large dataset)"
echo "  ✅ Weight Decay: 1e-5 (light regularization)"
echo "  🔥 Focal Loss: α=4.0, γ=6.0 (NUCLEAR: aggressive focus on hard examples)"
echo ""
echo "🍎 MLX APPLE SILICON ADVANTAGES:"
echo "  • Memory Efficiency: Unified memory architecture"
echo "  • Memory Safety: Auto-limits to 70% of system memory"
echo "  • Memory Monitoring: Real-time usage tracking and warnings"
echo "  • Memory Cleanup: Periodic cache clearing every 50 batches"
echo "  • Speed: Native Apple Silicon optimization via MLX"
echo "  • Stability: Local training with full system control"
echo "  • Cost: Zero cloud compute charges"
echo "  • Debug Friendly: Real-time monitoring and adjustment"
echo "  • Batch Tracking: Progress bars with memory usage display"
echo ""

# Check if dataset3_augmented_resized exists
if [ ! -d "dataset3_augmented_resized" ]; then
    echo "❌ Error: dataset3_augmented_resized folder not found!"
    echo "Please ensure dataset3_augmented_resized exists with train/val/test structure."
    exit 1
fi

# Check if MLX is installed
python3 -c "import mlx.core" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: MLX not installed!"
    echo "Install with: pip install mlx"
    exit 1
fi

# Check if psutil is installed for memory management
python3 -c "import psutil" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing psutil for memory management..."
    pip install psutil || echo "⚠️ psutil installation failed"
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

# Run MLX training with NUCLEAR FOCAL LOSS parameters for imbalanced dataset3
python3 mlx_ai_trainer.py \
  --dataset-path dataset3_augmented_resized \
  --results-dir results \
  --num-epochs 50 \
  --batch-size 2 \
  --learning-rate 2e-5 \
  --weight-decay 1e-5 \
  --dropout 0.3 \
  --scheduler none \
  --gradient-accumulation-steps 6 \
  --warmup-epochs 20 \
  --validation-frequency 1 \
  --checkpoint-frequency 2 \
  --patience 15 \
  --min-delta 0.001 \
  --enable-focal-loss \
  --focal-loss-alpha 4.0 \
  --focal-loss-gamma 6.0 \
  --enable-class-weights \
  --class-weight-severe 8.0 \
  --class-weight-pdr 8.0 \
  --enable-medical-grade \
  --gradient-clip-norm 1.0 \
  --use-lora \
  --lora-r 16 \
  --lora-alpha 32 \
  --experiment-name "mlx_medsiglip_lora_nuclear_focal_imbalanced_dataset3"

echo ""
echo "⏱️ MLX NUCLEAR FOCAL LOSS TRAINING COMPLETED!"
echo "📊 Results saved to: results/"
echo "💾 Checkpoints saved to: results/checkpoints/"
echo "📈 Outputs and plots saved to: results/outputs/"
echo ""
echo "🎯 MLX NUCLEAR FOCAL LOSS TRAINING TIMELINE:"
echo "  • Duration: 12-16 hours on M4 (large imbalanced dataset training)"
echo "  • Validation checks: Every epoch (continuous progress monitoring)"
echo "  • Imbalance handling: Expected breakthrough epoch 15-25 (nuclear focal loss)"
echo "  • 85%+ convergence: Expected by epoch 30-40 (aggressive parameters)"
echo ""
echo "🏥 MEDICAL-GRADE SUCCESS CRITERIA (IMBALANCED DATA):"
echo "  • Overall validation accuracy: ≥85% (nuclear focal loss target)"
echo "  • Severe NPDR sensitivity: ≥90% (critical for patient safety)"
echo "  • PDR sensitivity: ≥95% (sight-threatening detection)"
echo "  • Minority class performance: Classes 3&4 >85% sensitivity despite 5.4x imbalance"
echo ""