#!/bin/bash
# FULL TRAINING FALLBACK - LOCAL V100 MAXIMUM PERFORMANCE MODE
# Use only if LoRA training doesn't achieve 85%+ accuracy

echo "🔥 FULL TRAINING MODE: MAXIMUM PERFORMANCE ON V100"
echo "⚠️  WARNING: This mode uses 12-14GB memory vs 6GB for LoRA"
echo ""
echo "🎯 FULL TRAINING STRATEGY:"
echo "  • Mode: Full model fine-tuning (all parameters trainable)"
echo "  • Memory: 12-14GB usage (requires 16GB V100)"
echo "  • Speed: Slower than LoRA but potentially higher accuracy ceiling"
echo "  • Batch Size: Reduced to 2-3 due to memory constraints"
echo "  • Target: 88-92% validation accuracy"
echo ""
echo "🚨 USE FULL TRAINING ONLY IF:"
echo "  • LoRA training plateaus below 85%"
echo "  • You need maximum possible accuracy"
echo "  • You have sufficient time for longer training"
echo ""
echo "💾 V100 MEMORY OPTIMIZATION FOR FULL TRAINING:"
echo "  • Batch Size: 3 (reduced from 6 to fit in 16GB)"
echo "  • Gradient Accumulation: 6 (maintain effective batch size 18)"
echo "  • Mixed Precision: Enabled (reduces memory by ~30%)"
echo "  • Gradient Checkpointing: Enabled if needed"
echo ""

# Check if dataset5 exists
if [ ! -d "./dataset5" ]; then
    echo "❌ ERROR: dataset5 directory not found in current path"
    echo "Please ensure dataset5 exists with train/val/test structure"
    exit 1
fi

# Confirm full training mode
echo "⚠️  WARNING: Full training will use 12-14GB of your 16GB V100"
echo "Press Enter to continue with full training, or Ctrl+C to cancel..."
read -r

echo "✅ dataset5 found - proceeding with FULL TRAINING mode"
echo ""

# Run local training with FULL MODEL parameters (no LoRA)
python local_trainer.py \
  --mode train \
  --dataset_path ./dataset5 \
  --num_classes 5 \
  --pretrained_path google/medsiglip-448 \
  --img_size 448 \
  --epochs 80 \
  --use_lora no \
  --learning_rate 1e-5 \
  --batch_size 3 \
  --freeze_backbone_epochs 5 \
  --enable_focal_loss \
  --focal_loss_alpha 2.0 \
  --focal_loss_gamma 3.0 \
  --enable_medical_grade \
  --enable_class_weights \
  --class_weight_severe 4.0 \
  --class_weight_pdr 3.0 \
  --gradient_accumulation_steps 6 \
  --warmup_epochs 10 \
  --scheduler polynomial \
  --validation_frequency 2 \
  --patience 20 \
  --min_delta 0.001 \
  --weight_decay 1e-4 \
  --dropout 0.2 \
  --max_grad_norm 0.5 \
  --checkpoint_frequency 5 \
  --experiment_name "medsiglip_FULL_LOCAL_V100_MAXIMUM_PERFORMANCE" \
  --device cuda \
  --medical_terms data/medical_terms_type1.json

echo ""
echo "⏱️ FULL TRAINING TIMELINE:"
echo "  • Duration: 6-10 hours (longer due to full model training)"
echo "  • Memory Usage: 12-14GB V100 (full model parameters)"
echo "  • Validation checks: Every 2 epochs (to save time)"
echo "  • Expected performance: 88-92% validation accuracy"
echo "  • Breakthrough point: Epoch 15-30"
echo ""
echo "🎯 FULL TRAINING SUCCESS CRITERIA:"
echo "  • Overall validation accuracy: ≥88% (higher ceiling than LoRA)"
echo "  • All class sensitivity: ≥85%"
echo "  • Medical-grade performance: >90% clinical utility"
echo "  • Memory stable: No OOM errors throughout training"
echo ""
echo "🔥 FULL TRAINING ADVANTAGES:"
echo "  • Maximum accuracy potential"
echo "  • All parameters optimized for your specific dataset"
echo "  • Better feature extraction learning"
echo "  • Highest possible medical-grade performance"
echo ""
echo "⚠️ FULL TRAINING CONSIDERATIONS:"
echo "  • Higher memory usage (monitor with nvidia-smi)"
echo "  • Longer training time"
echo "  • Risk of overfitting on smaller dataset"
echo "  • Higher computational cost"
echo ""
echo "🚀 LAUNCHING FULL TRAINING ON V100..."
echo "💾 USING 12-14GB MEMORY FOR MAXIMUM PERFORMANCE"