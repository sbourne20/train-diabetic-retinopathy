#!/bin/bash
# MEDICAL-GRADE ANTI-OVERFITTING + DISK SPACE OPTIMIZED
# Aggressive overfitting prevention with minimal disk usage

echo "🛡️ DISK-OPTIMIZED ANTI-OVERFITTING TRAINING"
echo "Foundation Model: google/medsiglip-448 - MEMORY & DISK OPTIMIZED"
echo ""
echo "💾 DISK SPACE MANAGEMENT:"
echo "  🗑️ No frequent checkpoints (save space)"
echo "  🎯 Only save best model (essential only)"
echo "  📁 Minimal logging (reduce disk usage)"
echo "  ⚡ Fast cleanup after each epoch"
echo ""
echo "🛡️ ANTI-OVERFITTING CONFIGURATION:"
echo "  💧 Dropout: 0.6 (strong regularization)"
echo "  🎯 Learning Rate: 1e-5 (stable learning)"
echo "  ⚖️ Weight Decay: 1e-4 (10x stronger)"
echo "  ⏰ Early Stopping: patience 15"
echo "  📊 Validation Check: Every epoch"
echo "  📉 LR Scheduler: cosine_with_restarts"
echo ""

# Check if dataset3_augmented_resized exists
if [ ! -d "./dataset3_augmented_resized" ]; then
    echo "❌ ERROR: dataset3_augmented_resized directory not found"
    echo "Please ensure dataset3_augmented_resized exists"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ ERROR: .env file not found"
    echo "Please create .env file with your HuggingFace token"
    exit 1
fi

# Clean any existing results to free space
echo "🧹 Cleaning existing results to free disk space..."
rm -rf ./results/*
mkdir -p ./results/result

echo "✅ dataset3_augmented_resized found"
echo "✅ .env file found"
echo "💾 Disk space cleared for training"
echo ""

# DISK-OPTIMIZED training with anti-overfitting
python local_trainer.py \
  --mode train \
  --dataset_path ./dataset3_augmented_resized \
  --num_classes 5 \
  --pretrained_path google/medsiglip-448 \
  --img_size 448 \
  --epochs 100 \
  --use_lora yes \
  --lora_r 16 \
  --lora_alpha 32 \
  --learning_rate 1e-5 \
  --batch_size 6 \
  --freeze_backbone_epochs 0 \
  --enable_focal_loss \
  --focal_loss_alpha 4.0 \
  --focal_loss_gamma 6.0 \
  --enable_medical_grade \
  --enable_class_weights \
  --class_weight_severe 8.0 \
  --class_weight_pdr 6.0 \
  --gradient_accumulation_steps 4 \
  --warmup_epochs 10 \
  --scheduler cosine_with_restarts \
  --validation_frequency 1 \
  --patience 15 \
  --min_delta 0.005 \
  --weight_decay 1e-4 \
  --dropout 0.6 \
  --max_grad_norm 0.5 \
  --checkpoint_frequency 10 \
  --experiment_name "medsiglip_disk_optimized_anti_overfitting" \
  --device cuda \
  --no_wandb \
  --output_dir ./results \
  --medical_terms data/medical_terms_type1.json

echo ""
echo "💾 DISK SPACE OPTIMIZATION COMPLETED"
echo "🛡️ ANTI-OVERFITTING TRAINING COMPLETED"
echo "📈 CHECK: Validation/Training loss ratio should be <1.5"
echo "🎯 TARGET: >81.76% accuracy with proper generalization"