#!/bin/bash
# MEDICAL-GRADE ANTI-OVERFITTING LOCAL TRAINING
# Aggressive overfitting prevention for local V100 execution

echo "🛡️ ANTI-OVERFITTING V100 TRAINING: PREVENTING GENERALIZATION ISSUES"
echo "Foundation Model: google/medsiglip-448 - OVERFITTING PREVENTION OPTIMIZED"
echo ""
echo "🚨 OVERFITTING PREVENTION MODE: Aggressive regularization to prevent memorization"
echo "  🎯 ISSUE: Current training shows 2.56x validation/training loss ratio"
echo "  🛡️ SOLUTION: Stronger regularization, early stopping, reduced learning rate"
echo "  🎯 TARGET: Achieve 81.76%+ with proper generalization"
echo ""
echo "🔄 RESUME CONFIGURATION:"
echo "  📥 Resume from: best_model.pth (if exists)"
echo "  💾 Checkpoints: Save every epoch (full monitoring)"
echo "  ☁️ Cloud Storage: Epoch checkpoints to gs://dr-data-2/checkpoints"
echo "  🏠 Local Storage: Best model only (save disk space)"
echo ""
echo "🔧 ANTI-OVERFITTING CONFIGURATION:"
echo "  💧 Dropout: 0.6 (increased from 0.4 for stronger regularization)"
echo "  🎯 Learning Rate: 1e-5 (reduced from 2e-5 for stability)"
echo "  ⚖️ Weight Decay: 1e-4 (increased 10x from 1e-5)"
echo "  ⏰ Early Stopping: patience 15 (reduced from 40)"
echo "  📊 Validation Check: Every epoch (immediate overfitting detection)"
echo "  🎯 Min Delta: 0.005 (increased sensitivity)"
echo "  ✂️ Gradient Clipping: 0.5 (reduced from 1.0)"
echo "  📉 LR Scheduler: cosine_with_restarts (instead of none for gradual decay)"
echo ""

# Check if dataset3_augmented_resized exists
if [ ! -d "./dataset3_augmented_resized" ]; then
    echo "❌ ERROR: dataset3_augmented_resized directory not found in current path"
    echo "Please ensure dataset3_augmented_resized exists (same as 81.76% success)"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ ERROR: .env file not found in current directory"
    echo "Please create .env file with your HuggingFace token:"
    echo "HUGGINGFACE_TOKEN=hf_your_token_here"
    exit 1
fi

# Install python-dotenv if not available
echo "📦 Ensuring python-dotenv is available..."
pip install python-dotenv || echo "⚠️ python-dotenv installation failed"

echo "✅ dataset3_augmented_resized found - proceeding with anti-overfitting training"
echo "✅ .env file found - HuggingFace token should be loaded"
echo ""

# Run local training with ANTI-OVERFITTING parameters + CLOUD STORAGE
python3 local_trainer.py \
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
  --checkpoint_frequency 1 \
  --save_checkpoint gs://dr-data-2/checkpoints \
  --experiment_name "medsiglip_lora_r16_ANTI_OVERFITTING" \
  --device cuda \
  --no_wandb \
  --output_dir ./results \
  --medical_terms data/medical_terms_type1.json \
  --resume_from_checkpoint ./results/result/best_model.pth

echo ""
echo "🛡️ ANTI-OVERFITTING SUCCESS CRITERIA:"
echo "  • Validation/Training loss ratio: <1.5 (healthy generalization)"
echo "  • Validation accuracy: >81.76% (exceed previous baseline)"
echo "  • Training stability: Smooth convergence without spikes"
echo "  • Early stopping: Prevent memorization before it occurs"
echo "  • Medical-grade performance: >85% with proper generalization"
echo ""
echo "📊 OVERFITTING MONITORING:"
echo "  • Watch validation/training loss ratio every epoch"
echo "  • Early stop if validation starts diverging from training"
echo "  • Checkpoint saved every epoch for rollback capability"
echo "  • Stronger regularization prevents memorization"
echo ""
echo "🎯 ANTI-OVERFITTING TRAINING LAUNCHED"
echo "🛡️ REGULARIZATION: Enhanced dropout, weight decay, early stopping"
echo "📈 TARGET: 81.76%+ accuracy with proper generalization"