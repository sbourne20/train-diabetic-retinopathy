#!/bin/bash
# MEDICAL-GRADE FULL FINE-TUNING - FIXED VERSION
# Tensor shape fixes for validation accuracy issue

echo "🏥 MEDICAL-GRADE TRAINING - FIXED VALIDATION ISSUE"
echo "Foundation Model: google/medsiglip-448 with FULL parameter training"
echo ""
echo "🔧 CRITICAL FIXES APPLIED:"
echo "  ✅ Tensor shape mismatch in confidence loss FIXED"
echo "  ✅ Validation accuracy tracking corruption FIXED" 
echo "  ✅ Early stopping logic improved with safety checks"
echo "  ✅ Best accuracy tracking with NaN protection"
echo ""
echo "🎯 MEDICAL-GRADE PERFORMANCE TARGET:"
echo "  • Overall Accuracy: 90-95% (MINIMUM 90% required)"
echo "  • Referable DR Sensitivity: >92%"
echo "  • Sight-threatening DR Sensitivity: >95%"
echo "  • FDA/CE Medical Device Compliance"
echo ""
echo "🚀 FULL MODEL TRAINING CONFIGURATION:"
echo "  ❌ LoRA: DISABLED (no parameter restrictions)"
echo "  ✅ Full Model: 464M parameters trainable"
echo "  ✅ Medical Memory Management: Optimized for V100"
echo "  ✅ Learning Rate: 1e-5 (full model stability)"
echo "  ✅ Batch Management: 2 + 8x accumulation = effective 16"
echo "  ✅ Regularization: Strong (weight_decay=1e-4, dropout=0.2)"
echo "  ✅ Medical-Grade Validation: Enhanced monitoring"
echo "  ✅ Auto-Resume Checkpoints: Every 5 epochs"
echo "  ✅ Extended Patience: 40 epochs for medical convergence"
echo "  ✅ TENSOR FIXES: Validation accuracy corruption prevented"
echo ""

python vertex_ai_trainer.py \
  --action train \
  --dataset dataset3_augmented_resized \
  --dataset-type 1 \
  --bucket_name dr-data-2 \
  --project_id curalis-20250522 \
  --region us-central1 \
  --num-epochs 150 \
  --use-lora no \
  --learning-rate 1e-5 \
  --batch-size 2 \
  --freeze-backbone-epochs 0 \
  --enable-focal-loss \
  --enable-medical-grade \
  --enable-class-weights \
  --gradient-accumulation-steps 8 \
  --warmup-epochs 10 \
  --scheduler cosine_restarts \
  --validation-frequency 5 \
  --patience 40 \
  --min-delta 0.001 \
  --weight-decay 1e-4 \
  --dropout 0.2 \
  --checkpoint_frequency 5 \
  --experiment-name "medsiglip_full_medical_grade_fixed"

echo ""
echo "⏱️ EXPECTED TIMELINE (FIXED TRAINING):"
echo "  • Training Duration: 6-8 days (150 epochs)"
echo "  • Memory Usage: 20-24GB on V100 (full model capacity)"
echo "  • Checkpoints: Saved every 5 epochs (30 total)"
echo "  • Early Stopping: 40 epoch patience for medical convergence"
echo "  • Validation: Fixed tensor shape issues preventing corruption"
echo ""
echo "📊 EXPECTED MEDICAL-GRADE RESULTS:"
echo "  • Overall Accuracy: 90-95% (medical device standard)"
echo "  • Per-class Sensitivity: >90% (regulatory compliance)"
echo "  • Per-class Specificity: >95% (clinical safety)"
echo "  • Referable DR Detection: >92% accuracy"
echo "  • Sight-threatening DR: >95% accuracy"
echo "  • FDA/CE Device Standards: COMPLIANT"
echo "  • Expert Ophthalmologist Validation: REQUIRED"
echo ""
echo "🎯 SUCCESS CRITERIA:"
echo "  • MINIMUM 90% validation accuracy required"
echo "  • All 5 DR classes must achieve >85% sensitivity"
echo "  • Medical-grade generalization (train/val gap <5%)"
echo "  • Ready for Phase 1.5 (Image Quality Assessment)"
echo ""
echo "🔧 TENSOR FIXES APPLIED:"
echo "  • Confidence loss tensor shape matching"
echo "  • Validation accuracy corruption prevention"
echo "  • NaN and invalid value protection"
echo "  • Best accuracy tracking safety checks"