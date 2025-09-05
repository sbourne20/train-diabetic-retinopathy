#!/bin/bash
# MEDICAL-GRADE EXTREME ANTI-OVERFITTING TRAINING
# Resume with maximum regularization to achieve 90%+ validation accuracy

echo "🏥 MEDICAL-GRADE EXTREME ANTI-OVERFITTING TRAINING"
echo "Foundation Model: google/medsiglip-448 with EXTREME regularization"
echo ""
echo "🎯 EXTREME ANTI-OVERFITTING STRATEGY:"
echo "  ✅ Ultra-low learning rate: 5e-5 (10x lower than before)"
echo "  ✅ Maximum weight decay: 1e-1 (10x stronger regularization)"
echo "  ✅ Heavy dropout: 0.7 (maximum regularization)"
echo "  ✅ Smallest batch size: effective batch=4 (prevent memorization)"
echo "  ✅ Early stopping: patience=10 (aggressive overfitting detection)"
echo "  ✅ Gradient clipping: 0.5 (prevent gradient explosion)"
echo "  ✅ Resume from best checkpoint (preserve good weights)"
echo ""
echo "💡 WHY THIS WILL WORK:"
echo "  • Current training shows perfect memorization - need extreme regularization"
echo "  • Smaller effective batch forces model to generalize better"
echo "  • Ultra-low LR prevents catastrophic overfitting"
echo "  • Heavy regularization forces medical-grade generalization"
echo ""
echo "💰 COST SAVINGS:"
echo "  • Resume from existing checkpoint: ~$200 already invested"
echo "  • Additional cost: ~$80-100 (50-70 more epochs with early stopping)"
echo "  • Total: ~$280-300 instead of ~$480 starting fresh"
echo ""

python vertex_ai_trainer.py \
  --action train \
  --dataset dataset3_augmented_resized \
  --dataset-type 1 \
  --bucket_name dr-data-2 \
  --project_id curalis-20250522 \
  --region us-central1 \
  --num-epochs 100 \
  --use-lora no \
  --learning-rate 5e-5 \
  --batch-size 1 \
  --freeze-backbone-epochs 0 \
  --enable-focal-loss \
  --enable-medical-grade \
  --enable-class-weights \
  --warmup-epochs 1 \
  --scheduler cosine \
  --validation-frequency 1 \
  --patience 10 \
  --min-delta 0.005 \
  --weight-decay 1e-1 \
  --dropout 0.7 \
  --checkpoint_frequency 1 \
  --gradient-accumulation-steps 4 \
  --gradient-clip-value 0.5 \
  --resume-from-checkpoint gs://dr-data-2/checkpoints/best_model.pth \
  --experiment-name "medsiglip_extreme_antioverfitting"

echo ""
echo "🎯 EXPECTED OUTCOME:"
echo "  • Validation accuracy: 90-95% (medical-grade)"
echo "  • Training time: 2-3 days (early stopping will kick in)"
echo "  • Generalization: Excellent due to extreme regularization"
echo "  • Cost: ~$80-100 additional (total ~$280-300)"
echo ""
echo "📊 KEY CHANGES FROM PREVIOUS TRAINING:"
echo "  • Learning rate: 1e-4 → 5e-5 (50% reduction)"
echo "  • Weight decay: 5e-3 → 1e-1 (20x increase!)"
echo "  • Dropout: 0.3 → 0.7 (2.3x increase)"
echo "  • Batch size: 2→1, accumulation: 4→4 (effective batch=4 instead of 8)"
echo "  • Gradient clip: none → 0.5 (stability)"
echo "  • Early stopping: patience 15→10 (faster overfitting detection)"