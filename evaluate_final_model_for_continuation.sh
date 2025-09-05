#!/bin/bash
# EVALUATE FINAL_MODEL.PTH FOR CONTINUED TRAINING SUITABILITY
# Check validation metrics to determine if we can continue from this checkpoint

echo "🔍 EVALUATING FINAL_MODEL.PTH FOR CONTINUED TRAINING"
echo "Foundation Model: google/medsiglip-448 evaluation"
echo ""
echo "🎯 EVALUATION PURPOSE:"
echo "  • Check validation accuracy of final_model.pth"
echo "  • Assess if model is suitable for continued training"
echo "  • Determine baseline performance before extreme anti-overfitting"
echo ""
echo "💰 COST-EFFECTIVE STRATEGY:"
echo "  • Evaluation cost: ~$5 (much cheaper than full training)"
echo "  • If suitable: Continue training with extreme regularization (~$80-100)"
echo "  • Total cost: ~$285-305 instead of ~$480 starting fresh"
echo ""
echo "📊 WHAT WE'RE LOOKING FOR:"
echo "  • Validation accuracy: Current performance level"
echo "  • Per-class metrics: Balance across all 5 DR classes"
echo "  • Generalization gap: Training vs validation difference"
echo "  • Starting point quality: Better than random (>20% accuracy)"
echo ""

python vertex_ai_trainer.py \
  --action evaluate \
  --dataset dataset3_augmented_resized \
  --dataset-type 1 \
  --bucket_name dr-data-2 \
  --project_id curalis-20250522 \
  --region us-central1 \
  --model-path gs://dr-data-2/models/final_model.pth \
  --experiment-name "evaluate_final_model_continuation"

echo ""
echo "🎯 DECISION CRITERIA:"
echo "  • If validation accuracy >70%: EXCELLENT - continue with light regularization"
echo "  • If validation accuracy 50-70%: GOOD - continue with moderate regularization"  
echo "  • If validation accuracy 30-50%: FAIR - continue with extreme regularization"
echo "  • If validation accuracy <30%: POOR - consider starting fresh or different approach"
echo ""
echo "📋 NEXT STEPS BASED ON RESULTS:"
echo "  • >70%: Use learning_rate=1e-4, weight_decay=1e-2, dropout=0.5"
echo "  • 50-70%: Use learning_rate=5e-5, weight_decay=5e-2, dropout=0.6" 
echo "  • 30-50%: Use learning_rate=2e-5, weight_decay=1e-1, dropout=0.7"
echo "  • <30%: Reconsider strategy or start fresh"