#!/bin/bash
# EVALUATE EXISTING MODELS - CHECK IF ALREADY MEDICAL-GRADE
# Test both best_model.pth and final_model.pth before spending more money

echo "🔍 EVALUATING EXISTING MODELS - MEDICAL-GRADE CHECK"
echo "Testing both available models to see if we already have 90%+ validation accuracy"
echo ""

echo "📋 TESTING BEST_MODEL.PTH:"
echo "  • Location: gs://dr-data-2/checkpoints/best_model.pth"
echo "  • Expected: Best validation performance during training"
echo ""

python vertex_ai_trainer.py \
  --action evaluate \
  --dataset dataset3_augmented_resized \
  --dataset-type 1 \
  --bucket_name dr-data-2 \
  --project_id curalis-20250522 \
  --region us-central1 \
  --model-path gs://dr-data-2/checkpoints/best_model.pth \
  --experiment-name "evaluate_best_model"

echo ""
echo "📋 TESTING FINAL_MODEL.PTH:"
echo "  • Location: gs://dr-data-2/models/final_model.pth"
echo "  • Expected: Final training checkpoint"
echo ""

python vertex_ai_trainer.py \
  --action evaluate \
  --dataset dataset3_augmented_resized \
  --dataset-type 1 \
  --bucket_name dr-data-2 \
  --project_id curalis-20250522 \
  --region us-central1 \
  --model-path gs://dr-data-2/models/final_model.pth \
  --experiment-name "evaluate_final_model"

echo ""
echo "💡 DECISION MATRIX:"
echo "  • If either model achieves 90%+ validation accuracy: USE IT!"
echo "  • If both are <90%: Run extreme anti-overfitting training"
echo "  • Cost: ~$5 evaluation vs ~$100 additional training"