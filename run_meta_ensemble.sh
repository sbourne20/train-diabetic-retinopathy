#!/bin/bash

# Meta-Ensemble OVO Models - Combine EfficientNetB2 and DenseNet121

source venv/bin/activate

echo "🎯 META-ENSEMBLE: Combining OVO Models"
echo "========================================================================"
echo ""
echo "📦 Models to Combine:"
echo "   1. EfficientNetB2 OVO (64.20% test accuracy)"
echo "   2. DenseNet121 OVO (64.84% test accuracy)"
echo ""
echo "🔗 Meta-Ensemble Methods Available:"
echo "   • average: Simple average of probabilities"
echo "   • weighted: Weighted average (custom weights)"
echo "   • vote: Majority voting on predictions"
echo ""

# Method 1: Simple Average (RECOMMENDED START)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔹 METHOD 1: Simple Average"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 meta_ensemble_ovo.py \
    --models efficientnetb2 densenet121 \
    --model_dirs ./efficientnetb2_5class_results ./densenet_5class_results \
    --dataset_path ./dataset_eyepacs_5class_balanced \
    --method average \
    --img_size 260 \
    --batch_size 32 \
    --output_file ./meta_ensemble_average_results.json

echo ""
echo ""

# Method 2: Weighted Average (Give more weight to DenseNet since it's 0.64% better)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔹 METHOD 2: Weighted Average (DenseNet slightly favored)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 meta_ensemble_ovo.py \
    --models efficientnetb2 densenet121 \
    --model_dirs ./efficientnetb2_5class_results ./densenet_5class_results \
    --dataset_path ./dataset_eyepacs_5class_balanced \
    --method weighted \
    --weights 0.48 0.52 \
    --img_size 260 \
    --batch_size 32 \
    --output_file ./meta_ensemble_weighted_results.json

echo ""
echo ""

# Method 3: Majority Voting
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔹 METHOD 3: Majority Voting"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 meta_ensemble_ovo.py \
    --models efficientnetb2 densenet121 \
    --model_dirs ./efficientnetb2_5class_results ./densenet_5class_results \
    --dataset_path ./dataset_eyepacs_5class_balanced \
    --method vote \
    --img_size 260 \
    --batch_size 32 \
    --output_file ./meta_ensemble_vote_results.json

echo ""
echo ""
echo "✅ Meta-Ensemble Evaluation Complete!"
echo ""
echo "📊 Expected Improvement:"
echo "   Individual models: 64.20% and 64.84%"
echo "   Expected meta-ensemble: 68-72% (complementary errors)"
echo ""
echo "📁 Results saved to:"
echo "   • ./meta_ensemble_average_results.json"
echo "   • ./meta_ensemble_weighted_results.json"
echo "   • ./meta_ensemble_vote_results.json"
echo ""
echo "🚀 NEXT STEPS:"
echo "   1. Review which method performs best"
echo "   2. Train ResNet50 OVO model"
echo "   3. Add ResNet50 to meta-ensemble (3-model combination)"
echo "   4. Expected 3-model ensemble: 70-75% accuracy"
echo ""
