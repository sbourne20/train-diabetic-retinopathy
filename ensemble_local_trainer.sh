#!/bin/bash

# OVO (One-Versus-One) Ensemble Training Script
# Research-Based Implementation: Lightweight Transfer Learning Ensemble
# MobileNet-v2 + InceptionV3 + DenseNet121 for Diabetic Retinopathy Classification
# Target: 96.96% accuracy with medical-grade validation

echo "🔢 OVO ENSEMBLE TRAINING - RESEARCH IMPLEMENTATION"
echo "Paper: A lightweight transfer learning based ensemble approach for diabetic retinopathy detection"
echo "Base Models: MobileNet-v2 + InceptionV3 + DenseNet121"
echo "Method: One-Versus-One (10 binary classifiers for 5 classes)"
echo "Dataset: Balanced dataset6 (21,149 images, 4.6:1 ratio)"
echo "Target: 96.96% ensemble accuracy with medical-grade standards"
echo ""

# Check if balanced dataset6 exists
if [ ! -d "./dataset6" ]; then
    echo "❌ ERROR: Balanced dataset6 not found"
    echo "Please ensure dataset6 has been balanced using dataset_balancer.py"
    exit 1
fi

# Verify dataset structure
for split in train val test; do
    if [ ! -d "./dataset6/$split" ]; then
        echo "❌ ERROR: Missing dataset6/$split directory"
        exit 1
    fi
done

# Verify class balance
echo "🔍 VERIFYING BALANCED DATASET:"
for split in train val test; do
    echo "--- $split ---"
    for class in 0 1 2 3 4; do
        count=$(ls "./dataset6/$split/$class/" 2>/dev/null | wc -l | tr -d ' ')
        echo "Class $class: $count images"
    done
done
echo ""

# Check if medical terms file exists
if [ ! -f "data/medical_terms_type1.json" ]; then
    echo "⚠️  WARNING: Medical terms file 'data/medical_terms_type1.json' not found"
    echo "Training will continue but medical validation may be limited"
fi

echo "🏥 OVO MEDICAL-GRADE CONFIGURATION:"
echo "  ✅ Standard preprocessing (CLAHE disabled for stability)"
echo "  ✅ Balanced dataset (4.6:1 ratio)"
echo "  ✅ Transfer learning with frozen weights"
echo "  ✅ Binary classification heads for each class pair"
echo "  ✅ Majority voting strategy"
echo "  ✅ Medical-grade validation thresholds"
echo "  ✅ Early stopping with patience"
echo ""

echo "🔧 OVO TRAINING PARAMETERS:"
echo "  📊 Binary classifier epochs: 30-50 (with early stopping)"
echo "  📈 Batch size: 16 (optimized for binary tasks)"
echo "  🎯 Total binary classifiers: 30 (10 pairs × 3 models)"
echo "  🎯 Target ensemble accuracy: ≥96.96%"
echo "  🏥 Medical threshold: ≥90%"
echo "  🔬 Research methodology: One-Versus-One ensemble"
echo ""

echo "🚀 Starting OVO ensemble training..."
echo "=========================================="

# Ensure output directory structure exists
echo "📁 Creating output directory structure..."
mkdir -p ./ovo_ensemble_results
mkdir -p ./ovo_ensemble_results/models
mkdir -p ./ovo_ensemble_results/logs
mkdir -p ./ovo_ensemble_results/results
echo "✅ Output directories created: $(pwd)/ovo_ensemble_results"

python ensemble_local_trainer.py \
    --mode train \
    --dataset_path ./dataset6 \
    --output_dir ./ovo_ensemble_results \
    --epochs 25 \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --weight_decay 1e-4 \
    --base_models mobilenet_v2 inception_v3 densenet121 \
    --freeze_weights \
    --enable_class_weights \
    --patience 15 \
    --early_stopping_patience 8 \
    --experiment_name ovo_stable_no_transforms_issues \
    --target_accuracy 0.9696 \
    --seed 42

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ OVO ensemble training completed successfully!"
    echo "🎯 Check results in ./ovo_ensemble_results/ directory"
    echo ""
    echo "📁 Results verification:"
    if [ -d "./ovo_ensemble_results" ]; then
        echo "   ✅ Results directory exists"

        if [ -f "./ovo_ensemble_results/models/ovo_ensemble_best.pth" ]; then
            echo "   🏆 Best OVO ensemble model saved"
            model_size=$(du -h "./ovo_ensemble_results/models/ovo_ensemble_best.pth" | cut -f1)
            echo "   📊 Model size: $model_size"
        else
            echo "   ❌ Best OVO model not found"
        fi

        if [ -f "./ovo_ensemble_results/results/complete_ovo_results.json" ]; then
            echo "   📊 Training results saved"

            # Extract final accuracy if possible
            if command -v python3 &> /dev/null; then
                final_acc=$(python3 -c "
import json
try:
    with open('./ovo_ensemble_results/results/complete_ovo_results.json', 'r') as f:
        results = json.load(f)
    if 'evaluation_results' in results and 'ensemble_accuracy' in results['evaluation_results']:
        print(f\"{results['evaluation_results']['ensemble_accuracy']:.4f} ({results['evaluation_results']['ensemble_accuracy']*100:.2f}%)\")
    else:
        print('Available')
except:
    print('Available')
" 2>/dev/null)
                echo "   🎯 Final OVO accuracy: $final_acc"

                # Show medical grade status
                medical_status=$(python3 -c "
import json
try:
    with open('./ovo_ensemble_results/results/complete_ovo_results.json', 'r') as f:
        results = json.load(f)
    if 'evaluation_results' in results:
        medical_pass = results['evaluation_results'].get('medical_grade_pass', False)
        research_target = results['evaluation_results'].get('research_target_achieved', False)
        print(f'Medical: {\"✅ PASS\" if medical_pass else \"❌ FAIL\"}, Research: {\"✅ ACHIEVED\" if research_target else \"❌ NOT ACHIEVED\"}')
    else:
        print('Status: Available in results file')
except:
    print('Status: Check results file')
" 2>/dev/null)
                echo "   🏥 $medical_status"
            fi
        fi

        # Show binary classifiers count
        classifier_count=$(find "./ovo_ensemble_results/models/" -name "best_*_*.pth" | wc -l | tr -d ' ')
        echo "   🔢 Binary classifiers trained: $classifier_count"

    else
        echo "   ❌ Results directory not found"
    fi

    echo ""
    echo "📋 Next steps:"
    echo "1. Review OVO ensemble results in ovo_ensemble_results/"
    echo "2. Analyze individual binary classifier performances"
    echo "3. If accuracy is <96.96%, consider:"
    echo "   - Increasing epochs for binary classifiers"
    echo "   - Fine-tuning learning rates per model"
    echo "   - Adding focal loss for imbalanced pairs"
    echo "4. Use the OVO ensemble for Phase 1.5 (Image Analysis)"
    echo "5. Compare with research paper benchmarks"

else
    echo "❌ OVO training failed with exit code: $EXIT_CODE"
    echo "🔍 Check logs above for error details"
    echo ""
    echo "Common issues:"
    echo "- GPU memory: Reduce batch_size to 8 or 4"
    echo "- Missing dependencies: pip install opencv-python scikit-learn"
    echo "- Dataset path: Ensure ./dataset6 exists with train/val/test"
    echo "- Model loading: Check torchvision model availability"
fi

exit $EXIT_CODE