#!/bin/bash

# Clean Ensemble Training Script
# EfficientNetB2 + ResNet50 + DenseNet121 for Diabetic Retinopathy
# Target: 96.96% accuracy with balanced dataset6

echo "🚀 CLEAN ENSEMBLE TRAINING"
echo "Models: EfficientNetB2 (96.27%) + ResNet50 (94.95%) + DenseNet121 (91.21%)"
echo "Dataset: Balanced dataset6 (21,149 images, 4.6:1 ratio)"
echo "Target: 96.96% ensemble accuracy"
echo ""

# Check if balanced dataset6 exists
if [ ! -d "./dataset6" ]; then
    echo "❌ ERROR: Balanced dataset6 not found"
    echo "Please ensure dataset6 has been balanced using dataset_balancer.py"
    exit 1
fi

# Check dataset structure
for split in train val test; do
    if [ ! -d "./dataset6/$split" ]; then
        echo "❌ ERROR: Missing dataset6/$split directory"
        exit 1
    fi
done

# Verify class balance
echo "🔍 VERIFYING DATASET BALANCE:"
for split in train val test; do
    echo "--- $split ---"
    for class in 0 1 2 3 4; do
        count=$(ls "./dataset6/$split/$class/" 2>/dev/null | wc -l | tr -d ' ')
        echo "Class $class: $count images"
    done
done
echo ""

echo "🏥 MEDICAL-GRADE CONFIGURATION:"
echo "  ✅ CLAHE preprocessing enabled"
echo "  ✅ Class weights enabled (4.6:1 balanced)"
echo "  ✅ Medical-grade augmentation"
echo "  ✅ Early stopping with patience"
echo ""

echo "🔧 TRAINING PARAMETERS:"
echo "  📊 Epochs: 100 (with early stopping)"
echo "  📈 Batch size: 16"
echo "  🎯 Learning rate: 1e-4"
echo "  🏥 Target ensemble accuracy: 96.96%"
echo ""

echo "🚀 Starting clean ensemble training..."
echo "==========================================="

# Create output directory
mkdir -p ./clean_ensemble_results

python clean_ensemble_trainer.py \
    --dataset_path ./dataset6 \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --ensemble_weights 0.4 0.35 0.25 \
    --optimizer adam \
    --enable_clahe \
    --enable_class_weights \
    --patience 10 \
    --checkpoint_frequency 5 \
    --output_dir ./clean_ensemble_results \
    --experiment_name clean_ensemble_balanced_dataset6 \
    --seed 42

EXIT_CODE=$?

echo ""
echo "==========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Clean ensemble training completed successfully!"
    echo "🎯 Check results in ./clean_ensemble_results/ directory"
    echo ""

    echo "📁 Results verification:"
    if [ -d "./clean_ensemble_results" ]; then
        echo "   ✅ Results directory exists"

        if [ -f "./clean_ensemble_results/models/ensemble_best.pth" ]; then
            echo "   🏆 Best ensemble model saved"

            # Check model size
            model_size=$(du -h "./clean_ensemble_results/models/ensemble_best.pth" | cut -f1)
            echo "   📊 Model size: $model_size"
        else
            echo "   ❌ Best model not found"
        fi

        if [ -f "./clean_ensemble_results/results/training_results.json" ]; then
            echo "   📊 Training results saved"

            # Extract final accuracy if possible
            if command -v python3 &> /dev/null; then
                final_acc=$(python3 -c "
import json
try:
    with open('./clean_ensemble_results/results/training_results.json', 'r') as f:
        results = json.load(f)
    if 'final_results' in results and 'ensemble_accuracy' in results['final_results']:
        print(f\"{results['final_results']['ensemble_accuracy']:.2f}%\")
    elif 'best_accuracy' in results:
        print(f\"{results['best_accuracy']:.2f}%\")
    else:
        print('Available')
except:
    print('Available')
" 2>/dev/null)
                echo "   🎯 Final accuracy: $final_acc"
            fi
        fi
    fi

    echo ""
    echo "📋 Next steps:"
    echo "1. Check the training results in clean_ensemble_results/"
    echo "2. Review individual model performances"
    echo "3. If accuracy is <96.96%, consider:"
    echo "   - Increasing epochs"
    echo "   - Adding focal loss (--enable_focal_loss)"
    echo "   - Fine-tuning ensemble weights"
    echo "4. Use the ensemble for Phase 1.5 (Image Analysis)"

else
    echo "❌ Training failed with exit code: $EXIT_CODE"
    echo "🔍 Check logs above for error details"
    echo ""
    echo "Common issues:"
    echo "- GPU memory: Reduce batch_size to 8 or 4"
    echo "- Missing dependencies: pip install opencv-python scikit-learn"
    echo "- Dataset path: Ensure ./dataset6 exists with train/val/test"
fi

exit $EXIT_CODE