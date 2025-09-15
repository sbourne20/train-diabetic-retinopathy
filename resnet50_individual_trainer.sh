#!/bin/bash

# Individual ResNet50 Training Script
# Diabetic Retinopathy Classification - Target: 94.95% accuracy
# Part of ensemble approach: EfficientNetB2 (79.56% ✅) + ResNet50 + DenseNet121

echo "🔥 RESNET50 INDIVIDUAL TRAINING"
echo "Target: 94.95% individual accuracy (research literature)"
echo "Status: EfficientNetB2 completed (79.56%) ✅"
echo "Next: DenseNet121 after ResNet50 completion"
echo ""

# Check if dataset exists
if [ ! -d "./dataset5" ]; then
    echo "❌ ERROR: Dataset directory './dataset5' not found"
    echo "Please ensure the dataset is available in the current directory"
    exit 1
fi

# Check if medical terms file exists
if [ ! -f "data/medical_terms_type1.json" ]; then
    echo "⚠️  WARNING: Medical terms file 'data/medical_terms_type1.json' not found"
    echo "Training will continue but medical validation may be limited"
fi

# Verify EfficientNetB2 is completed
if [ ! -f "./results/checkpoints/best_efficientnetb2.pth" ]; then
    echo "⚠️  WARNING: EfficientNetB2 checkpoint not found"
    echo "Expected: ./results/checkpoints/best_efficientnetb2.pth"
    echo "Ensemble will be incomplete without all three models"
fi

echo "🏥 MEDICAL-GRADE CONFIGURATION:"
echo "  ✅ CLAHE preprocessing enabled"
echo "  ✅ SMOTE class balancing enabled"
echo "  ✅ Focal loss for imbalanced classes"
echo "  ✅ Class weights optimization"
echo "  ✅ Medical-grade validation thresholds"
echo ""

echo "🔧 RESNET50 TRAINING PARAMETERS:"
echo "  🏗️  Architecture: ResNet50 (pre-trained)"
echo "  📊 Epochs: 100"
echo "  🎯 Target accuracy: ≥94.95%"
echo "  🏥 Medical threshold: ≥90%"
echo "  📈 Validation frequency: Every epoch"
echo "  💾 Checkpoint frequency: Every 5 epochs"
echo "  📦 Batch size: 6 (optimized for V100)"
echo ""

echo "🚀 Starting ResNet50 training..."
echo "=========================================="

# Ensure output directory structure exists
echo "📁 Creating ResNet50 output directories..."
mkdir -p ./results
mkdir -p ./results/checkpoints
mkdir -p ./results/logs
mkdir -p ./results/resnet50
echo "✅ Output directories created: $(pwd)/results"

python individual_model_trainer.py \
    --model resnet50 \
    --dataset_path ./dataset5 \
    --output_dir ./results \
    --epochs 100 \
    --batch_size 6 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --dropout 0.3 \
    --max_grad_norm 1.0 \
    --patience 10 \
    --min_delta 0.001 \
    --scheduler cosine \
    --warmup_epochs 5 \
    --enable_clahe \
    --enable_smote \
    --enable_focal_loss \
    --enable_class_weights \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --medical_terms data/medical_terms_type1.json \
    --experiment_name resnet50_individual_dr_classification

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ ResNet50 training completed successfully!"
    echo "🎯 Check results in ./results/ directory"
    echo ""
    echo "📁 ResNet50 checkpoint verification:"
    if [ -f "./results/checkpoints/best_resnet50.pth" ]; then
        echo "   ✅ ResNet50 model saved: best_resnet50.pth"

        # Use model analyzer to check performance
        if [ -f "model_analyzer.py" ]; then
            echo ""
            echo "📊 Running model analysis..."
            python model_analyzer.py --model ./results/checkpoints/best_resnet50.pth
        fi
    else
        echo "   ❌ ResNet50 checkpoint not found"
    fi

    echo ""
    echo "🎯 ENSEMBLE PROGRESS:"
    echo "   Phase 1: EfficientNetB2 $([ -f "./results/checkpoints/best_efficientnetb2.pth" ] && echo "✅ (79.56%)" || echo "❌")"
    echo "   Phase 2: ResNet50 $([ -f "./results/checkpoints/best_resnet50.pth" ] && echo "✅ (COMPLETED)" || echo "❌")"
    echo "   Phase 3: DenseNet121 ⏳ (Next: ./densenet121_individual_trainer.sh)"
    echo ""
    echo "📋 NEXT STEPS:"
    echo "   1. Run: ./densenet121_individual_trainer.sh"
    echo "   2. After completion, run ensemble combination script"
    echo "   3. Validate ensemble performance (target: 96.96%)"

else
    echo "❌ ResNet50 training failed with exit code: $EXIT_CODE"
    echo "🔍 Check logs above for error details"
    echo ""
    echo "💡 Troubleshooting suggestions:"
    echo "   1. Check GPU memory availability (V100 16GB required)"
    echo "   2. Verify dataset path is correct (./dataset5)"
    echo "   3. Ensure individual_model_trainer.py exists"
    echo "   4. Review Python dependencies and environment"
    echo ""
    echo "🔄 To retry after fixing issues:"
    echo "   ./resnet50_individual_trainer.sh"
fi

exit $EXIT_CODE