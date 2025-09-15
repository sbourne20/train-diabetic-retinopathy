#!/bin/bash

# Individual DenseNet121 Training Script
# Diabetic Retinopathy Classification - Target: 91.21% accuracy
# Part of ensemble approach: EfficientNetB2 (79.56% ✅) + ResNet50 + DenseNet121

echo "🌿 DENSENET121 INDIVIDUAL TRAINING"
echo "Target: 91.21% individual accuracy (research literature)"
echo "Status: EfficientNetB2 ✅ + ResNet50 completion expected"
echo "Final step before ensemble combination"
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

# Verify previous models are completed
echo "🔍 Checking ensemble prerequisites..."
if [ ! -f "./results/checkpoints/best_efficientnetb2.pth" ]; then
    echo "⚠️  WARNING: EfficientNetB2 checkpoint not found"
    echo "Expected: ./results/checkpoints/best_efficientnetb2.pth"
fi

if [ ! -f "./results/checkpoints/best_resnet50.pth" ]; then
    echo "⚠️  WARNING: ResNet50 checkpoint not found"
    echo "Expected: ./results/checkpoints/best_resnet50.pth"
    echo "Please complete ResNet50 training first: ./resnet50_individual_trainer.sh"
fi

echo "🏥 MEDICAL-GRADE CONFIGURATION:"
echo "  ✅ CLAHE preprocessing enabled"
echo "  ✅ SMOTE class balancing enabled"
echo "  ✅ Focal loss for imbalanced classes"
echo "  ✅ Class weights optimization"
echo "  ✅ Medical-grade validation thresholds"
echo ""

echo "🔧 DENSENET121 TRAINING PARAMETERS:"
echo "  🏗️  Architecture: DenseNet121 (pre-trained)"
echo "  📊 Epochs: 100"
echo "  🎯 Target accuracy: ≥91.21%"
echo "  🏥 Medical threshold: ≥90%"
echo "  📈 Validation frequency: Every epoch"
echo "  💾 Checkpoint frequency: Every 5 epochs"
echo "  📦 Batch size: 6 (optimized for V100)"
echo ""

echo "🚀 Starting DenseNet121 training..."
echo "=========================================="

# Ensure output directory structure exists
echo "📁 Creating DenseNet121 output directories..."
mkdir -p ./results
mkdir -p ./results/checkpoints
mkdir -p ./results/logs
mkdir -p ./results/densenet121
echo "✅ Output directories created: $(pwd)/results"

python individual_model_trainer.py \
    --model densenet121 \
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
    --experiment_name densenet121_individual_dr_classification

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ DenseNet121 training completed successfully!"
    echo "🎯 Check results in ./results/ directory"
    echo ""
    echo "📁 DenseNet121 checkpoint verification:"
    if [ -f "./results/checkpoints/best_densenet121.pth" ]; then
        echo "   ✅ DenseNet121 model saved: best_densenet121.pth"

        # Use model analyzer to check performance
        if [ -f "model_analyzer.py" ]; then
            echo ""
            echo "📊 Running model analysis..."
            python model_analyzer.py --model ./results/checkpoints/best_densenet121.pth
        fi
    else
        echo "   ❌ DenseNet121 checkpoint not found"
    fi

    echo ""
    echo "🎯 ENSEMBLE COMPLETION CHECK:"
    efficientnet_status="❌"
    resnet50_status="❌"
    densenet121_status="❌"

    if [ -f "./results/checkpoints/best_efficientnetb2.pth" ]; then
        efficientnet_status="✅ (79.56%)"
    fi
    if [ -f "./results/checkpoints/best_resnet50.pth" ]; then
        resnet50_status="✅ (COMPLETED)"
    fi
    if [ -f "./results/checkpoints/best_densenet121.pth" ]; then
        densenet121_status="✅ (COMPLETED)"
    fi

    echo "   Phase 1: EfficientNetB2 $efficientnet_status"
    echo "   Phase 2: ResNet50 $resnet50_status"
    echo "   Phase 3: DenseNet121 $densenet121_status"

    # Check if all models are ready for ensemble
    if [ -f "./results/checkpoints/best_efficientnetb2.pth" ] && [ -f "./results/checkpoints/best_resnet50.pth" ] && [ -f "./results/checkpoints/best_densenet121.pth" ]; then
        echo ""
        echo "🎉 ALL INDIVIDUAL MODELS COMPLETED!"
        echo "📋 READY FOR ENSEMBLE COMBINATION:"
        echo "   1. Run: ./ensemble_combiner.sh"
        echo "   2. Validate ensemble performance (target: 96.96%)"
        echo "   3. Analyze final results with model_analyzer.py"
    else
        echo ""
        echo "⏳ MISSING MODELS - Complete remaining training:"
        if [ ! -f "./results/checkpoints/best_efficientnetb2.pth" ]; then
            echo "   • EfficientNetB2: Use existing checkpoint from previous training"
        fi
        if [ ! -f "./results/checkpoints/best_resnet50.pth" ]; then
            echo "   • ResNet50: Run ./resnet50_individual_trainer.sh"
        fi
    fi

else
    echo "❌ DenseNet121 training failed with exit code: $EXIT_CODE"
    echo "🔍 Check logs above for error details"
    echo ""
    echo "💡 Troubleshooting suggestions:"
    echo "   1. Check GPU memory availability (V100 16GB required)"
    echo "   2. Verify dataset path is correct (./dataset5)"
    echo "   3. Ensure individual_model_trainer.py exists"
    echo "   4. Review Python dependencies and environment"
    echo ""
    echo "🔄 To retry after fixing issues:"
    echo "   ./densenet121_individual_trainer.sh"
fi

exit $EXIT_CODE