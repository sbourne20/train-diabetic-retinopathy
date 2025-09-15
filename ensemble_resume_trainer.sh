#!/bin/bash

# Ensemble Resume Training Script
# Resume from existing EfficientNetB2 checkpoint and continue with ResNet50 + DenseNet121
# Target: 96.96% accuracy with medical-grade validation

echo "🔄 ENSEMBLE RESUME TRAINING"
echo "Resuming from: EfficientNetB2 checkpoint (79.56% achieved)"
echo "Next models: ResNet50 (94.95%) + DenseNet121 (91.21%)"
echo "Target: 96.96% ensemble accuracy with medical-grade standards"
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

# Check if EfficientNetB2 checkpoint exists
if [ ! -f "./results/checkpoints/best_efficientnetb2.pth" ]; then
    echo "❌ ERROR: EfficientNetB2 checkpoint './results/checkpoints/best_efficientnetb2.pth' not found"
    echo "Available checkpoints:"
    ls -la ./results/checkpoints/efficientnetb2*.pth 2>/dev/null || echo "   No EfficientNetB2 checkpoints found"
    echo "Please ensure EfficientNetB2 training was completed first"
    exit 1
fi

echo "✅ RESUME STATUS:"
echo "  🔍 Found EfficientNetB2 checkpoint: ./results/checkpoints/best_efficientnetb2.pth"
echo "  📊 Previous accuracy: ~79.56%"
echo "  🎯 Continuing with ResNet50 training..."
echo ""

echo "🏥 MEDICAL-GRADE CONFIGURATION:"
echo "  ✅ CLAHE preprocessing enabled"
echo "  ✅ SMOTE class balancing enabled"
echo "  ✅ Focal loss for imbalanced classes"
echo "  ✅ Class weights optimization"
echo "  ✅ Medical-grade validation thresholds"
echo ""

echo "🔧 TRAINING PARAMETERS:"
echo "  📊 Epochs: 100"
echo "  📈 Validation frequency: Every epoch"
echo "  💾 Checkpoint frequency: Every 5 epochs"
echo "  🎯 Target accuracy: ≥96.96%"
echo "  🏥 Medical threshold: ≥90%"
echo "  🔄 Resume from: EfficientNetB2 checkpoint"
echo ""

echo "🚀 Resuming ensemble training..."
echo "=========================================="

# Ensure output directory structure exists
echo "📁 Verifying output directory structure..."
mkdir -p ./results
mkdir -p ./results/checkpoints
mkdir -p ./results/logs
mkdir -p ./results/results
echo "✅ Output directories verified: $(pwd)/results"

python ensemble_local_trainer.py \
    --mode train \
    --dataset_path ./dataset5 \
    --output_dir ./results \
    --epochs 100 \
    --batch_size 6 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --individual_dropout 0.3 0.3 0.3 \
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
    --resume_from_checkpoint ./results/checkpoints/best_efficientnetb2.pth

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Ensemble resume training completed successfully!"
    echo "🎯 Check results in ./results/ directory"
    echo "📊 Use ensemble_evaluator.py for comprehensive assessment"
    echo ""
    echo "📁 Final checkpoint verification:"
    if [ -d "./results/checkpoints" ]; then
        checkpoint_count=$(ls -1 ./results/checkpoints/*.pth 2>/dev/null | wc -l || echo "0")
        echo "   ✅ Checkpoints directory exists"
        echo "   📊 Found $checkpoint_count checkpoint files"

        # Check for individual model checkpoints
        if [ -f "./results/checkpoints/best_efficientnetb2.pth" ]; then
            echo "   ✅ EfficientNetB2: best_efficientnetb2.pth (79.56%)"
        fi
        if [ -f "./results/checkpoints/best_resnet50.pth" ]; then
            echo "   ✅ ResNet50: best_resnet50.pth"
        fi
        if [ -f "./results/checkpoints/best_densenet121.pth" ]; then
            echo "   ✅ DenseNet121: best_densenet121.pth"
        fi
        if [ -f "./results/checkpoints/ensemble_best.pth" ]; then
            echo "   🏆 Final ensemble model: ensemble_best.pth"
        fi
    else
        echo "   ❌ Checkpoints directory not found"
    fi

    echo ""
    echo "🎯 TRAINING SUMMARY:"
    echo "   Phase 1: EfficientNetB2 ✅ (79.56% - resumed from checkpoint)"
    echo "   Phase 2: ResNet50 $([ -f "./results/checkpoints/best_resnet50.pth" ] && echo "✅" || echo "❌")"
    echo "   Phase 3: DenseNet121 $([ -f "./results/checkpoints/best_densenet121.pth" ] && echo "✅" || echo "❌")"
    echo "   Phase 4: Ensemble $([ -f "./results/checkpoints/ensemble_best.pth" ] && echo "✅" || echo "❌")"

else
    echo "❌ Resume training failed with exit code: $EXIT_CODE"
    echo "🔍 Check logs above for error details"
    echo ""
    echo "💡 Troubleshooting suggestions:"
    echo "   1. Verify EfficientNetB2 checkpoint exists and is valid"
    echo "   2. Check GPU memory availability"
    echo "   3. Ensure dataset path is correct (./dataset5)"
    echo "   4. Review Python dependencies and environment"
fi

exit $EXIT_CODE