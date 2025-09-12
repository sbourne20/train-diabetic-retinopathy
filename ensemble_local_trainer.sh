#!/bin/bash

# Ensemble Multi-Architecture Training Script
# EfficientNetB2 + ResNet50 + DenseNet121 for Diabetic Retinopathy Classification
# Target: 96.96% accuracy with medical-grade validation

echo "🚀 ENSEMBLE MULTI-ARCHITECTURE TRAINING"
echo "Models: EfficientNetB2 (96.27%) + ResNet50 (94.95%) + DenseNet121 (91.21%)"
echo "Target: 96.96% ensemble accuracy with medical-grade standards"
echo ""

# Check if dataset exists
if [ ! -d "./dataset3_augmented_resized" ]; then
    echo "❌ ERROR: Dataset directory './dataset3_augmented_resized' not found"
    echo "Please ensure the dataset is available in the current directory"
    exit 1
fi

# Check if medical terms file exists
if [ ! -f "data/medical_terms_type1.json" ]; then
    echo "⚠️  WARNING: Medical terms file 'data/medical_terms_type1.json' not found"
    echo "Training will continue but medical validation may be limited"
fi

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
echo ""

echo "🚀 Starting ensemble training..."
echo "=========================================="

# Ensure output directory structure exists
echo "📁 Creating output directory structure..."
mkdir -p ./results
mkdir -p ./results/checkpoints
mkdir -p ./results/logs  
mkdir -p ./results/results
echo "✅ Output directories created: $(pwd)/results"

python ensemble_local_trainer.py \
    --mode train \
    --dataset_path ./dataset3_augmented_resized \
    --output_dir ./results \
    --epochs 100 \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --weight_decay 5e-3 \
    --individual_dropout 0.7 0.7 0.7 \
    --max_grad_norm 0.5 \
    --patience 8 \
    --min_delta 0.01 \
    --enable_clahe \
    --enable_smote \
    --enable_focal_loss \
    --enable_class_weights \
    --validation_frequency 1 \
    --checkpoint_frequency 4 \
    --medical_terms data/medical_terms_type1.json

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Ensemble training completed successfully!"
    echo "🎯 Check results in ./results/ directory"
    echo "📊 Use ensemble_evaluator.py for comprehensive assessment"
    echo ""
    echo "📁 Checkpoint verification:"
    if [ -d "./results/checkpoints" ]; then
        checkpoint_count=$(ls -1 ./results/checkpoints/*.pth 2>/dev/null | wc -l || echo "0")
        echo "   ✅ Checkpoints directory exists"
        echo "   📊 Found $checkpoint_count checkpoint files"
        if [ -f "./results/checkpoints/ensemble_best.pth" ]; then
            echo "   🏆 Best ensemble model saved"
        fi
    else
        echo "   ❌ Checkpoints directory not found"
    fi
else
    echo "❌ Training failed with exit code: $EXIT_CODE"
    echo "🔍 Check logs above for error details"
fi

exit $EXIT_CODE