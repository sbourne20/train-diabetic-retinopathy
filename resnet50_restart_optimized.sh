#!/bin/bash

# ResNet50 Optimized Restart Training Script
# Anti-overfitting configuration based on previous training analysis
# Fixed settings: Lower LR, Higher dropout, Better regularization, Early stopping

echo "🔥 RESNET50 OPTIMIZED RESTART TRAINING"
echo "Previous run: 73.44% validation (overfitted at epoch 11)"
echo "Fixed settings: Anti-overfitting configuration applied"
echo "Target: >79.56% (beat EfficientNetB2) with stable generalization"
echo ""

# Check if dataset exists
if [ ! -d "./dataset5" ]; then
    echo "❌ ERROR: Dataset directory './dataset5' not found"
    echo "Please ensure the dataset is available in the current directory"
    exit 1
fi

# Remove previous ResNet50 checkpoints to start fresh
echo "🧹 Cleaning previous ResNet50 checkpoints..."
rm -f ./results/checkpoints/best_resnet50.pth
rm -f ./results/checkpoints/resnet50_epoch_*.pth
echo "✅ Previous ResNet50 checkpoints removed"

echo "🛡️ ANTI-OVERFITTING CONFIGURATION:"
echo "  🎯 Learning Rate: 5e-5 (reduced from 1e-4)"
echo "  🛡️ Dropout: 0.5 (increased from 0.3)"
echo "  ⚖️ Weight Decay: 1e-3 (increased from 1e-4)"
echo "  🛑 Early Stopping: 5 epochs patience (reduced from 10)"
echo "  📊 Min Delta: 0.005 (increased sensitivity)"
echo "  🎭 Label Smoothing: 0.1 (prevent overconfidence)"
echo "  📈 Scheduler: Conservative cosine annealing"
echo ""

echo "🏥 MEDICAL-GRADE CONFIGURATION:"
echo "  ✅ CLAHE preprocessing enabled"
echo "  ✅ SMOTE class balancing enabled"
echo "  ✅ Focal loss for imbalanced classes"
echo "  ✅ Class weights optimization"
echo "  ✅ Medical-grade validation thresholds"
echo ""

echo "🔧 OPTIMIZED TRAINING PARAMETERS:"
echo "  🏗️  Architecture: ResNet50 (pre-trained)"
echo "  📊 Epochs: 100 (with early stopping)"
echo "  🎯 Target accuracy: >79.56% (beat EfficientNetB2)"
echo "  🏥 Medical threshold: ≥90%"
echo "  📈 Validation frequency: Every epoch"
echo "  💾 Checkpoint frequency: Every 5 epochs"
echo "  📦 Batch size: 6 (optimized for V100)"
echo "  🛡️ Overfitting protection: ENABLED"
echo ""

echo "🚀 Starting ResNet50 optimized training..."
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
    --learning_rate 5e-5 \
    --weight_decay 1e-3 \
    --dropout 0.5 \
    --max_grad_norm 1.0 \
    --patience 5 \
    --min_delta 0.005 \
    --scheduler cosine \
    --warmup_epochs 5 \
    --enable_clahe \
    --enable_smote \
    --enable_focal_loss \
    --enable_class_weights \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --medical_terms data/medical_terms_type1.json \
    --experiment_name resnet50_optimized_anti_overfitting

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ ResNet50 optimized training completed successfully!"
    echo "🎯 Check results in ./results/ directory"
    echo ""
    echo "📁 ResNet50 checkpoint verification:"
    if [ -f "./results/checkpoints/best_resnet50.pth" ]; then
        echo "   ✅ ResNet50 model saved: best_resnet50.pth"

        # Use model analyzer to check performance
        if [ -f "model_analyzer.py" ]; then
            echo ""
            echo "📊 Running model analysis..."
            python model_analyzer.py --model ./results/checkpoints/best_resnet50.pth --verbose
        fi
    else
        echo "   ❌ ResNet50 checkpoint not found"
    fi

    echo ""
    echo "🎯 PERFORMANCE COMPARISON:"
    echo "   EfficientNetB2: 79.56% (baseline)"
    echo "   ResNet50 (previous): 73.44% (overfitted)"
    echo "   ResNet50 (optimized): $([ -f "./results/checkpoints/best_resnet50.pth" ] && echo "CHECK ANALYSIS ABOVE" || echo "TRAINING FAILED")"
    echo ""
    echo "🎯 ENSEMBLE PROGRESS:"
    echo "   Phase 1: EfficientNetB2 $([ -f "./results/checkpoints/best_efficientnetb2.pth" ] && echo "✅ (79.56%)" || echo "❌")"
    echo "   Phase 2: ResNet50 $([ -f "./results/checkpoints/best_resnet50.pth" ] && echo "✅ (OPTIMIZED)" || echo "❌")"
    echo "   Phase 3: DenseNet121 ⏳ (Next: ./densenet121_individual_trainer.sh)"
    echo ""
    echo "📋 NEXT STEPS:"
    if [ -f "./results/checkpoints/best_resnet50.pth" ]; then
        echo "   1. ✅ ResNet50 training successful - proceed to DenseNet121"
        echo "   2. Run: ./densenet121_individual_trainer.sh"
        echo "   3. After completion, create ensemble combination script"
    else
        echo "   1. ❌ Check training logs for issues"
        echo "   2. Consider further hyperparameter adjustments"
    fi

else
    echo "❌ ResNet50 optimized training failed with exit code: $EXIT_CODE"
    echo "🔍 Check logs above for error details"
    echo ""
    echo "💡 Troubleshooting suggestions:"
    echo "   1. Check GPU memory availability (V100 16GB required)"
    echo "   2. Verify dataset path is correct (./dataset5)"
    echo "   3. Review anti-overfitting settings effectiveness"
    echo "   4. Consider further learning rate reduction"
    echo ""
    echo "🔄 To retry with different settings:"
    echo "   ./resnet50_restart_optimized.sh"
fi

exit $EXIT_CODE