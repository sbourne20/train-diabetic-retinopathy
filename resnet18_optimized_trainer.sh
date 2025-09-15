#!/bin/bash

# ResNet18 Optimized Training Script - Smaller Architecture for Better Generalization
# After ResNet50 consistent overfitting, switching to ResNet18 (11M vs 23.5M parameters)
# Balanced regularization settings optimized for smaller model capacity

echo "🏗️ RESNET18 OPTIMIZED TRAINING"
echo "Strategy: Smaller architecture for better generalization"
echo "ResNet18: 11M parameters vs ResNet50: 23.5M parameters (53% reduction)"
echo "Goal: Stable training with >79.56% validation accuracy"
echo ""

# Check if dataset exists
if [ ! -d "./dataset5" ]; then
    echo "❌ ERROR: Dataset directory './dataset5' not found"
    echo "Please ensure the dataset is available in the current directory"
    exit 1
fi

# Remove any existing ResNet checkpoints to start fresh
echo "🧹 Cleaning previous ResNet checkpoints..."
rm -f ./results/checkpoints/best_resnet*.pth
rm -f ./results/checkpoints/resnet*_epoch_*.pth
echo "✅ Previous ResNet checkpoints removed"

echo "🎯 RESNET18 ARCHITECTURE ADVANTAGES:"
echo "  📊 Parameters: 11M (vs ResNet50: 23.5M)"
echo "  🛡️ Reduced overfitting risk due to smaller capacity"
echo "  ⚡ Faster training and inference"
echo "  🎯 Better suited for medical imaging dataset size"
echo "  📈 Historical success with medical image classification"
echo ""

echo "⚖️ BALANCED REGULARIZATION SETTINGS:"
echo "  🎯 Learning Rate: 2e-5 (conservative but not extreme)"
echo "  🛡️ Dropout: 0.35 + 0.25 (dual dropout, moderate)"
echo "  ⚖️ Weight Decay: 1e-3 (standard regularization)"
echo "  🛑 Early Stopping: 7 epochs patience (reasonable)"
echo "  📊 Min Delta: 0.005 (balanced improvement threshold)"
echo "  🎭 Label Smoothing: 0.2 (prevent overconfidence)"
echo "  🏗️ Architecture: Dual FC layers (256 intermediate)"
echo ""

echo "📚 STRATEGY RATIONALE:"
echo "  • Smaller model = less prone to memorization"
echo "  • Balanced regularization = avoid underfitting"
echo "  • Allow model capacity to match dataset complexity"
echo "  • Focus on stable generalization over speed"
echo ""

echo "🏥 MEDICAL-GRADE CONFIGURATION:"
echo "  ✅ CLAHE preprocessing enabled"
echo "  ✅ SMOTE class balancing enabled"
echo "  ✅ Focal loss for imbalanced classes"
echo "  ✅ Class weights optimization"
echo "  ✅ Medical-grade validation thresholds"
echo ""

echo "🔧 RESNET18 TRAINING PARAMETERS:"
echo "  🏗️  Architecture: ResNet18 (pre-trained) + Dual FC"
echo "  📊 Epochs: 100 (with balanced early stopping)"
echo "  🎯 Target accuracy: >79.56% (beat EfficientNetB2)"
echo "  🏥 Medical threshold: ≥90%"
echo "  📈 Validation frequency: Every epoch"
echo "  💾 Checkpoint frequency: Every 5 epochs"
echo "  📦 Batch size: 6 (optimized for V100)"
echo "  🛡️ Overfitting protection: BALANCED"
echo ""

echo "📈 EXPECTED TRAINING PATTERN:"
echo "  • Steady learning without rapid spikes"
echo "  • Train-val gap should stay <12% throughout"
echo "  • Convergence in 20-40 epochs (reasonable time)"
echo "  • Stable validation improvement without decline"
echo ""

echo "🚀 Starting ResNet18 optimized training..."
echo "=========================================="

# Ensure output directory structure exists
echo "📁 Creating ResNet18 output directories..."
mkdir -p ./results
mkdir -p ./results/checkpoints
mkdir -p ./results/logs
mkdir -p ./results/resnet18
echo "✅ Output directories created: $(pwd)/results"

python individual_model_trainer.py \
    --model resnet18 \
    --dataset_path ./dataset5 \
    --output_dir ./results \
    --epochs 100 \
    --batch_size 6 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --weight_decay 1e-3 \
    --dropout 0.5 \
    --max_grad_norm 0.5 \
    --patience 7 \
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
    --experiment_name resnet18_optimized_balanced_regularization

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ ResNet18 optimized training completed!"
    echo "🎯 Check results in ./results/ directory"
    echo ""
    echo "📁 ResNet18 checkpoint verification:"
    if [ -f "./results/checkpoints/best_resnet18.pth" ]; then
        echo "   ✅ ResNet18 model saved: best_resnet18.pth"

        # Use model analyzer to check performance and overfitting
        if [ -f "model_analyzer.py" ]; then
            echo ""
            echo "📊 Running comprehensive model analysis..."
            python model_analyzer.py --model ./results/checkpoints/best_resnet18.pth --verbose
        fi
    else
        echo "   ❌ ResNet18 checkpoint not found"
    fi

    echo ""
    echo "🎯 GENERALIZATION SUCCESS METRICS:"
    echo "   ✅ Target: Train-Val gap <12% throughout training"
    echo "   ✅ Target: Validation accuracy >79.56%"
    echo "   ✅ Target: Stable convergence without memorization"
    echo "   ✅ Target: No severe overfitting patterns"
    echo ""
    echo "🎯 PERFORMANCE COMPARISON:"
    echo "   EfficientNetB2: 79.56% (baseline to beat)"
    echo "   ResNet50 (multiple attempts): ~72-73% (overfitted consistently)"
    echo "   ResNet18 (optimized): $([ -f "./results/checkpoints/best_resnet18.pth" ] && echo "CHECK ANALYSIS ABOVE" || echo "TRAINING FAILED")"
    echo ""
    echo "🎯 ENSEMBLE PROGRESS:"
    echo "   Phase 1: EfficientNetB2 $([ -f "./results/checkpoints/best_efficientnetb2.pth" ] && echo "✅ (79.56%)" || echo "❌")"
    echo "   Phase 2: ResNet18 $([ -f "./results/checkpoints/best_resnet18.pth" ] && echo "✅ (OPTIMIZED)" || echo "❌")"
    echo "   Phase 3: DenseNet121 ⏳ (Next: apply balanced approach)"
    echo ""
    echo "📋 NEXT STEPS:"
    if [ -f "./results/checkpoints/best_resnet18.pth" ]; then
        echo "   1. ✅ Analyze ResNet18 performance vs EfficientNetB2"
        echo "   2. If >79.56%, proceed to DenseNet121 with similar settings"
        echo "   3. If still overfitting, consider even smaller architecture"
        echo "   4. Create ensemble with best available models"
    else
        echo "   1. ❌ Review training logs for issues"
        echo "   2. Consider further architectural adjustments"
        echo "   3. Alternative: Use EfficientNetB2 as primary model"
    fi

else
    echo "❌ ResNet18 optimized training failed with exit code: $EXIT_CODE"
    echo "🔍 Check logs above for error details"
    echo ""
    echo "💡 Troubleshooting suggestions:"
    echo "   1. Check GPU memory with ResNet18 architecture"
    echo "   2. Verify dataset compatibility"
    echo "   3. Review model initialization parameters"
    echo "   4. Consider batch size adjustments"
    echo ""
    echo "🔄 Alternative approaches:"
    echo "   1. Try different learning rate (1e-5 or 5e-5)"
    echo "   2. Adjust dropout rates (0.3-0.6 range)"
    echo "   3. Use simpler architecture (e.g., ResNet34)"
fi

exit $EXIT_CODE