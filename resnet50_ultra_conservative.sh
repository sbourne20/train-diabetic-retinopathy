#!/bin/bash

# ResNet50 Ultra-Conservative Anti-Overfitting Training Script
# Maximum regularization settings to prevent overfitting at all costs
# Based on analysis: Previous attempts overfitted despite aggressive settings

echo "🛡️ RESNET50 ULTRA-CONSERVATIVE TRAINING"
echo "Previous attempts: 79.56% (EfficientNet) vs 72.73% (ResNet50 overfitted)"
echo "Ultra-aggressive regularization: Maximum dropout, minimum learning rate"
echo "Goal: Stable generalization even if slower convergence"
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

echo "🛡️ ULTRA-CONSERVATIVE ANTI-OVERFITTING SETTINGS:"
echo "  🐌 Learning Rate: 1e-5 (extremely slow - 5x reduction)"
echo "  🛡️ Dropout: 0.7 + 0.56 (dual dropout layers - maximum regularization)"
echo "  ⚖️ Weight Decay: 5e-3 (5x stronger regularization)"
echo "  🛑 Early Stopping: 3 epochs patience (immediate stop)"
echo "  📊 Min Delta: 0.01 (requires significant improvement)"
echo "  🎭 Label Smoothing: 0.2 (maximum uncertainty injection)"
echo "  ✂️ Gradient Clipping: 0.5 (strict gradient control)"
echo "  🏗️ Architecture: Dual FC layers with heavy dropout"
echo ""

echo "📚 THEORETICAL APPROACH:"
echo "  • Sacrifice training speed for generalization stability"
echo "  • Force model to learn robust patterns, not memorize"
echo "  • Stop immediately if validation doesn't improve"
echo "  • Accept slower convergence to prevent overfitting"
echo ""

echo "🏥 MEDICAL-GRADE CONFIGURATION:"
echo "  ✅ CLAHE preprocessing enabled"
echo "  ✅ SMOTE class balancing enabled"
echo "  ✅ Focal loss for imbalanced classes"
echo "  ✅ Class weights optimization"
echo "  ✅ Medical-grade validation thresholds"
echo ""

echo "🔧 ULTRA-CONSERVATIVE TRAINING PARAMETERS:"
echo "  🏗️  Architecture: ResNet50 (pre-trained) + Dual FC"
echo "  📊 Epochs: 100 (with aggressive early stopping)"
echo "  🎯 Target accuracy: >79.56% (beat EfficientNetB2)"
echo "  🏥 Medical threshold: ≥90%"
echo "  📈 Validation frequency: Every epoch"
echo "  💾 Checkpoint frequency: Every 5 epochs"
echo "  📦 Batch size: 6 (optimized for V100)"
echo "  🛡️ Overfitting protection: MAXIMUM"
echo ""

echo "⚠️ EXPECTED BEHAVIOR:"
echo "  • Very slow initial learning (by design)"
echo "  • Train-val gap should stay <10% throughout"
echo "  • May stop early (3-15 epochs) if no improvement"
echo "  • Quality over speed - stable generalization priority"
echo ""

echo "🚀 Starting ResNet50 ultra-conservative training..."
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
    --learning_rate 1e-5 \
    --weight_decay 5e-3 \
    --dropout 0.7 \
    --max_grad_norm 0.5 \
    --patience 3 \
    --min_delta 0.01 \
    --scheduler plateau \
    --warmup_epochs 5 \
    --enable_clahe \
    --enable_smote \
    --enable_focal_loss \
    --enable_class_weights \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --medical_terms data/medical_terms_type1.json \
    --experiment_name resnet50_ultra_conservative_max_regularization

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ ResNet50 ultra-conservative training completed!"
    echo "🎯 Check results in ./results/ directory"
    echo ""
    echo "📁 ResNet50 checkpoint verification:"
    if [ -f "./results/checkpoints/best_resnet50.pth" ]; then
        echo "   ✅ ResNet50 model saved: best_resnet50.pth"

        # Use model analyzer to check performance and overfitting
        if [ -f "model_analyzer.py" ]; then
            echo ""
            echo "📊 Running comprehensive model analysis..."
            python model_analyzer.py --model ./results/checkpoints/best_resnet50.pth --verbose
        fi
    else
        echo "   ❌ ResNet50 checkpoint not found"
    fi

    echo ""
    echo "🎯 ANTI-OVERFITTING SUCCESS METRICS:"
    echo "   ✅ Target: Train-Val gap <15% throughout training"
    echo "   ✅ Target: Validation accuracy >79.56%"
    echo "   ✅ Target: No declining validation accuracy pattern"
    echo "   ✅ Target: Stable convergence without memorization"
    echo ""
    echo "🎯 PERFORMANCE COMPARISON:"
    echo "   EfficientNetB2: 79.56% (baseline to beat)"
    echo "   ResNet50 (1st): 73.44% (overfitted at epoch 11)"
    echo "   ResNet50 (2nd): 72.73% (overfitted at epoch 7)"
    echo "   ResNet50 (3rd): $([ -f "./results/checkpoints/best_resnet50.pth" ] && echo "CHECK ANALYSIS ABOVE" || echo "TRAINING FAILED")"
    echo ""
    echo "🎯 ENSEMBLE PROGRESS:"
    echo "   Phase 1: EfficientNetB2 $([ -f "./results/checkpoints/best_efficientnetb2.pth" ] && echo "✅ (79.56%)" || echo "❌")"
    echo "   Phase 2: ResNet50 $([ -f "./results/checkpoints/best_resnet50.pth" ] && echo "✅ (ULTRA-CONSERVATIVE)" || echo "❌")"
    echo "   Phase 3: DenseNet121 ⏳ (Next: apply same ultra-conservative approach)"
    echo ""
    echo "📋 NEXT STEPS:"
    if [ -f "./results/checkpoints/best_resnet50.pth" ]; then
        echo "   1. Analyze train-val gap from model analyzer"
        echo "   2. If successful (>79.56%), proceed to DenseNet121"
        echo "   3. Apply same ultra-conservative settings to DenseNet121"
        echo "   4. Create ensemble with all three models"
    else
        echo "   1. Review training logs for early stopping trigger"
        echo "   2. Consider if even more conservative settings needed"
        echo "   3. Alternative: Accept current best models and create ensemble"
    fi

else
    echo "❌ ResNet50 ultra-conservative training failed with exit code: $EXIT_CODE"
    echo "🔍 Check logs above for error details"
    echo ""
    echo "💡 Analysis suggestions:"
    echo "   1. Check if early stopping triggered too quickly"
    echo "   2. Verify GPU memory with ultra-regularized model"
    echo "   3. Review if learning rate is too low for convergence"
    echo "   4. Consider batch size adjustment for stability"
    echo ""
    echo "🔄 Fallback options:"
    echo "   1. Use best existing model (EfficientNetB2: 79.56%)"
    echo "   2. Proceed to DenseNet121 with current settings"
    echo "   3. Create ensemble with available models"
fi

exit $EXIT_CODE