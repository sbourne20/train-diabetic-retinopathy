#!/bin/bash
set -e

echo "🎯 DIRECT 5-CLASS ENSEMBLE TRAINER"
echo "=================================="
echo "Solving OVO overfitting issues with direct multi-class training"
echo "=================================="

# Configuration
DATASET_PATH="./dataset6"
OUTPUT_DIR="./ensemble_direct_results"
EXPERIMENT_NAME="direct_ensemble_dataset6_v1"

# Check if dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ Error: Dataset path not found: $DATASET_PATH"
    echo "Available datasets:"
    ls -la | grep dataset
    echo ""
    echo "Please update DATASET_PATH in this script or create symlink:"
    echo "ln -s /path/to/your/dataset ./dataset6"
    exit 1
fi

echo "📁 Dataset: $DATASET_PATH"
echo "📂 Output: $OUTPUT_DIR"
echo "🏷️ Experiment: $EXPERIMENT_NAME"

# Validate dataset structure
echo ""
echo "📊 Validating dataset structure..."
if [ ! -d "$DATASET_PATH/train" ] || [ ! -d "$DATASET_PATH/val" ] || [ ! -d "$DATASET_PATH/test" ]; then
    echo "❌ Error: Dataset must have train/val/test structure"
    echo "Current structure:"
    ls -la "$DATASET_PATH"
    exit 1
fi

# Count samples per split
echo "📈 Dataset statistics:"
echo "   Train: $(find $DATASET_PATH/train -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" | wc -l) images"
echo "   Val:   $(find $DATASET_PATH/val -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" | wc -l) images"
echo "   Test:  $(find $DATASET_PATH/test -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" | wc -l) images"

# Count classes
echo "   Classes in train: $(ls $DATASET_PATH/train | wc -l)"

echo ""
echo "🚀 Starting Direct Ensemble Training..."
echo "⚙️ Features enabled:"
echo "   - Overfitting prevention (early stopping, LR reduction)"
echo "   - CLAHE preprocessing"
echo "   - Focal loss for class imbalance"
echo "   - Class weights for severe DR"
echo "   - Enhanced regularization (dropout, weight decay)"
echo "   - Conservative learning rate (1e-5)"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training with comprehensive overfitting prevention
python ensemble_direct_trainer.py \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --experiment_name "$EXPERIMENT_NAME" \
    --epochs 50 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --weight_decay 1e-4 \
    --dropout 0.5 \
    --enable_overfitting_prevention \
    --early_stopping_patience 10 \
    --reduce_lr_patience 5 \
    --min_lr 1e-7 \
    --label_smoothing 0.1 \
    --enable_clahe \
    --augmentation_strength 0.3 \
    --enable_focal_loss \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    --enable_class_weights \
    --models mobilenet_v2 inception_v3 densenet121 \
    --img_size 224 \
    --num_classes 5 \
    --device cuda \
    --seed 42

# Check training success
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Direct ensemble training completed successfully!"
    echo ""
    echo "📁 Results saved to: $OUTPUT_DIR"
    echo "📄 Check these files:"
    echo "   - $OUTPUT_DIR/models/best_mobilenet_v2.pth"
    echo "   - $OUTPUT_DIR/models/best_inception_v3.pth"
    echo "   - $OUTPUT_DIR/models/best_densenet121.pth"
    echo "   - $OUTPUT_DIR/models/ensemble_best.pth"
    echo "   - $OUTPUT_DIR/results.json"
    echo "   - $OUTPUT_DIR/config.json"
    echo ""
    echo "🔍 Analyze results with:"
    echo "python model_analyzer.py --model $OUTPUT_DIR/models/ensemble_best.pth"
    echo ""
    echo "📊 Expected improvements over OVO:"
    echo "   ✅ Individual models: 88-92% accuracy (stable)"
    echo "   ✅ Ensemble: 90-94% accuracy (medical grade)"
    echo "   ✅ Overfitting gap: <5% (healthy)"
    echo "   ✅ No critical overfitting warnings"
else
    echo "❌ Training failed. Check the error messages above."
    exit 1
fi