#!/bin/bash

echo "======================================================================"
echo "3-CLASS DR TRAINING SYSTEM - VERIFICATION"
echo "======================================================================"
echo ""

echo "📁 Checking files..."
if [ -f "ensemble_3class_trainer.py" ]; then
    echo "  ✅ ensemble_3class_trainer.py exists"
    lines=$(wc -l < ensemble_3class_trainer.py)
    echo "     Lines: $lines"
else
    echo "  ❌ ensemble_3class_trainer.py NOT FOUND"
fi

if [ -f "train_3class_densenet.sh" ]; then
    echo "  ✅ train_3class_densenet.sh exists"
    if [ -x "train_3class_densenet.sh" ]; then
        echo "     Executable: YES"
    else
        echo "     Executable: NO (run: chmod +x train_3class_densenet.sh)"
    fi
else
    echo "  ❌ train_3class_densenet.sh NOT FOUND"
fi

echo ""
echo "📊 Checking dataset..."
DATASET_PATH="./dataset_eyepacs_3class_balanced"
if [ -d "$DATASET_PATH" ]; then
    echo "  ✅ Dataset directory exists: $DATASET_PATH"

    # Check splits
    for split in train val test; do
        if [ -d "$DATASET_PATH/$split" ]; then
            echo "  ✅ $split/ exists"

            # Check classes
            for class in NORMAL NPDR PDR; do
                if [ -d "$DATASET_PATH/$split/$class" ]; then
                    count=$(find "$DATASET_PATH/$split/$class" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l | tr -d ' ')
                    echo "     └─ $class: $count images"
                else
                    echo "     └─ ❌ $class NOT FOUND"
                fi
            done
        else
            echo "  ❌ $split/ NOT FOUND"
        fi
    done
else
    echo "  ❌ Dataset directory NOT FOUND: $DATASET_PATH"
fi

echo ""
echo "🔧 Checking Python syntax..."
if python3 -m py_compile ensemble_3class_trainer.py 2>/dev/null; then
    echo "  ✅ Python syntax is valid"
else
    echo "  ❌ Python syntax error detected"
fi

echo ""
echo "🔧 Checking bash syntax..."
if bash -n train_3class_densenet.sh 2>/dev/null; then
    echo "  ✅ Bash syntax is valid"
else
    echo "  ❌ Bash syntax error detected"
fi

echo ""
echo "📋 Key Configuration Parameters:"
echo "  • Classes: 3 (NORMAL, NPDR, PDR)"
echo "  • Model: DenseNet121"
echo "  • Image size: 299×299"
echo "  • Batch size: 10 (V100 16GB optimized)"
echo "  • Learning rate: 1e-4"
echo "  • Epochs: 100"
echo "  • Target accuracy: 95%+"
echo "  • Class weights: 0.515 (NORMAL), 1.323 (NPDR), 3.321 (PDR)"
echo "  • Dropout: 0.3"
echo "  • CLAHE: Enabled (clip=2.5)"
echo "  • Focal loss: alpha=2.5, gamma=3.0"
echo "  • Augmentation: 25° rotation, 20% brightness/contrast"
echo ""
echo "🚀 Ready to Train!"
echo "  Run: ./train_3class_densenet.sh"
echo "======================================================================"
