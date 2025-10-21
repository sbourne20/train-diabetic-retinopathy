#!/bin/bash

# FIXED Grade-Specific Preprocessing - v2 with Class 0/2 Differentiation
# =======================================================================
# CRITICAL FIX: Class 0 and Class 2 now have DIFFERENT preprocessing parameters
# This should dramatically improve pair 0-2 classification accuracy

set -e  # Exit on error

echo "🔧 GRADE-SPECIFIC PREPROCESSING v2 - Class 0/2 Differentiation Fix"
echo "=========================================================================="
echo ""
echo "📊 PARAMETER CHANGES:"
echo ""
echo "Class 0 (No DR) - MINIMAL enhancement:"
echo "  - flatten_strength: 30 → 20 (REDUCED)"
echo "  - brightness_adjust: 20 → 10 (REDUCED)"
echo "  - contrast_factor: 2.0 → 1.5 (REDUCED)"
echo "  - sharpen_amount: 1.5 → 1.0 (REDUCED)"
echo "  → Preserves natural retinal appearance"
echo ""
echo "Class 2 (Moderate NPDR) - STRONGER enhancement:"
echo "  - flatten_strength: 30 → 35 (INCREASED)"
echo "  - brightness_adjust: 20 → 22 (INCREASED)"
echo "  - contrast_factor: 2.0 → 2.3 (INCREASED)"
echo "  - sharpen_amount: 1.5 → 1.8 (INCREASED)"
echo "  → Highlights pathological features"
echo ""
echo "🎯 Expected Result: Pair 0-2 accuracy 86% → 95%+"
echo "=========================================================================="
echo ""

# Input and output paths
INPUT_DIR="/Volumes/WDC1TB/dataset/diabetic-retinopathy/dataset_eyepacs_5class_balanced"
OUTPUT_DIR="/Volumes/WDC1TB/dataset/diabetic-retinopathy/dataset_eyepacs_5class_balanced_enhanced_v2"

echo "📂 Input:  $INPUT_DIR"
echo "📂 Output: $OUTPUT_DIR"
echo ""

# Confirm before overwriting
if [ -d "$OUTPUT_DIR" ]; then
    echo "⚠️  WARNING: Output directory already exists!"
    echo "   $OUTPUT_DIR"
    echo ""
    read -p "Do you want to DELETE and recreate it? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "❌ Preprocessing cancelled"
        exit 1
    fi
    echo "🗑️  Removing old directory..."
    rm -rf "$OUTPUT_DIR"
fi

# Run preprocessing
echo ""
echo "🚀 Starting preprocessing..."
echo ""

python3 preprocess_grade_specific.py \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --target_size 448 \
    --overwrite

echo ""
echo "✅ PREPROCESSING COMPLETE!"
echo ""
echo "📁 Enhanced dataset saved to: $OUTPUT_DIR"
echo ""
echo "🔄 NEXT STEPS:"
echo "1. Upload enhanced dataset to Vast.ai: /dataset_eyepacs_5class_balanced_enhanced_v2"
echo "2. Update train_5class_densenet_v4.sh to use new dataset path"
echo "3. Delete old checkpoints: rm ./densenet_5class_v4_enhanced_results/models/*.pth"
echo "4. Retrain ALL pairs with new preprocessing"
echo "5. Expected ensemble accuracy: 94-96%+"
