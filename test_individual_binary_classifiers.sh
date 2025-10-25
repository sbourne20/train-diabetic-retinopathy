#!/bin/bash

# Test Individual Binary Classifiers (SEResNeXt50 OVO)
echo "🧪 Testing Individual Binary Classifiers (10 classifiers: 0-1, 0-2, ..., 3-4)"
echo "=============================================================================="

python3 test_binary_classifiers.py \
    --models_dir ./seresnext50_5class_results/models \
    --dataset_path ./dataset_eyepacs_5class_balanced_enhanced_v2 \
    --img_size 224 \
    --output ./seresnext50_5class_results/binary_test_results.json 2>&1 | tee binary_classifier_test_log.txt

echo ""
echo "✅ Binary classifier testing completed!"
echo "📊 Results saved to: ./seresnext50_5class_results/binary_test_results.json"
