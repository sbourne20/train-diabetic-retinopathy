#!/bin/bash

# Evaluate Multi-Architecture Ensemble on Test Set
echo "🧪 Evaluating Multi-Architecture Ensemble on Test Set"
echo "========================================================"
echo ""
echo "Ensemble Models: EfficientNetB2 + ResNet50 + DenseNet121"
echo "Expected Ensemble Accuracy: 96.96%"
echo ""

python3 ensemble_5class_trainer.py \
    --mode evaluate \
    --base_models efficientnet_b2 resnet50 densenet121 \
    --dataset_path ./dataset_eyepacs_5class_balanced_enhanced_v2 \
    --num_classes 5 \
    --output_dir ./ensemble_multi_arch_results \
    --experiment_name 5class_ensemble_efficientnet_resnet_densenet 2>&1 | tee evaluation_log_multi_arch_ensemble.txt