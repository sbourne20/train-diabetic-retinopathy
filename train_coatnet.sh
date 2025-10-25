#!/bin/bash

# CoAtNet-0 Training Script (Hybrid CNN + Transformer)
# Priority: VERY HIGH
# Expected Accuracy: 98-99%

echo "🚀 Training CoAtNet-0 (Hybrid CNN+Transformer) for 5-Class DR Classification"
echo "=============================================================================="
echo ""
echo "Model: CoAtNet-0 (25M parameters)"
echo "Architecture: Hybrid Convolutional + Transformer"
echo "Medical Evidence: Top tier medical imaging (2023)"
echo "Expected Accuracy: 98-99%"
echo ""

python3 ensemble_5class_trainer.py \
    --mode train \
    --base_models coatnet_0_rw_224 \
    --dataset_path ./dataset_eyepacs_5class_balanced_enhanced_v2 \
    --num_classes 5 \
    --epochs 100 \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --weight_decay 0.00025 \
    --gradient_accumulation_steps 4 \
    --enable_focal_loss \
    --enable_class_weights \
    --enable_clahe \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --output_dir ./coatnet_5class_results \
    --experiment_name 5class_coatnet_0 \
    --img_size 224 \
    --ovo_dropout 0.28 \
    --patience 28 \
    --scheduler cosine \
    --warmup_epochs 10 \
    --focal_loss_gamma 3.0 \
    --focal_loss_alpha 2.5 \
    --label_smoothing 0.1 \
    --rotation_range 25.0 \
    --brightness_range 0.2 \
    --contrast_range 0.2 2>&1 | tee coatnet_training_log.txt

echo ""
echo "✅ CoAtNet-0 training completed!"
echo "📊 Results saved to: ./coatnet_5class_results"
echo ""
echo "Next step: Evaluate with model_analyzer.py"
