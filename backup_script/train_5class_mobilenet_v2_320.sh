#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# MobileNetV2 v2 - OPTIMAL RESOLUTION (320×320) - Anti-Overfitting
echo "🏥 5-CLASS MobileNetV2 v2 - OPTIMAL RESOLUTION (320×320)"
echo "===================================================================="
echo "🎯 Previous attempts:"
echo "   v2 (384×384):       Train 98.78%, Val 89.25%, Gap 9.53% 🚨"
echo "   v2_fixed (384×384): Train 97.36%, Val 89.34%, Gap 8.02% 🚨"
echo ""
echo "🎯 New strategy: Lower resolution + Stronger regularization"
echo "   Target: Train ~90-92%, Val ~89-91%, Gap <3% ✅"
echo ""

mkdir -p ./mobilenet_5class_v2_320_results

echo "📊 KEY CHANGES FROM v2_fixed:"
echo "  1. Resolution: 384×384 → 320×320 (-40% pixels, less memorization)"
echo "  2. Batch size: 16 → 20 (+25%, better gradient estimates)"
echo "  3. Dropout: 0.5 → 0.55 (+10%, more aggressive)"
echo "  4. Weight decay: 4e-4 → 5e-4 (+25%, stronger L2)"
echo "  5. Early stopping: 12 → 10 epochs (stop faster)"
echo ""
echo "💡 RATIONALE:"
echo "   MobileNet designed for mobile devices (224×224 optimal)"
echo "   320×320 is sweet spot: good detail + prevents overfitting"
echo "   384×384 was too much capacity for 3.5M parameter model"
echo ""

python3 ensemble_5class_trainer.py \
    --mode train \
    --dataset_path ./dataset_eyepacs_5class_balanced \
    --output_dir ./mobilenet_5class_v2_320_results \
    --experiment_name "5class_mobilenet_v2_320_optimal" \
    --base_models mobilenet_v2 \
    --num_classes 5 \
    --img_size 320 \
    --batch_size 20 \
    --epochs 100 \
    --learning_rate 3e-4 \
    --weight_decay 5e-4 \
    --ovo_dropout 0.55 \
    --freeze_weights false \
    --enable_clahe \
    --clahe_clip_limit 2.5 \
    --enable_medical_augmentation \
    --rotation_range 25.0 \
    --brightness_range 0.20 \
    --contrast_range 0.20 \
    --enable_focal_loss \
    --enable_class_weights \
    --class_weight_0 1.0 \
    --class_weight_1 1.0 \
    --class_weight_2 1.0 \
    --class_weight_3 1.0 \
    --class_weight_4 1.0 \
    --focal_loss_alpha 2.5 \
    --focal_loss_gamma 3.0 \
    --label_smoothing 0.15 \
    --scheduler cosine \
    --warmup_epochs 10 \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --patience 22 \
    --early_stopping_patience 10 \
    --target_accuracy 0.91 \
    --max_grad_norm 0.5 \
    --seed 42 \
    --device cuda \
    --no_wandb

echo ""
echo "✅ MobileNetV2 (320×320) training completed!"
echo ""
echo "📊 EXPECTED IMPROVEMENTS:"
echo "  Resolution:  384×384 → 320×320 (40% fewer pixels)"
echo "  Train-Val Gap: 8% → <3% (healthy training)"
echo "  Validation: ~89-91% (similar or better)"
echo "  Overfitting: Severe → Minimal ✅"
echo ""
echo "📈 SUCCESS CRITERIA:"
echo "  ✅ Train accuracy: 90-92% (not 97%+)"
echo "  ✅ Validation accuracy: 89-91%"
echo "  ✅ Train-Val gap: <3%"
echo "  ✅ Ensemble test: 86-89% (with voting loss)"
echo ""
echo "⚠️ IF STILL OVERFITTING (Gap >5%):"
echo "  → Stop MobileNet entirely"
echo "  → Use EfficientNetB2 v2 or SEResNext instead"
echo "  → MobileNet architecture may be insufficient for this task"
echo ""
echo "🚀 Training started at: $(date)"
echo "📁 Results directory: ./mobilenet_5class_v2_320_results/"
echo "📊 Monitor: tail -f ./mobilenet_5class_v2_320_results/logs/*.log"
echo ""
