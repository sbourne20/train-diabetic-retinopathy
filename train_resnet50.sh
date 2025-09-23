#!/bin/bash

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Messidor + InceptionV3 Medical-Grade Training Script
echo "🏥 Messidor + InceptionV3 Medical-Grade Training"
echo "==============================================="
echo "🎯 Target: 85%+ accuracy with InceptionV3 architecture"
echo "📊 Dataset: Messidor (5-class DR classification)"
echo "🏗️ Model: InceptionV3 (deep inception architecture)"
echo "🔬 Medical-grade architecture with optimized hyperparameters"
echo ""

# Create output directory for Messidor results
mkdir -p ./messidor_results

echo "🔬 Messidor InceptionV3 Medical-Grade Configuration:"
echo "  - Dataset: Messidor (./dataset_messidor)"
echo "  - Model: InceptionV3 (deep inception capacity)"
echo "  - Image size: 224x224 (research paper standard)"
echo "  - Batch size: 16 (optimized for InceptionV3)"
echo "  - Learning rate: 1e-4 (InceptionV3 optimized)"
echo "  - Weight decay: 1e-3 (strong regularization)"
echo "  - Dropout: 0.4 (balanced regularization)"
echo "  - Epochs: 80 (sufficient for convergence)"
echo "  - Enhanced augmentation + progressive training"
echo "  - Target: 85%+ accuracy (ensemble-ready performance)"
echo ""

# Train Messidor with research-validated hyperparameters
python ensemble_local_trainer.py \
    --mode train \
    --dataset_path ./dataset_aptos \
    --output_dir ./messidor_results \
    --experiment_name "messidor_inception_medical" \
    --base_models inception_v3 \
    --img_size 224 \
    --batch_size 16 \
    --epochs 80 \
    --learning_rate 1e-4 \
    --weight_decay 1e-3 \
    --ovo_dropout 0.4 \
    --enable_medical_augmentation \
    --rotation_range 20.0 \
    --brightness_range 0.15 \
    --contrast_range 0.15 \
    --enable_focal_loss \
    --enable_class_weights \
    --scheduler cosine \
    --warmup_epochs 10 \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --patience 15 \
    --early_stopping_patience 12 \
    --target_accuracy 0.85 \
    --seed 42

echo ""
echo "✅ Messidor InceptionV3 training completed!"
echo "📁 Results saved to: ./messidor_results"
echo ""
echo "🎯 Medical-Grade Configuration Applied:"
echo "  🏗️ Architecture: InceptionV3 (deep inception learning)"
echo "  📊 Model capacity: 27M parameters (medical-grade capacity)"
echo "  🎓 Optimized learning rate: 1e-4 (InceptionV3 optimized)"
echo "  💧 Balanced dropout: 0.4 (prevents overfitting)"
echo "  ⏰ Training epochs: 80 (sufficient convergence)"
echo "  🔀 Enhanced augmentation: Medical imaging optimized settings"
echo ""
echo "📊 Expected Performance:"
echo "  🎯 Target: 85%+ validation accuracy (ensemble-ready performance)"
echo "  🏥 Medical grade: Contribute to ensemble 90%+ target"
echo "  📈 Generalization: Better performance with deeper architecture"
echo "  🔬 Messidor dataset: Test performance on different dataset distribution"
echo ""
echo "📋 Next Steps:"
echo "  1. Analyze results: python model_analyzer.py --model ./messidor_results/models/best_inception_v3.pth"
echo "  2. If performance good: Keep for 3-model ensemble"
echo "  3. If performance poor: Retrain on APTOS dataset"
echo "  4. Build ensemble: DenseNet121 + MobileNet + InceptionV3"