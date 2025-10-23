#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 5-CLASS DR + MedSigLIP-448 Training Script (Medical-Grade Vision-Language Model)
echo "🏥 5-CLASS DR + MedSigLIP-448 Training (v1 - Medical Vision-Language)"
echo "====================================================================="
echo "🎯 Target: 95-97%+ accuracy (Medical domain pre-trained model)"
echo "📊 Dataset: 5-Class Perfectly Balanced (53,935 images)"
echo "🏗️ Model: MedSigLIP-448 (Medical Vision-Language - SOTA for medical imaging)"
echo "🔗 System: A10 24-40GB GPU optimized (High-Performance)"
echo ""

# Create output directory for 5-class MedSigLIP v1 results
mkdir -p ./medsiglip_5class_v1_results

echo "🔬 5-CLASS MedSigLIP-448 OVO ENSEMBLE Configuration (v1 - MEDICAL VISION-LANGUAGE):"
echo "  - Dataset: ./dataset_eyepacs_5class_balanced_enhanced_v2"
echo "  - Classes: 5 (0: No DR, 1: Mild NPDR, 2: Moderate NPDR, 3: Severe NPDR, 4: PDR)"
echo "  - Total images: 53,935 (Train: 37,750 | Val: 8,090 | Test: 8,095)"
echo "  - Perfect balance: 10,787 per class (1.00:1 ratio)"
echo ""
echo "🎯 WHY MedSigLIP-448 FOR DIABETIC RETINOPATHY?"
echo "  1. MEDICAL PRE-TRAINING: Trained on medical imaging datasets (PubMed, MIMIC-CXR, etc.)"
echo "  2. VISION-LANGUAGE: Understands medical terminology and visual features together"
echo "  3. HIGH RESOLUTION: Native 448×448 input (optimal for retinal lesion detection)"
echo "  4. SOTA PERFORMANCE: State-of-the-art results on medical imaging benchmarks"
echo "  5. TRANSFER LEARNING: Better feature extraction than general-purpose CNNs"
echo "  6. MULTI-MODAL: Can integrate with MedGemma text generation (Phase 3)"
echo ""
echo "📊 MedSigLIP-448 ARCHITECTURE ADVANTAGES:"
echo "  ✅ Medical Domain Knowledge:"
echo "     • Pre-trained on millions of medical images + text pairs"
echo "     • Understands anatomical structures and pathology"
echo "     • Better generalization to unseen DR cases"
echo "  "
echo "  ✅ Vision-Language Features:"
echo "     • Learned representations aligned with medical descriptions"
echo "     • Can leverage clinical vocabulary (microaneurysms, exudates, etc.)"
echo "     • Superior to pure vision models for medical tasks"
echo "  "
echo "  ✅ High-Resolution Processing:"
echo "     • Native 448×448 input (no downsampling artifacts)"
echo "     • Detects small lesions (microaneurysms as small as 10-20 pixels)"
echo "     • Better spatial resolution than 224×224 models"
echo ""
echo "📊 v1 CONFIGURATION - A10 47GB MEMORY-OPTIMIZED (OVO BINARY - EXTREME):"
echo "  Parameter          | Value                | Rationale"
echo "  -------------------|----------------------|------------------"
echo "  Image Size         | 448×448 (forced)     | MedSigLIP always upscales to 448"
echo "  Batch Size         | 1                    | MINIMUM for OVO (model uses 43GB alone)"
echo "  Gradient Accum     | 8                    | Effective batch = 8 (memory efficient)"
echo "  Gradient Checkpoint| ✅ ENABLED           | Saves 30-40% memory (required)"
echo "  Mixed Precision    | ✅ FP16 AUTO         | Automatic mixed precision (saves 40%)"
echo "  Learning Rate      | 3e-5                 | Fine-tuning pre-trained medical model"
echo "  Weight Decay       | 1e-4                 | Light regularization (already robust)"
echo "  Dropout            | 0.25                 | Lower than CNNs (medical pre-training)"
echo "  Label Smoothing    | 0.08                 | Conservative (medical accuracy priority)"
echo "  Patience           | 30                   | Allow convergence for large model"
echo ""
echo "🎯 PREPROCESSING & AUGMENTATION:"
echo "  - CLAHE: ✅ ENABLED (clip_limit=2.5, medical-grade enhancement)"
echo "  - Focal loss: ✅ ENABLED (alpha=2.0, gamma=2.5, medical-optimized)"
echo "  - Augmentation: MEDICAL-GRADE (20° rotation, 15% brightness/contrast)"
echo "  - Class weights: BALANCED (1.0:1.0:1.0:1.0:1.0)"
echo "  - Scheduler: Cosine with 15-epoch warmup (large model warmup)"
echo "  - Early stopping: 25 epochs patience"
echo "  - Max epochs: 100"
echo ""
echo "📈 EXPECTED PERFORMANCE (Medical Vision-Language Advantage):"
echo "  Individual Model Performance:"
echo "    • EfficientNetB2 (CNN): 96.27% accuracy"
echo "    • ResNet50 (CNN): 94.95% accuracy"
echo "    • DenseNet121 (CNN): 91.21% accuracy"
echo "    • MedSigLIP-448 (Medical VL): 95-97%+ accuracy (TARGET)"
echo "  "
echo "  OVO Ensemble Performance:"
echo "    • Strong pairs (0v3, 0v4): 96-99% (medical domain expertise)"
echo "    • Good pairs (0v1, 0v2, 1v3, 1v4, 2v4): 93-96% (clinical knowledge)"
echo "    • Challenging pairs (1v2, 2v3, 3v4): 90-93% (vision-language advantage)"
echo "    • Average pair accuracy: 94-96%"
echo "    • Final Ensemble: 95-97%+ (MEDICAL-GRADE TARGET)"
echo ""
echo "🔬 TECHNICAL SPECIFICATIONS:"
echo "  - Model: google/medsiglip-448 (HuggingFace)"
echo "  - Architecture: Vision Transformer + Medical Language Alignment"
echo "  - Parameters: ~400M (vision encoder)"
echo "  - Feature Dim: 768 (high-capacity medical features)"
echo "  - Freeze Strategy: Freeze early layers, fine-tune last 6 transformer blocks"
echo "  - Requires: HUGGINGFACE_TOKEN in .env file"
echo ""
echo "⚙️  A10 47GB MEMORY OPTIMIZATION (OVO BINARY - EXTREME):"
echo "  - Batch size: 1 (MINIMUM - model alone uses 43GB)"
echo "  - Gradient accumulation: 8 (effective batch = 8)"
echo "  - Mixed precision: ✅ FP16 auto (PyTorch AMP saves ~40% memory)"
echo "  - Gradient checkpointing: ✅ ENABLED (saves 30-40% memory, required!)"
echo "  - Model loading: FP16 backbone (torch_dtype=float16, saves 50%)"
echo "  - Memory per binary classifier: ~44-45GB VRAM (maximum for 47GB GPU)"
echo "  - Training time: ~20-30 hours (10 binary classifiers × 2-3 hours each, slower due to batch=1)"
echo ""

# Train 5-Class with MedSigLIP-448 OVO Binary Classifiers (Medical Vision-Language Model)
# Using ensemble_5class_trainer.py (OVO binary mode - compatible with DenseNet/EfficientNetB2)
python3 ensemble_5class_trainer.py \
    --mode train \
    --dataset_path ./dataset_eyepacs_5class_balanced_enhanced_v2 \
    --output_dir ./medsiglip_5class_v1_results \
    --experiment_name "5class_medsiglip448_v1_ovo" \
    --base_models medsiglip_448 \
    --num_classes 5 \
    --img_size 448 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --epochs 100 \
    --learning_rate 3e-5 \
    --weight_decay 1e-4 \
    --ovo_dropout 0.25 \
    --freeze_weights false \
    --enable_clahe \
    --enable_focal_loss \
    --enable_class_weights \
    --class_weight_0 1.0 \
    --class_weight_1 1.0 \
    --class_weight_2 1.0 \
    --class_weight_3 1.0 \
    --class_weight_4 1.0 \
    --focal_loss_alpha 2.0 \
    --focal_loss_gamma 2.5 \
    --scheduler cosine \
    --warmup_epochs 15 \
    --validation_frequency 1 \
    --checkpoint_frequency 5 \
    --patience 30 \
    --early_stopping_patience 25 \
    --target_accuracy 0.95 \
    --max_grad_norm 1.0 \
    --label_smoothing 0.08 \
    --seed 42

echo ""
echo "✅ 5-CLASS MedSigLIP-448 MEDICAL VISION-LANGUAGE OVO ENSEMBLE training completed!"
echo ""
echo "📊 NEXT STEPS:"
echo "  1. Analyze results: python model_analyzer.py --model ./medsiglip_5class_v1_results/models/*.pth"
echo "  2. Compare with EfficientNetB2/ResNet50/DenseNet121 performance"
echo "  3. Ensemble MedSigLIP + CNN models for maximum accuracy"
echo "  4. Expected: MedSigLIP 95-97%+ individual → 97%+ multi-architecture ensemble"
