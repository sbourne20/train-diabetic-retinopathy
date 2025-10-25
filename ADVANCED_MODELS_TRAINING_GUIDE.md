# Advanced Models Training Guide
## CoAtNet, ConvNeXt, and Swin Transformer V2 for Medical-Grade DR Classification

**Date**: 2025-10-26
**Objective**: Train 3 state-of-the-art architectures to boost ensemble accuracy from 98% → 99%

---

## 🎯 **Overview**

### **Current Ensemble (4 models)**
| Model | Accuracy | Architecture Type |
|-------|----------|-------------------|
| DenseNet121 | 98.70% | Dense CNN |
| EfficientNetB2 | 98.51% | Scaled CNN |
| ResNet50 | 97.96% | Residual CNN |
| SEResNeXt50 | 95.43% | SE + Grouped CNN |

**Limitation**: All are traditional CNNs - low architectural diversity

### **New Models (3 models)**
| Model | Expected Accuracy | Architecture Type | Priority |
|-------|-------------------|-------------------|----------|
| **CoAtNet-0** | 98-99% | **Hybrid CNN + Transformer** | VERY HIGH |
| **ConvNeXt-Tiny** | 98-99% | **Modern CNN (Transformer-like)** | HIGH |
| **Swin Transformer V2** | 98-99% | **Pure Transformer (Hierarchical)** | HIGH |

**Benefit**: High architectural diversity → Better ensemble performance

---

## 📚 **Model Details**

### **1. CoAtNet-0** (Hybrid CNN + Transformer)

**Architecture**:
- Combines convolutional layers (early stages) + transformer blocks (later stages)
- Best of both worlds: Local patterns (CNN) + Global context (Transformer)

**Why for Medical Imaging**:
- ✅ Convolution provides inductive bias for local lesion detection
- ✅ Attention captures long-range dependencies (vessel patterns across retina)
- ✅ Top tier performance in medical imaging (2023)

**Parameters**: 25M
**Input Size**: 224×224
**Training Time**: ~1-2 days on V100

**Key Features**:
- Hybrid architecture reduces overfitting
- Attention maps provide explainability for clinicians
- SOTA on ImageNet and medical datasets

**Training Command**:
```bash
./train_coatnet.sh
```

---

### **2. ConvNeXt-Tiny** (Modern CNN)

**Architecture**:
- Pure CNN but with Transformer-inspired design choices
- Modernized ResNet with improved components
- Larger kernels, depthwise convolutions, layer normalization

**Why for Medical Imaging**:
- ✅ Outperforms Swin Transformers in many benchmarks
- ✅ Better than traditional CNNs (ResNet, DenseNet)
- ✅ Proven in medical imaging (2023-2024)

**Parameters**: 28M
**Input Size**: 224×224
**Training Time**: ~1-2 days on V100

**Key Features**:
- Pure CNN (no attention) but modern design
- Better generalization than older CNNs
- Faster training than transformers

**Training Command**:
```bash
./train_convnext.sh
```

---

### **3. Swin Transformer V2-Tiny** (Hierarchical Vision Transformer)

**Architecture**:
- Pure transformer with hierarchical structure
- Shifted window attention for efficiency
- Multi-scale feature representation

**Why for Medical Imaging**:
- ✅ Better than ViT for dense prediction tasks
- ✅ Multi-scale features capture lesions at different sizes
- ✅ Top performer in medical challenges

**Parameters**: 28M
**Input Size**: 256×256 (⚠️ Note: Larger than others)
**Training Time**: ~1-2 days on V100

**Key Features**:
- Pure attention mechanism (complementary to CNNs)
- Hierarchical structure provides multi-scale understanding
- Window-based attention reduces computational cost

**Training Command**:
```bash
./train_swinv2.sh
```

---

## 🚀 **Training Instructions**

### **Step 1: Prepare Environment**

Ensure you have the required libraries:
```bash
pip install timm  # For advanced architectures
pip install python-dotenv
```

Verify dataset exists:
```bash
ls ./dataset_eyepacs_5class_balanced_enhanced_v2/
# Should show: train/ val/ test/ subdirectories
```

### **Step 2: Train Models Sequentially**

**Option A: Train all 3 models in sequence** (recommended for unattended training)
```bash
# Train CoAtNet-0 first (highest priority)
./train_coatnet.sh

# Then train ConvNeXt-Tiny
./train_convnext.sh

# Finally train Swin Transformer V2
./train_swinv2.sh
```

**Option B: Train individual models** (recommended for monitoring)
```bash
# Train one model at a time, monitor results, then proceed
./train_coatnet.sh
# Wait for completion, check results
python model_analyzer.py --model ./coatnet_5class_results/models/*.pth

# If satisfied, train next
./train_convnext.sh
```

### **Step 3: Monitor Training**

Check training logs:
```bash
tail -f coatnet_training_log.txt
tail -f convnext_training_log.txt
tail -f swinv2_training_log.txt
```

Monitor GPU usage:
```bash
nvidia-smi -l 1  # Update every second
```

### **Step 4: Evaluate Individual Models**

After each training completes:

```bash
# Analyze binary classifiers
python model_analyzer.py --model ./coatnet_5class_results/models

# Evaluate full ensemble
python ensemble_5class_trainer.py --mode evaluate \
  --base_models coatnet_0_rw_224 \
  --dataset_path ./dataset_eyepacs_5class_balanced_enhanced_v2 \
  --output_dir ./coatnet_5class_results
```

### **Step 5: Create 7-Model Super-Ensemble**

After all 3 models are trained, create the super-ensemble evaluation script.

---

## 📊 **Expected Results**

### **Individual Model Performance**
| Model | Expected Accuracy | Expected Recall (Class 1) |
|-------|-------------------|---------------------------|
| CoAtNet-0 | 98-99% | >97% |
| ConvNeXt-Tiny | 98-99% | >97% |
| Swin Transformer V2 | 98-99% | >96% |

### **7-Model Ensemble Performance**
| Metric | 4-Model Current | 7-Model Expected | Improvement |
|--------|----------------|------------------|-------------|
| **Accuracy** | 97.65% | **98.5-99.0%** | +1.0-1.5% |
| **Class 1 Recall** | ~96% | **>98%** | +2%+ |
| **AUC** | 0.9977 | **>0.998** | +0.001 |
| **Medical Grade** | A | **A+** | ✅ |

---

## ⚙️ **Training Configuration**

All three models use the same optimized hyperparameters:

```python
{
    "epochs": 100,
    "batch_size": 2,  # V100 16GB optimized
    "learning_rate": 5e-5,
    "weight_decay": 0.00025,
    "gradient_accumulation_steps": 4,  # Effective batch size = 8
    "scheduler": "cosine",
    "warmup_epochs": 10,
    "patience": 28,  # Early stopping

    # Loss function
    "enable_focal_loss": true,
    "focal_loss_gamma": 3.0,
    "focal_loss_alpha": 2.5,
    "enable_class_weights": true,
    "label_smoothing": 0.1,

    # Data augmentation
    "enable_clahe": true,
    "rotation_range": 25.0,
    "brightness_range": 0.2,
    "contrast_range": 0.2,

    # OVO ensemble
    "ovo_dropout": 0.28,
    "num_classes": 5
}
```

### **Model-Specific Settings**

**CoAtNet-0**:
- Input size: 224×224
- Partial fine-tuning: Freeze stages 0-1, train stages 2-3 (transformer blocks)

**ConvNeXt-Tiny**:
- Input size: 224×224
- Partial fine-tuning: Freeze stages 0-1, train stages 2-3

**Swin Transformer V2**:
- Input size: **256×256** (⚠️ different from others)
- Partial fine-tuning: Freeze layers 0-1, train layers 2-3

---

## 🔍 **Troubleshooting**

### **Out of Memory (OOM) Error**

If you get CUDA OOM error:

1. **Reduce batch size**:
   ```bash
   # Edit training script, change:
   --batch_size 2 → --batch_size 1
   --gradient_accumulation_steps 4 → --gradient_accumulation_steps 8
   ```

2. **Enable more gradient checkpointing**:
   - Already enabled in code for all models
   - Trades computation for memory (40% saving)

3. **Use mixed precision** (if supported):
   ```bash
   # Add to training command:
   --use_amp
   ```

### **Model Not Found Error**

If `timm` can't find the model:

```bash
# Check available models
python -c "import timm; print(timm.list_models('coatnet*'))"
python -c "import timm; print(timm.list_models('convnext*'))"
python -c "import timm; print(timm.list_models('swinv2*'))"

# Update timm to latest version
pip install --upgrade timm
```

### **Slow Training**

If training is too slow:

1. **Check DataLoader workers**:
   - Default is 4 workers
   - Increase if you have more CPU cores

2. **Enable CLAHE caching**:
   - CLAHE preprocessing can be slow
   - Consider pre-processing dataset once

3. **Use smaller image size** (last resort):
   - ConvNeXt/CoAtNet: Can use 192×192
   - SwinV2: Can use 224×224
   - May reduce accuracy slightly

---

## 📈 **Post-Training Analysis**

### **Compare Models**

```bash
python compare_models.py \
  --models densenet121 efficientnetb2 resnet50 seresnext50 coatnet convnext swinv2 \
  --results_dirs ./v2.5-model-dr/*_5class_results
```

### **Ensemble Diversity Analysis**

Check architectural diversity:
```bash
python analyze_ensemble_diversity.py \
  --models densenet121 efficientnetb2 resnet50 coatnet convnext swinv2
```

### **Per-Class Performance**

Identify which models excel at which classes:
```bash
python analyze_per_class_strengths.py \
  --results_dirs ./v2.5-model-dr/*_5class_results
```

---

## 🏥 **Medical Validation**

After training all 3 models:

1. **Individual model validation**: Each must achieve >97% accuracy
2. **Binary classifier validation**: All 10 classifiers per model >90% accuracy
3. **Ensemble validation**: 7-model ensemble must achieve >98.5% accuracy
4. **Class 1 recall validation**: Must exceed 95% (fixes SEResNeXt50 weakness)
5. **Clinical validation**: Test on independent hold-out set with expert ground truth

---

## 🎯 **Success Criteria**

✅ **Individual Models**:
- CoAtNet-0: >98% accuracy
- ConvNeXt-Tiny: >98% accuracy
- Swin Transformer V2: >97% accuracy

✅ **7-Model Ensemble**:
- Overall accuracy: >98.5%
- All classes recall: >95%
- AUC: >0.998
- Medical grade: A+ (full FDA/CE compliance)

✅ **Diversity Metrics**:
- Architecture types: 4 (CNN, Modern CNN, Hybrid, Transformer)
- Parameter range: 25M-60M (good diversity)
- Training paradigms: Convolutional + Attention-based

---

## 📚 **References**

1. **CoAtNet**: "CoAtNet: Marrying Convolution and Attention for All Data Sizes" (NeurIPS 2021)
2. **ConvNeXt**: "A ConvNet for the 2020s" (CVPR 2022)
3. **Swin Transformer V2**: "Swin Transformer V2: Scaling Up Capacity and Resolution" (CVPR 2022)
4. **Medical Imaging Applications**: Various DR detection papers using these architectures (2023-2024)

---

## 🚀 **Quick Start**

```bash
# 1. Verify environment
python -c "import timm; print('timm version:', timm.__version__)"

# 2. Train highest priority model first
./train_coatnet.sh

# 3. Monitor progress
tail -f coatnet_training_log.txt

# 4. Evaluate when complete
python model_analyzer.py --model ./coatnet_5class_results/models

# 5. Repeat for other models
./train_convnext.sh
./train_swinv2.sh
```

---

**Status**: ✅ Ready for training
**Estimated Total Time**: 3-6 days (sequential training on V100)
**Expected Outcome**: 99% accuracy medical-grade ensemble

**Next Steps**: Start with CoAtNet-0 (highest priority) →
