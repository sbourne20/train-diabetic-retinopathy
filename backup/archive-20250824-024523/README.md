# Diabetic Retinopathy Classification with Medical Reasoning

A comprehensive AI system for diabetic retinopathy classification that combines state-of-the-art computer vision with medical reasoning capabilities. The system provides:

- **Multi-task Classification**: Simultaneous prediction of retinopathy grade (0-3) and macular edema risk (0-2)
- **Medical Reasoning**: AI-generated textual explanations with clinical terminology
- **Foundation Model**: Built on RETFound, a vision transformer pre-trained on 1.6M retinal images
- **Clinical Integration**: Structured medical reports with risk assessments and recommendations

## Features

### Core Capabilities
- ✅ **Retinopathy Grade Classification**: 4-class classification (No DR, Mild, Moderate, Severe NPDR)
- ✅ **Macular Edema Risk Assessment**: 3-class classification (No Risk, Low Risk, High Risk)
- ✅ **Medical Report Generation**: AI-generated clinical explanations with proper medical terminology
- ✅ **Risk Stratification**: Overall risk assessment with urgency levels
- ✅ **Clinical Recommendations**: Evidence-based treatment and follow-up recommendations

### Technical Features
- 🔬 **RETFound Backbone**: Pre-trained foundation model for retinal analysis
- 🔄 **Multi-task Learning**: Joint optimization of classification and reasoning tasks
- ⚖️ **Class Balancing**: Automatic handling of imbalanced datasets
- 📊 **Comprehensive Evaluation**: Detailed metrics, visualizations, and error analysis
- 🚀 **Production Ready**: Inference pipeline for clinical deployment

## Dataset Structure

```
dataset/
├── RG/                    # Retinopathy Grade
│   ├── 0/                # No DR (546 images)
│   ├── 1/                # Mild NPDR (153 images)
│   ├── 2/                # Moderate NPDR (247 images)
│   └── 3/                # Severe NPDR (254 images)
└── ME/                    # Macular Edema Risk
    ├── 0/                # No Risk (974 images)
    ├── 1/                # Low Risk (75 images)
    ├── 2/                # High Risk (151 images)
    └── 3/                # (Empty - handled automatically)
```

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Key Dependencies
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- timm >= 0.9.0
- Albumentations >= 1.3.0
- OpenCV >= 4.5.0
- scikit-learn >= 1.0.0

## Quick Start

### 1. Training
```bash
# Basic training
python main.py --mode train --epochs 100 --batch_size 16

# With custom paths
python main.py --mode train \
    --rg_path dataset/RG \
    --me_path dataset/ME \
    --pretrained_path models/RETFound_cfp_weights.pth \
    --epochs 100 \
    --experiment_name my_experiment
```

### 2. Evaluation
```bash
# Evaluate trained model
python main.py --mode evaluate --checkpoint_path checkpoints/best_model.pth
```

### 3. Inference
```bash
# Single image
python inference.py \
    --model_path checkpoints/best_model.pth \
    --image_path path/to/image.jpg \
    --output_dir results

# Batch processing
python inference.py \
    --model_path checkpoints/best_model.pth \
    --image_dir path/to/images/ \
    --output_dir results \
    --format json
```

## Model Architecture

### RETFound Backbone
- **Base Model**: Vision Transformer Large (ViT-L/16)
- **Pre-training**: 1.6M retinal images via self-supervised learning
- **Input Size**: 224×224 RGB images
- **Features**: 1024-dimensional embeddings

### Multi-task Head
- **Shared Features**: LayerNorm + Linear projection
- **Attention Mechanism**: Multi-head attention for feature aggregation
- **Task-specific Heads**: Separate classifiers for RG and ME
- **Language Features**: Projection for medical reasoning module

### Medical Reasoning Module
- **Architecture**: GPT-2 based language model
- **Integration**: Visual features projected to language model dimension
- **Output**: Structured medical reports with clinical terminology
- **Vocabulary**: Enhanced with medical terms and clinical expressions

## Training Configuration

### Key Parameters
```python
# Model Configuration
img_size: 224
patch_size: 16
embed_dim: 1024
depth: 24
num_heads: 16

# Training Configuration  
batch_size: 16
learning_rate: 1e-4
num_epochs: 100
warmup_epochs: 10

# Class Weights (automatically computed)
# Handles imbalanced dataset distribution
```

### Progressive Unfreezing
- **Initial**: Backbone frozen for 20 epochs
- **Progressive**: Unfreeze layers every 5 epochs
- **Strategy**: Top-down unfreezing for stable training

## Performance Metrics

### Expected Performance (based on RETFound benchmarks)
- **RG Classification**: 85-92% accuracy
- **ME Classification**: 80-90% accuracy  
- **Combined AUC**: 85-94% (varies by dataset)
- **Clinical Sensitivity**: >90% for severe cases

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-score
- ROC-AUC (macro and weighted)
- Cohen's Kappa
- Confusion matrices
- Calibration curves
- Clinical relevance assessment

## Output Examples

### JSON Output
```json
{
  "image_path": "sample_image.jpg",
  "retinopathy_grade": {
    "grade": 2,
    "name": "Moderate NPDR",
    "confidence": 0.89,
    "risk_level": "Moderate",
    "probabilities": {
      "No DR": 0.02,
      "Mild NPDR": 0.09, 
      "Moderate NPDR": 0.89,
      "Severe NPDR": 0.00
    }
  },
  "macular_edema": {
    "grade": 1,
    "name": "Low Risk",
    "confidence": 0.76,
    "risk_level": "Moderate"
  },
  "medical_report": "Analysis reveals moderate non-proliferative diabetic retinopathy with dot-blot hemorrhages and hard exudates visible throughout the posterior pole...",
  "overall_risk_assessment": {
    "risk_level": "High",
    "urgency": "Moderate - Ophthalmological follow-up within 3-6 months"
  },
  "recommendations": [
    "Refer to ophthalmologist within 3-6 months",
    "Consider laser photocoagulation if progressing",
    "Intensive diabetes management"
  ]
}
```

### Text Report
```
DIABETIC RETINOPATHY ANALYSIS REPORT
====================================

FINDINGS:
---------
Retinopathy Grade: 2 - Moderate NPDR
Confidence: 0.890
Risk Level: Moderate

Macular Edema Risk: 1 - Low Risk  
Confidence: 0.760
Risk Level: Moderate

MEDICAL REASONING:
-----------------
Analysis reveals moderate non-proliferative diabetic retinopathy with multiple dot-blot hemorrhages scattered throughout the posterior pole. Hard exudates are present temporal to the macula. Cotton wool spots are observed in the superior retinal quadrants indicating focal retinal ischemia...

RECOMMENDATIONS:
---------------
• Refer to ophthalmologist within 3-6 months
• Consider laser photocoagulation if progressing  
• Intensive diabetes management
• Maintain HbA1c < 7% (53 mmol/mol)
```

## File Structure

```
train-diabetic-retinopathy/
├── config.py              # Configuration management
├── dataset.py              # Data loading and preprocessing  
├── models.py               # Model architectures
├── trainer.py              # Training pipeline
├── evaluator.py            # Evaluation and metrics
├── inference.py            # Inference pipeline
├── main.py                 # Main training script
├── utils.py                # Utility functions
├── requirements.txt        # Dependencies
├── data/
│   └── medical_terms.json  # Medical vocabulary
├── outputs/                # Training outputs
├── checkpoints/            # Model checkpoints
├── logs/                   # Training logs
└── visualizations/         # Evaluation plots
```

## Advanced Usage

### Custom Configuration
```python
from config import get_config

config = get_config()
config.model.img_size = 384  # Higher resolution
config.training.num_epochs = 200
config.training.learning_rate = 5e-5
```

### Data Augmentation
The system uses extensive augmentation:
- Rotation, flipping, brightness/contrast adjustment
- Gaussian noise, blur, CLAHE enhancement
- Maintains medical image integrity

### Multi-GPU Training
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --mode train --batch_size 64
```

## Clinical Integration

### Risk Assessment Framework
- **Low Risk**: Routine annual screening
- **Moderate Risk**: 3-6 month follow-up
- **High Risk**: Urgent ophthalmological referral

### Quality Assurance
- Prediction confidence scoring
- Calibration assessment
- Error analysis and edge case handling
- Clinical validation metrics

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this code in your research, please cite:

```bibtex
@article{retfound2023,
  title={A foundation model for generalizable disease detection from retinal images},
  author={Zhou, Yukun and others},
  journal={Nature},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

⚠️ **Medical Disclaimer**: This AI system is designed for screening and research purposes only. It should not be used as a substitute for professional medical diagnosis, treatment, or care. Always consult qualified healthcare professionals for medical decisions.

## Support

For questions and support:
- 📧 Email: [your-email@domain.com]
- 🐛 Issues: GitHub Issues
- 📚 Documentation: [Project Wiki]

---

**Developed with ❤️ for advancing diabetic retinopathy screening and improving patient outcomes worldwide.**