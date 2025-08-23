# Vertex AI Training Manager - Enhanced for Medical-Grade Diabetic Retinopathy Classification

## Overview

This enhanced Vertex AI trainer supports both original and new dataset structures for achieving 90-95% medical-grade accuracy in diabetic retinopathy classification.

## Key Features

- **Dual Dataset Support**: Handles both original (RG/ME) and new 5-class DR classification structures
- **Medical-Grade Optimization**: Enhanced parameters for achieving 90-95% accuracy
- **Mandatory Parameter Validation**: No fallbacks - ensures explicit dataset configuration
- **Advanced Training Techniques**: Focal loss, class weighting, medical-grade metrics

## Dataset Types

### Type 0: Original Dataset Structure
```
dataset/
├── RG/
│   ├── 0/  (No retinal grading issues)
│   └── 1/  (Retinal grading issues present)
└── ME/
    ├── 0/  (No macular edema)
    └── 1/  (Macular edema present)
```

### Type 1: Enhanced 5-Class DR Classification
```
dataset3_augmented_resized/
├── train/
│   ├── 0/  (No DR)
│   ├── 1/  (Mild NPDR)
│   ├── 2/  (Moderate NPDR)
│   ├── 3/  (Severe NPDR)
│   └── 4/  (PDR)
├── val/
│   └── [same structure]
└── test/
    └── [same structure]
```

## Required Parameters

- `--dataset`: Dataset folder name within GCS bucket (MANDATORY)
- `--dataset-type`: 0 or 1 (MANDATORY)
- `--action`: upload/train/tune/monitor/download

## Usage Examples

### 1. Upload Dataset Type 1 (Enhanced 5-Class)
```bash
python vertex_ai_trainer.py \
  --action upload \
  --dataset data \
  --dataset-type 1 \
  --dataset_path ./dataset3_augmented_resized \
  --bucket_name dr-data-2 \
  --project_id your-project-id
```

### 2. Train Medical-Grade Model (Type 1)
```bash
python vertex_ai_trainer.py \
  --action train \
  --dataset data \
  --dataset-type 1 \
  --bucket_name dr-data-2 \
  --project_id your-project-id
```

### 3. Upload Original Dataset (Type 0)
```bash
python vertex_ai_trainer.py \
  --action upload \
  --dataset legacy_data \
  --dataset-type 0 \
  --dataset_path ./dataset \
  --bucket_name dr-data-2 \
  --project_id your-project-id
```

### 4. Train with Original Dataset (Type 0)
```bash
 python vertex_ai_trainer.py --action train --dataset dataset3_augmented_resized --dataset-type 1 --bucket_name dr-data-2 --project_id curalis-20250522
 --region us-east1
```

### 5. Hyperparameter Tuning
```bash
python vertex_ai_trainer.py \
  --action tune \
  --dataset data \
  --dataset-type 1 \
  --bucket_name dr-data-2 \
  --project_id your-project-id
```

## Medical Terms Configuration

- **Type 0**: Uses `medical_terms.json` (RG + ME classification)
- **Type 1**: Uses `medical_terms_type1.json` (5-class DR classification)

Ensure both files are uploaded to your GCS bucket:
- `gs://dr-data-2/medical_terms.json`
- `gs://dr-data-2/medical_terms_type1.json`

## Medical-Grade Training Enhancements

### For Type 1 (5-Class DR):
- **200 epochs** for comprehensive learning
- **Batch size 16** for stable gradients
- **Learning rate 3e-5** for fine-grained convergence
- **Focal loss** for minority class performance
- **Class weighting** for imbalanced data handling
- **Medical-grade validation** metrics

### Expected Performance:
- **Overall Accuracy**: 90-95%
- **Per-class Sensitivity**: ≥85%
- **Per-class Specificity**: ≥90%
- **AUC per class**: ≥85%

## GCS Bucket Structure

```
gs://dr-data-2/
├── data/                          # Dataset folder (--dataset parameter)
│   ├── train/0/ ... train/4/     # For type 1
│   ├── val/0/ ... val/4/         # For type 1
│   └── test/0/ ... test/4/       # For type 1
├── medical_terms.json            # For type 0
├── medical_terms_type1.json      # For type 1
├── staging/                      # Training packages
└── outputs/                      # Training results
```

## Monitoring and Results

```bash
# Monitor training job
python vertex_ai_trainer.py \
  --action monitor \
  --job_id projects/.../jobs/... \
  --dataset data \
  --dataset-type 1

# Download results
python vertex_ai_trainer.py \
  --action download \
  --dataset data \
  --dataset-type 1 \
  --bucket_name dr-data-2
```

## Error Handling

The system validates:
- Dataset structure matches the specified type
- All required directories exist
- Mandatory parameters are provided
- Medical terms files are available

## Production Deployment

For medical applications, ensure:
1. Model achieves ≥90% accuracy on validation set
2. All classes meet minimum sensitivity/specificity thresholds
3. Confidence scores are properly calibrated
4. Regular model performance monitoring in production