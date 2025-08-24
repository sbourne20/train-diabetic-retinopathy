# Diabetic Retinopathy Training Project - Claude Code Rules

## Project Overview
This project trains a medical-grade diabetic retinopathy classification model using RETFound foundation model on Google Cloud Vertex AI.

## Core Requirements

### Model Architecture
- **MUST** use official RETFound model from HuggingFace: `YukunZhou/RETFound_mae_natureCFP`
- **NO** fallback models allowed (timm, pretrained weights, etc.)
- **FAIL IMMEDIATELY** with clear error if RETFound model cannot be loaded
- Require `HUGGINGFACE_TOKEN` in `.env` file - no exceptions

### Parameter Validation
- Bucket names **MUST** come from `--bucket_name` parameter only
- Dataset names **MUST** come from `--dataset` parameter only
- **NEVER** hardcode `dr-data-2` or `dataset3_augmented_resized` in code
- All GCS paths must be constructed dynamically from parameters
- Validate all required parameters before starting training

### Code Organization
- Keep **ONLY** essential training files in root directory:
  - `vertex_ai_trainer.py`, `vertex_ai_config.py`, `models.py`
  - `config.py`, `dataset.py`, `trainer.py`, `evaluator.py`
  - `utils.py`, `main.py`, `requirements.txt`
  - `data/medical_terms_type1.json`
- Move documentation, old datasets, and utilities to `/backup/` directory
- Maintain clean separation between training and inference code

### Vertex AI Configuration
- Use `us-central1` region for V100 GPU quota availability
- Handle quota exceeded errors gracefully with clear instructions
- Support both GPU types (V100, T4) based on available quota
- Validate GCS bucket accessibility before training
- Use proper error handling for authentication issues

### Environment Variables
Required in `.env` file:
- `HUGGINGFACE_TOKEN` - HuggingFace API token for model access (**SENSITIVE**)
- `GOOGLE_CLOUD_PROJECT` - GCP project ID
- `GOOGLE_CLOUD_REGION` - Training region  
- `GCS_BUCKET` - Default bucket (overridden by --bucket_name)

**Additional sensitive variables that MUST be in `.env`:**
- `OPENAI_API_KEY` - If using OpenAI services (**SENSITIVE**)
- `WANDB_API_KEY` - For experiment tracking (**SENSITIVE**)
- `DATABASE_PASSWORD` - Database authentication (**SENSITIVE**)
- `SERVICE_ACCOUNT_KEY` - GCP service account credentials (**SENSITIVE**)
- `PRIVATE_REGISTRY_TOKEN` - For private model repositories (**SENSITIVE**)

**Rule**: Any configuration value that contains passwords, tokens, keys, or private information MUST be stored in `.env` and loaded via `os.getenv()`.

### Dataset Structure Support
- **Type 0**: Original dataset structure (RG/ME with 0,1 classes)
- **Type 1**: 5-class DR structure (train/val/test with 0-4 classes)
- Validate dataset structure before training
- Support both local upload and existing GCS datasets

### Error Handling Standards
- Provide **clear, actionable** error messages
- Include troubleshooting steps for common issues:
  - Missing HuggingFace token
  - Quota exceeded scenarios
  - Invalid dataset structures
  - GCS permission problems
- Never fail silently or use random fallbacks

### Security Requirements
- **NEVER** commit tokens or credentials to repository
- **ALL** sensitive or secret configuration **MUST** be stored in `.env` file:
  - API tokens (HuggingFace, OpenAI, etc.)
  - Database passwords
  - Service account keys
  - Private URLs or endpoints
  - Any authentication credentials
- Use environment variables for all sensitive configuration
- Add sensitive config keys to `.gitignore` to prevent accidental commits
- Validate all user inputs before processing
- Ensure proper GCS permissions before operations

### Medical AI Compliance
- Use medical-grade validation metrics
- Support comprehensive clinical feature extraction
- Include confidence estimation for all predictions
- Generate structured clinical reports
- Maintain audit trail for model decisions

## Command Line Interface
Standard training command:
```bash
python vertex_ai_trainer.py --action train --dataset DATASET_NAME --dataset-type 1 --bucket_name BUCKET_NAME --project_id PROJECT_ID --region us-central1
```

All parameters must be explicitly provided - no implicit defaults from hardcoded values.

## File Modification Guidelines
- Always read files before editing
- Preserve existing code structure and patterns
- Add comprehensive error handling
- Include clear documentation for complex logic
- Test parameter validation thoroughly

## Testing Requirements
- Validate all parameter combinations
- Test error conditions and edge cases
- Verify GCS connectivity before training
- Confirm model loading with HuggingFace token
- Check quota availability for selected region

This configuration ensures consistent, secure, and reliable medical AI model training with proper parameter validation and error handling.