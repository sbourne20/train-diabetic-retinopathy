# Vertex AI Training Setup Guide

This guide will help you train your diabetic retinopathy model on Google Cloud Vertex AI.

## Prerequisites

1. **Google Cloud Project** with billing enabled
2. **Vertex AI API** enabled
3. **Cloud Storage API** enabled
4. **Service Account** with appropriate permissions
5. **gcloud CLI** installed and authenticated

## Step 1: Initial Setup

### 1.1 Install Dependencies
```bash
pip install google-cloud-aiplatform google-cloud-storage
```

### 1.2 Authenticate with Google Cloud
```bash
# Login to your Google Cloud account
gcloud auth login

# Set your project
gcloud config set project YOUR_PROJECT_ID

# Create application default credentials
gcloud auth application-default login
```

### 1.3 Enable Required APIs
```bash
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable compute.googleapis.com
```

## Step 2: Create Cloud Storage Bucket

### 2.1 Setup GCS Bucket
```bash
python setup_gcs.py \
    --project_id YOUR_PROJECT_ID \
    --bucket_name your-dr-training-bucket \
    --region us-central1
```

### 2.2 Configure Environment Variables
```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_REGION="us-central1"
export GCS_BUCKET="your-dr-training-bucket"
```

## Step 3: Upload Dataset and Dependencies

### 3.1 Upload Your Dataset
```bash
python vertex_ai_trainer.py \
    --action upload \
    --dataset_path ./dataset \
    --project_id YOUR_PROJECT_ID \
    --bucket_name your-dr-training-bucket
```

### 3.2 Upload RETFound Weights (if you have them)
```bash
gsutil cp path/to/RETFound_cfp_weights.pth gs://your-dr-training-bucket/models/
```

## Step 4: Start Training

### 4.1 Basic Training Job
```bash
python vertex_ai_trainer.py \
    --action train \
    --project_id YOUR_PROJECT_ID \
    --bucket_name your-dr-training-bucket \
    --region us-central1
```

### 4.2 Hyperparameter Tuning Job
```bash
python vertex_ai_trainer.py \
    --action tune \
    --project_id YOUR_PROJECT_ID \
    --bucket_name your-dr-training-bucket \
    --region us-central1
```

## Step 5: Monitor Training

### 5.1 Monitor via CLI
```bash
python vertex_ai_trainer.py \
    --action monitor \
    --job_id YOUR_JOB_ID \
    --project_id YOUR_PROJECT_ID
```

### 5.2 Monitor via Console
Visit the [Vertex AI Console](https://console.cloud.google.com/vertex-ai/training/custom-jobs) to monitor your jobs visually.

## Step 6: Download Results

### 6.1 Download Training Results
```bash
python vertex_ai_trainer.py \
    --action download \
    --project_id YOUR_PROJECT_ID \
    --bucket_name your-dr-training-bucket
```

## Configuration Options

### Machine Types
- **n1-highmem-8**: 8 vCPUs, 52GB RAM (default)
- **n1-highmem-16**: 16 vCPUs, 104GB RAM (for larger datasets)
- **n1-standard-32**: 32 vCPUs, 120GB RAM (for distributed training)

### GPU Options
- **NVIDIA_TESLA_V100**: High performance (default)
- **NVIDIA_TESLA_T4**: Cost-effective option
- **NVIDIA_TESLA_A100**: Latest high-performance GPU

### Modify Configuration
Edit `vertex_ai_config.py` to customize:
```python
# Machine configuration
self.machine_type = "n1-highmem-16"  # Larger machine
self.accelerator_type = "NVIDIA_TESLA_A100"  # Better GPU
self.accelerator_count = 2  # Multiple GPUs
```

## Cost Optimization

### 1. Use Preemptible Instances
```python
# In vertex_ai_config.py, add:
"scheduling": {
    "restart_job_on_worker_restart": True
}
```

### 2. Choose Appropriate Machine Types
- Start with smaller machines (n1-highmem-8)
- Scale up only if needed
- Use T4 GPUs for development, V100/A100 for production

### 3. Monitor Costs
- Set up billing alerts
- Use `gcloud billing` commands to track usage
- Stop jobs when not needed

## Troubleshooting

### Common Issues

#### 1. Permission Errors
```bash
# Grant Vertex AI permissions to your service account
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:YOUR_SERVICE_ACCOUNT@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:YOUR_SERVICE_ACCOUNT@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"
```

#### 2. Package Installation Errors
- Ensure all dependencies are in `requirements.txt`
- Check container compatibility
- Use specific package versions

#### 3. Out of Memory Errors
- Reduce batch size
- Use larger machine type
- Enable gradient checkpointing in config

#### 4. Timeout Issues
- Increase job timeout in configuration
- Use checkpointing to resume training
- Split large datasets into smaller chunks

### Debugging Training Jobs

#### 1. Check Logs
```bash
# View job logs
gcloud ai custom-jobs describe JOB_ID \
    --region=us-central1 \
    --project=YOUR_PROJECT_ID
```

#### 2. Access Job Outputs
```bash
# List output files
gsutil ls gs://your-dr-training-bucket/outputs/

# Download specific logs
gsutil cp gs://your-dr-training-bucket/outputs/*/logs/* ./logs/
```

## Best Practices

### 1. Data Management
- Upload dataset once, reuse for multiple experiments
- Use efficient data formats (preprocessed tensors)
- Implement data validation

### 2. Experiment Tracking
- Use meaningful job names
- Tag experiments with metadata
- Save hyperparameters with results

### 3. Model Versioning
- Save models with version numbers
- Track model performance metrics
- Use model registry for production models

### 4. Security
- Use service accounts with minimal permissions
- Enable VPC-native clusters
- Encrypt sensitive data

## Example Commands Summary

```bash
# 1. Setup
python setup_gcs.py --project_id PROJECT --bucket_name BUCKET

# 2. Upload data
python vertex_ai_trainer.py --action upload --dataset_path ./dataset

# 3. Train model
python vertex_ai_trainer.py --action train

# 4. Monitor training
python vertex_ai_trainer.py --action monitor --job_id JOB_ID

# 5. Download results
python vertex_ai_trainer.py --action download
```

## Next Steps

After training completes:
1. **Evaluate** your model performance
2. **Deploy** to Vertex AI Endpoints for inference
3. **Monitor** model performance in production
4. **Retrain** with new data as needed

For deployment to Vertex AI Endpoints, see the deployment guide in the next section.