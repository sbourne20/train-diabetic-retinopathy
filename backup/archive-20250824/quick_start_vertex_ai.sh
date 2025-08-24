#!/bin/bash

# Quick Start Script for Vertex AI Training
# This script helps you get started with training on Vertex AI quickly

set -e

echo "🚀 Diabetic Retinopathy Model - Vertex AI Quick Start"
echo "=================================================="

# Configuration
PROJECT_ID="${1:-your-project-id}"
BUCKET_NAME="${2:-dr-training-$(date +%Y%m%d)}"
REGION="${3:-us-central1}"

if [ "$PROJECT_ID" = "your-project-id" ]; then
    echo "❌ Please provide your Google Cloud Project ID"
    echo "Usage: $0 <PROJECT_ID> [BUCKET_NAME] [REGION]"
    echo "Example: $0 my-project-123 my-dr-bucket us-central1"
    exit 1
fi

echo "📋 Configuration:"
echo "   Project ID: $PROJECT_ID"
echo "   Bucket: $BUCKET_NAME"
echo "   Region: $REGION"
echo ""

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Not authenticated with gcloud. Please run: gcloud auth login"
    exit 1
fi

# Set project
echo "🔧 Setting up project..."
gcloud config set project $PROJECT_ID

# Enable APIs
echo "🔌 Enabling required APIs..."
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable compute.googleapis.com

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install google-cloud-aiplatform google-cloud-storage

# Setup GCS bucket
echo "🪣 Setting up Cloud Storage..."
python setup_gcs.py \
    --project_id $PROJECT_ID \
    --bucket_name $BUCKET_NAME \
    --region $REGION

# Upload dataset
echo "📊 Uploading dataset..."
if [ -d "dataset" ]; then
    python vertex_ai_trainer.py \
        --action upload \
        --dataset_path ./dataset \
        --project_id $PROJECT_ID \
        --bucket_name $BUCKET_NAME
else
    echo "⚠️  Dataset directory not found. Please ensure 'dataset' folder exists with RG and ME subfolders."
    echo "   You can upload it later using:"
    echo "   python vertex_ai_trainer.py --action upload --dataset_path ./dataset --project_id $PROJECT_ID --bucket_name $BUCKET_NAME"
fi

# Set environment variables
echo "🌍 Setting environment variables..."
export GOOGLE_CLOUD_PROJECT=$PROJECT_ID
export GOOGLE_CLOUD_REGION=$REGION
export GCS_BUCKET=$BUCKET_NAME

# Create .env file for persistence
cat > .env << EOF
GOOGLE_CLOUD_PROJECT=$PROJECT_ID
GOOGLE_CLOUD_REGION=$REGION
GCS_BUCKET=$BUCKET_NAME
EOF

echo "✅ Setup completed!"
echo ""
echo "🚀 Next steps:"
echo "1. Start training:"
echo "   python vertex_ai_trainer.py --action train --project_id $PROJECT_ID --bucket_name $BUCKET_NAME"
echo ""
echo "2. Monitor training:"
echo "   Visit: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=$PROJECT_ID"
echo ""
echo "3. Or start hyperparameter tuning:"
echo "   python vertex_ai_trainer.py --action tune --project_id $PROJECT_ID --bucket_name $BUCKET_NAME"
echo ""
echo "4. Download results when complete:"
echo "   python vertex_ai_trainer.py --action download --project_id $PROJECT_ID --bucket_name $BUCKET_NAME"
echo ""
echo "💡 Tips:"
echo "   - Monitor costs: https://console.cloud.google.com/billing"
echo "   - View logs: gcloud ai custom-jobs describe JOB_ID --region=$REGION"
echo "   - Environment variables saved to .env file"
echo ""
echo "🎯 Your training bucket: gs://$BUCKET_NAME"