#!/bin/bash

# Quick Kaggle Dataset Download Script
# Downloads augmented_resized_V2 from Kaggle directly to GCS

echo "🚀 Kaggle Dataset Downloader for Diabetic Retinopathy"
echo "======================================================"
echo ""

# Check if credentials are provided
if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
    echo "❌ Error: Kaggle credentials not set"
    echo ""
    echo "Please set your Kaggle credentials:"
    echo "export KAGGLE_USERNAME='your_username'"
    echo "export KAGGLE_KEY='your_api_key'"
    echo ""
    echo "Get your credentials from: https://www.kaggle.com/settings/account"
    exit 1
fi

# Set default values
PROJECT_ID=${PROJECT_ID:-"your-project-id"}
BUCKET_NAME=${BUCKET_NAME:-"dr-data-2"}
REGION=${REGION:-"us-central1"}
TARGET_FOLDER=${TARGET_FOLDER:-"dataset3_augmented_resized"}

echo "📋 Configuration:"
echo "  Project ID: $PROJECT_ID"
echo "  Bucket: $BUCKET_NAME"
echo "  Target Folder: $TARGET_FOLDER"
echo "  Region: $REGION"
echo ""

# Choose download method
echo "Choose download method:"
echo "1. Vertex AI Job (Recommended - faster, more reliable)"
echo "2. Direct download (Local execution)"
echo ""
read -p "Enter choice (1 or 2): " choice

case $choice in
    1)
        echo "🔄 Starting Vertex AI download job..."
        python vertex_ai_kaggle_downloader.py \
            --action download \
            --kaggle_username "$KAGGLE_USERNAME" \
            --kaggle_key "$KAGGLE_KEY" \
            --bucket_name "$BUCKET_NAME" \
            --project_id "$PROJECT_ID" \
            --region "$REGION" \
            --target_folder "$TARGET_FOLDER"
        ;;
    2)
        echo "🔄 Starting direct download..."
        python colab_kaggle_downloader.py \
            --kaggle_username "$KAGGLE_USERNAME" \
            --kaggle_key "$KAGGLE_KEY" \
            --bucket_name "$BUCKET_NAME" \
            --target_folder "$TARGET_FOLDER"
        ;;
    *)
        echo "❌ Invalid choice. Please run again and select 1 or 2."
        exit 1
        ;;
esac

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Download completed successfully!"
    echo ""
    echo "📁 Dataset location: gs://$BUCKET_NAME/$TARGET_FOLDER"
    echo ""
    echo "🎯 Next steps:"
    echo "1. Verify the dataset structure in GCS console"
    echo "2. Train your model:"
    echo "   python vertex_ai_trainer.py \\"
    echo "     --action train \\"
    echo "     --dataset $TARGET_FOLDER \\"
    echo "     --dataset-type 1 \\"
    echo "     --bucket_name $BUCKET_NAME \\"
    echo "     --project_id $PROJECT_ID"
else
    echo ""
    echo "❌ Download failed. Check the error messages above."
    exit 1
fi