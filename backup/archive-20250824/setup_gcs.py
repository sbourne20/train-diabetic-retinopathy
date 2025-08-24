#!/usr/bin/env python3
"""
Setup Google Cloud Storage for Vertex AI training
"""

import os
import argparse
from google.cloud import storage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_bucket(project_id: str, bucket_name: str, region: str = "us-central1"):
    """Create GCS bucket if it doesn't exist."""
    
    client = storage.Client(project=project_id)
    
    try:
        bucket = client.bucket(bucket_name)
        if bucket.exists():
            logger.info(f"Bucket {bucket_name} already exists")
            return bucket
        
        bucket = client.create_bucket(bucket_name, location=region)
        logger.info(f"Created bucket {bucket_name} in {region}")
        return bucket
        
    except Exception as e:
        logger.error(f"Error creating bucket: {e}")
        raise

def setup_bucket_structure(bucket_name: str, project_id: str):
    """Setup required folder structure in GCS bucket."""
    
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    
    # Create folder structure by uploading placeholder files
    folders = [
        "data/RG/",
        "data/ME/", 
        "models/",
        "outputs/",
        "checkpoints/"
    ]
    
    for folder in folders:
        placeholder_blob = bucket.blob(f"{folder}.gitkeep")
        if not placeholder_blob.exists():
            placeholder_blob.upload_from_string("")
            logger.info(f"Created folder: {folder}")

def upload_retfound_weights(bucket_name: str, project_id: str, local_weights_path: str):
    """Upload RETFound weights to GCS."""
    
    if not os.path.exists(local_weights_path):
        logger.warning(f"RETFound weights not found at {local_weights_path}")
        logger.info("Please download RETFound weights and upload manually to gs://{bucket_name}/models/")
        return
    
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    
    blob = bucket.blob("models/RETFound_cfp_weights.pth")
    blob.upload_from_filename(local_weights_path)
    logger.info(f"Uploaded RETFound weights to gs://{bucket_name}/models/RETFound_cfp_weights.pth")

def verify_setup(bucket_name: str, project_id: str):
    """Verify GCS setup is correct."""
    
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    
    logger.info("Verifying GCS setup...")
    
    # Check if bucket exists
    if not bucket.exists():
        logger.error(f"Bucket {bucket_name} does not exist")
        return False
    
    # List contents
    blobs = list(client.list_blobs(bucket_name, max_results=10))
    logger.info(f"Found {len(blobs)} objects in bucket")
    
    for blob in blobs:
        logger.info(f"  - {blob.name}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Setup GCS for Vertex AI training")
    parser.add_argument("--project_id", required=True, help="Google Cloud Project ID")
    parser.add_argument("--bucket_name", required=True, help="GCS bucket name")
    parser.add_argument("--region", default="us-central1", help="GCS region")
    parser.add_argument("--retfound_weights", help="Path to RETFound weights file")
    
    args = parser.parse_args()
    
    # Create bucket
    create_bucket(args.project_id, args.bucket_name, args.region)
    
    # Setup folder structure
    setup_bucket_structure(args.bucket_name, args.project_id)
    
    # Upload RETFound weights if provided
    if args.retfound_weights:
        upload_retfound_weights(args.bucket_name, args.project_id, args.retfound_weights)
    
    # Verify setup
    verify_setup(args.bucket_name, args.project_id)
    
    logger.info("GCS setup completed!")
    logger.info(f"Bucket: gs://{args.bucket_name}")
    logger.info("Next steps:")
    logger.info("1. Upload your dataset using: python vertex_ai_trainer.py --action upload")
    logger.info("2. Start training using: python vertex_ai_trainer.py --action train")

if __name__ == "__main__":
    main()