"""
Cloud utilities for Google Cloud Storage integration
"""

import os
import tempfile
import shutil
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def is_gcs_path(path: str) -> bool:
    """Check if path is a GCS path."""
    return path.startswith('gs://')

def download_from_gcs(gcs_path: str, local_path: str):
    """Download file or directory from GCS to local path."""
    try:
        from google.cloud import storage
        
        # Parse GCS path
        path_parts = gcs_path.replace('gs://', '').split('/', 1)
        bucket_name = path_parts[0]
        blob_prefix = path_parts[1] if len(path_parts) > 1 else ''
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        os.makedirs(local_path, exist_ok=True)
        
        # List and download all blobs with the prefix
        blobs = bucket.list_blobs(prefix=blob_prefix)
        
        for blob in blobs:
            # Skip folder markers
            if blob.name.endswith('/'):
                continue
                
            # Create local file path
            relative_path = blob.name[len(blob_prefix):].lstrip('/')
            if not relative_path:
                continue
                
            local_file_path = os.path.join(local_path, relative_path)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            # Download file
            blob.download_to_filename(local_file_path)
            logger.info(f"Downloaded {blob.name} to {local_file_path}")
            
    except ImportError:
        logger.error("google-cloud-storage not installed. Cannot download from GCS.")
        raise
    except Exception as e:
        logger.error(f"Error downloading from GCS: {e}")
        raise

def upload_to_gcs(local_path: str, gcs_path: str):
    """Upload file or directory from local path to GCS."""
    try:
        from google.cloud import storage
        
        # Parse GCS path
        path_parts = gcs_path.replace('gs://', '').split('/', 1)
        bucket_name = path_parts[0]
        blob_prefix = path_parts[1] if len(path_parts) > 1 else ''
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        if os.path.isfile(local_path):
            # Upload single file
            blob = bucket.blob(blob_prefix)
            blob.upload_from_filename(local_path)
            logger.info(f"Uploaded {local_path} to {gcs_path}")
        
        elif os.path.isdir(local_path):
            # Upload directory
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_file_path, local_path)
                    blob_name = os.path.join(blob_prefix, relative_path).replace('\\', '/')
                    
                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(local_file_path)
                    logger.info(f"Uploaded {local_file_path} to gs://{bucket_name}/{blob_name}")
        
    except ImportError:
        logger.error("google-cloud-storage not installed. Cannot upload to GCS.")
        raise
    except Exception as e:
        logger.error(f"Error uploading to GCS: {e}")
        raise

def prepare_cloud_paths(config):
    """Prepare paths for cloud training by downloading datasets if needed."""
    
    # Check if paths are GCS paths and download if needed
    if is_gcs_path(config.data.rg_path):
        logger.info(f"Downloading RG dataset from {config.data.rg_path}")
        local_rg_path = "/tmp/dataset/RG"
        download_from_gcs(config.data.rg_path, local_rg_path)
        config.data.rg_path = local_rg_path
    
    if is_gcs_path(config.data.me_path):
        logger.info(f"Downloading ME dataset from {config.data.me_path}")
        local_me_path = "/tmp/dataset/ME"
        download_from_gcs(config.data.me_path, local_me_path)
        config.data.me_path = local_me_path
    
    # Download RETFound weights if on GCS
    if hasattr(config.model, 'pretrained_path') and is_gcs_path(config.model.pretrained_path):
        logger.info(f"Downloading RETFound weights from {config.model.pretrained_path}")
        local_weights_path = "/tmp/models/RETFound_cfp_weights.pth"
        os.makedirs(os.path.dirname(local_weights_path), exist_ok=True)
        download_from_gcs(config.model.pretrained_path, local_weights_path)
        config.model.pretrained_path = local_weights_path
    
    return config

def upload_results_to_cloud(local_output_dir: str, gcs_output_path: Optional[str] = None):
    """Upload training results to cloud storage."""
    
    if gcs_output_path and os.path.exists(local_output_dir):
        logger.info(f"Uploading results to {gcs_output_path}")
        upload_to_gcs(local_output_dir, gcs_output_path)
    
def sync_checkpoints_to_cloud(checkpoint_dir: str, gcs_checkpoint_path: Optional[str] = None):
    """Sync model checkpoints to cloud storage during training."""
    
    if gcs_checkpoint_path and os.path.exists(checkpoint_dir):
        # Upload only the best model and latest checkpoint
        best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            gcs_best_path = f"{gcs_checkpoint_path}/best_model.pth"
            upload_to_gcs(best_model_path, gcs_best_path)
            
        # Find and upload latest checkpoint
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_")]
        if checkpoints:
            latest_checkpoint = sorted(checkpoints)[-1]
            latest_path = os.path.join(checkpoint_dir, latest_checkpoint)
            gcs_latest_path = f"{gcs_checkpoint_path}/{latest_checkpoint}"
            upload_to_gcs(latest_path, gcs_latest_path)