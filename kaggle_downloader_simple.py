#!/usr/bin/env python3
"""
Simplified Kaggle downloader that uses environment variables for authentication
This avoids the kaggle.json file issues in Vertex AI
"""

import os
import sys
import tempfile
import zipfile
from pathlib import Path
import logging
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_packages():
    """Install required packages."""
    packages = ["kaggle", "google-cloud-storage"]
    for package in packages:
        logger.info(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def download_kaggle_dataset_to_gcs():
    """Download dataset using environment variables for auth."""
    
    # Get parameters from environment or command line
    bucket_name = os.environ.get('BUCKET_NAME') or sys.argv[1]
    target_folder = os.environ.get('TARGET_FOLDER', 'dataset3_augmented_resized')
    kaggle_username = os.environ.get('KAGGLE_USERNAME')
    kaggle_key = os.environ.get('KAGGLE_KEY')
    
    if not all([bucket_name, kaggle_username, kaggle_key]):
        raise ValueError("Missing required parameters: BUCKET_NAME, KAGGLE_USERNAME, KAGGLE_KEY")
    
    logger.info(f"Parameters: bucket={bucket_name}, folder={target_folder}")
    
    # Install packages
    install_packages()
    
    # Set Kaggle environment variables
    os.environ['KAGGLE_USERNAME'] = kaggle_username
    os.environ['KAGGLE_KEY'] = kaggle_key
    
    # Import after setting environment
    import kaggle
    from google.cloud import storage
    
    # Initialize GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    dataset_slug = "ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy"
    target_subfolder = "augmented_resized_V2"
    
    logger.info(f"🔄 Starting download of {dataset_slug}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        download_path = temp_path / "download"
        download_path.mkdir()
        
        # Download dataset
        logger.info("📥 Downloading from Kaggle...")
        try:
            kaggle.api.dataset_download_files(
                dataset_slug,
                path=str(download_path),
                unzip=True
            )
            logger.info("✅ Download completed")
        except Exception as e:
            logger.error(f"❌ Download failed: {str(e)}")
            # Try downloading without unzip and extract manually
            logger.info("🔄 Trying manual extraction...")
            kaggle.api.dataset_download_files(
                dataset_slug,
                path=str(download_path),
                unzip=False
            )
            
            # Extract manually with memory management
            import zipfile
            zip_file = None
            for f in download_path.glob("*.zip"):
                zip_file = f
                break
            
            if zip_file:
                logger.info(f"📦 Extracting {zip_file}...")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(download_path)
                logger.info("✅ Extraction completed")
                zip_file.unlink()  # Delete zip file to save space
            else:
                raise ValueError("No zip file found after download")
        
        # Find target folder
        augmented_folder = None
        for item in download_path.rglob("*"):
            if item.is_dir() and target_subfolder in item.name:
                augmented_folder = item
                break
        
        if not augmented_folder:
            raise ValueError(f"Could not find {target_subfolder} folder")
        
        logger.info(f"✅ Found: {augmented_folder}")
        
        # Upload to GCS with memory optimization
        logger.info(f"📤 Uploading to gs://{bucket_name}/{target_folder}")
        
        # Count files first (more memory efficient)
        logger.info("📊 Counting files...")
        file_list = []
        for local_file in augmented_folder.rglob("*"):
            if local_file.is_file():
                file_list.append(local_file)
        
        total_files = len(file_list)
        logger.info(f"Found {total_files:,} files to upload")
        
        # Upload in batches to manage memory
        batch_size = 100
        uploaded_files = 0
        
        for i in range(0, len(file_list), batch_size):
            batch = file_list[i:i + batch_size]
            
            for local_file in batch:
                relative_path = local_file.relative_to(augmented_folder)
                gcs_path = f"{target_folder}/{relative_path}"
                
                # Check if file already exists to resume interrupted uploads
                blob = bucket.blob(gcs_path)
                if not blob.exists():
                    blob.upload_from_filename(str(local_file))
                    uploaded_files += 1
                else:
                    uploaded_files += 1  # Count as uploaded
                
                if uploaded_files % 1000 == 0:
                    progress = (uploaded_files / total_files) * 100
                    logger.info(f"Progress: {uploaded_files:,}/{total_files:,} ({progress:.1f}%)")
            
            # Clear batch from memory
            del batch
        
        logger.info(f"🎉 Uploaded {uploaded_files} files successfully!")
        
        # Quick verification
        logger.info("🔍 Verifying upload...")
        train_count = len(list(storage_client.list_blobs(bucket_name, prefix=f"{target_folder}/train/")))
        val_count = len(list(storage_client.list_blobs(bucket_name, prefix=f"{target_folder}/val/")))
        test_count = len(list(storage_client.list_blobs(bucket_name, prefix=f"{target_folder}/test/")))
        
        logger.info(f"📊 Upload summary:")
        logger.info(f"  Train: {train_count} files")
        logger.info(f"  Val: {val_count} files") 
        logger.info(f"  Test: {test_count} files")
        logger.info(f"  Total: {train_count + val_count + test_count} files")

if __name__ == "__main__":
    try:
        download_kaggle_dataset_to_gcs()
        logger.info("✅ Download completed successfully!")
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise