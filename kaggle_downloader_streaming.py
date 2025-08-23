#!/usr/bin/env python3
"""
Memory-efficient streaming Kaggle downloader
Downloads and processes files one by one to avoid memory issues
"""

import os
import sys
import tempfile
import zipfile
from pathlib import Path
import logging
import subprocess
import gc

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_packages():
    """Install required packages."""
    packages = ["kaggle", "google-cloud-storage"]
    for package in packages:
        logger.info(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def download_and_stream_to_gcs():
    """Download and stream files to GCS one by one to avoid memory issues."""
    
    # Get parameters from environment
    bucket_name = os.environ['BUCKET_NAME']
    target_folder = os.environ.get('TARGET_FOLDER', 'dataset3_augmented_resized')
    kaggle_username = os.environ['KAGGLE_USERNAME']
    kaggle_key = os.environ['KAGGLE_KEY']
    
    logger.info(f"📋 Config: bucket={bucket_name}, folder={target_folder}")
    logger.info(f"🧠 Available memory info:")
    
    # Check available memory
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemAvailable' in line or 'MemTotal' in line:
                    logger.info(f"   {line.strip()}")
    except:
        pass
    
    # Install packages
    install_packages()
    
    # Set Kaggle environment
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
    
    logger.info(f"🔄 Starting streaming download of {dataset_slug}")
    
    # Use /tmp for temporary files
    temp_dir = Path("/tmp")
    download_dir = temp_dir / "kaggle_download"
    download_dir.mkdir(exist_ok=True)
    
    try:
        # Download WITHOUT extracting to avoid memory issues
        logger.info("📥 Downloading compressed files from Kaggle...")
        kaggle.api.dataset_download_files(
            dataset_slug,
            path=str(download_dir),
            unzip=False  # Keep compressed to save memory
        )
        logger.info("✅ Download completed")
        
        # Find the zip file
        zip_files = list(download_dir.glob("*.zip"))
        if not zip_files:
            raise ValueError("No zip file found after download")
        
        zip_file = zip_files[0]
        logger.info(f"📦 Found zip file: {zip_file} ({zip_file.stat().st_size / 1024 / 1024:.1f} MB)")
        
        # Stream extraction and upload
        logger.info("🔄 Starting streaming extraction and upload...")
        
        uploaded_count = 0
        target_files = 0
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # First pass: count target files
            logger.info("📊 Counting target files...")
            for file_info in zip_ref.filelist:
                if target_subfolder in file_info.filename and not file_info.is_dir():
                    target_files += 1
            
            logger.info(f"Found {target_files:,} target files to process")
            
            # Second pass: extract and upload one by one
            logger.info("🔄 Processing files...")
            
            for file_info in zip_ref.filelist:
                # Only process files in our target subfolder
                if target_subfolder not in file_info.filename or file_info.is_dir():
                    continue
                
                try:
                    # Extract single file to memory
                    with zip_ref.open(file_info) as source:
                        file_data = source.read()
                    
                    # Calculate GCS path
                    # Remove everything before target_subfolder
                    path_parts = Path(file_info.filename).parts
                    target_index = None
                    for i, part in enumerate(path_parts):
                        if target_subfolder in part:
                            target_index = i
                            break
                    
                    if target_index is not None:
                        # Keep path from target_subfolder onwards, but replace target_subfolder with our structure
                        relative_path = Path(*path_parts[target_index + 1:])
                        gcs_path = f"{target_folder}/{relative_path}"
                        
                        # Upload to GCS
                        blob = bucket.blob(gcs_path)
                        
                        # Check if already exists (for resume capability)
                        if not blob.exists():
                            blob.upload_from_string(file_data)
                        
                        uploaded_count += 1
                        
                        # Progress logging
                        if uploaded_count % 500 == 0:
                            progress = (uploaded_count / target_files) * 100
                            logger.info(f"Progress: {uploaded_count:,}/{target_files:,} ({progress:.1f}%)")
                            
                            # Force garbage collection to free memory
                            gc.collect()
                    
                    # Clear file data from memory immediately
                    del file_data
                    
                except Exception as e:
                    logger.warning(f"⚠️  Failed to process {file_info.filename}: {str(e)}")
                    continue
        
        logger.info(f"🎉 Uploaded {uploaded_count:,} files successfully!")
        
        # Verification
        logger.info("🔍 Quick verification...")
        try:
            # Count files in GCS
            blobs = list(storage_client.list_blobs(bucket_name, prefix=f"{target_folder}/"))
            gcs_count = len([b for b in blobs if not b.name.endswith('/')])
            
            logger.info(f"📊 GCS verification: {gcs_count:,} files found")
            
            if gcs_count >= uploaded_count * 0.95:  # 95% tolerance
                logger.info("✅ Upload verification successful")
            else:
                logger.warning(f"⚠️  Upload may be incomplete: {gcs_count}/{uploaded_count}")
                
        except Exception as e:
            logger.warning(f"⚠️  Verification failed: {str(e)}")
        
    finally:
        # Cleanup
        logger.info("🧹 Cleaning up...")
        try:
            import shutil
            shutil.rmtree(download_dir)
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.warning(f"⚠️  Cleanup warning: {str(e)}")

if __name__ == "__main__":
    try:
        download_and_stream_to_gcs()
        logger.info("✅ Streaming download completed successfully!")
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise