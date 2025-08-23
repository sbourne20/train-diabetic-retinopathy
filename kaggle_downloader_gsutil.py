#!/usr/bin/env python3
"""
Ultra-efficient Kaggle downloader using gsutil for parallel uploads
This should handle large datasets better with memory optimization
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_packages():
    """Install required packages."""
    packages = ["kaggle"]
    for package in packages:
        logger.info(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def download_and_upload_with_gsutil():
    """Download dataset and upload using gsutil for better performance."""
    
    # Get parameters from environment
    bucket_name = os.environ['BUCKET_NAME']
    target_folder = os.environ.get('TARGET_FOLDER', 'dataset3_augmented_resized')
    kaggle_username = os.environ['KAGGLE_USERNAME']
    kaggle_key = os.environ['KAGGLE_KEY']
    
    logger.info(f"📋 Config: bucket={bucket_name}, folder={target_folder}")
    
    # Install packages
    install_packages()
    
    # Set Kaggle environment
    os.environ['KAGGLE_USERNAME'] = kaggle_username
    os.environ['KAGGLE_KEY'] = kaggle_key
    
    # Import after setting environment
    import kaggle
    
    dataset_slug = "ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy"
    target_subfolder = "augmented_resized_V2"
    
    logger.info(f"🔄 Starting download of {dataset_slug}")
    
    # Use /tmp for download (usually has more space in containers)
    download_dir = Path("/tmp/kaggle_download")
    download_dir.mkdir(exist_ok=True)
    
    try:
        # Download dataset
        logger.info("📥 Downloading from Kaggle...")
        kaggle.api.dataset_download_files(
            dataset_slug,
            path=str(download_dir),
            unzip=True
        )
        logger.info("✅ Download completed")
        
        # Find target folder
        augmented_folder = None
        for item in download_dir.rglob("*"):
            if item.is_dir() and target_subfolder in item.name:
                augmented_folder = item
                break
        
        if not augmented_folder:
            raise ValueError(f"Could not find {target_subfolder} folder")
        
        logger.info(f"✅ Found: {augmented_folder}")
        
        # Count files for progress tracking
        logger.info("📊 Counting files...")
        file_count = sum(1 for _ in augmented_folder.rglob("*") if _.is_file())
        logger.info(f"Found {file_count:,} files to upload")
        
        # Use gsutil for parallel upload (much faster)
        gcs_destination = f"gs://{bucket_name}/{target_folder}"
        logger.info(f"📤 Uploading to {gcs_destination} using gsutil...")
        
        # gsutil command with parallel processing
        gsutil_cmd = [
            "gsutil", "-m",  # Enable parallel processing
            "cp", "-r",      # Copy recursively
            str(augmented_folder) + "/*",  # Source (all contents)
            gcs_destination  # Destination
        ]
        
        logger.info(f"Running: {' '.join(gsutil_cmd)}")
        
        # Run gsutil with real-time output
        process = subprocess.Popen(
            gsutil_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output in real-time
        for line in process.stdout:
            if line.strip():
                logger.info(f"gsutil: {line.strip()}")
        
        # Wait for completion
        return_code = process.wait()
        
        if return_code == 0:
            logger.info("🎉 Upload completed successfully!")
        else:
            raise subprocess.CalledProcessError(return_code, gsutil_cmd)
        
        # Verification
        logger.info("🔍 Verifying upload...")
        verify_cmd = ["gsutil", "ls", "-l", f"{gcs_destination}/**"]
        result = subprocess.run(verify_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Count uploaded files
            lines = result.stdout.strip().split('\n')
            uploaded_count = len([line for line in lines if not line.strip().endswith(':') and line.strip()])
            logger.info(f"📊 Verification: {uploaded_count:,} files uploaded successfully")
            
            if uploaded_count >= file_count * 0.95:  # Allow 5% tolerance
                logger.info("✅ Upload verification successful")
            else:
                logger.warning(f"⚠️  Upload may be incomplete: {uploaded_count}/{file_count}")
        else:
            logger.warning("⚠️  Could not verify upload")
        
    finally:
        # Cleanup download directory
        logger.info("🧹 Cleaning up temporary files...")
        try:
            import shutil
            shutil.rmtree(download_dir)
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.warning(f"⚠️  Cleanup warning: {str(e)}")

if __name__ == "__main__":
    try:
        download_and_upload_with_gsutil()
        logger.info("✅ Process completed successfully!")
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise