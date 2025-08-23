#!/usr/bin/env python3
"""
Colab Enterprise script to download Kaggle dataset directly to GCS
Alternative to Vertex AI approach - can be run in Colab Enterprise
"""

import os
import sys
import tempfile
import zipfile
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_dependencies():
    """Install required packages in Colab."""
    packages = [
        "kaggle",
        "google-cloud-storage"
    ]
    
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
            logger.info(f"✅ {package} already installed")
        except ImportError:
            logger.info(f"Installing {package}...")
            os.system(f"pip install {package}")

def setup_kaggle_credentials(kaggle_username: str, kaggle_key: str):
    """Setup Kaggle API credentials."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    kaggle_json = kaggle_dir / "kaggle.json"
    kaggle_json.write_text(f'{{"username":"{kaggle_username}","key":"{kaggle_key}"}}')
    kaggle_json.chmod(0o600)
    
    logger.info("✅ Kaggle credentials configured")

def authenticate_gcs():
    """Authenticate with Google Cloud (for Colab Enterprise)."""
    try:
        from google.colab import auth
        auth.authenticate_user()
        logger.info("✅ Authenticated with Google Cloud")
    except ImportError:
        logger.info("Not running in Colab - assuming authentication is already set up")

def download_kaggle_dataset_to_gcs(
    kaggle_username: str, 
    kaggle_key: str,
    bucket_name: str,
    target_folder: str = "dataset3_augmented_resized"
):
    """Download Kaggle dataset directly to GCS."""
    
    # Install dependencies
    install_dependencies()
    
    # Setup authentication
    authenticate_gcs()
    
    # Setup Kaggle credentials
    setup_kaggle_credentials(kaggle_username, kaggle_key)
    
    # Import libraries
    import kaggle
    from google.cloud import storage
    
    # Initialize GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    dataset_slug = "ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy"
    target_subfolder = "augmented_resized_V2"
    
    logger.info(f"Starting download of {dataset_slug}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        download_path = temp_path / "download"
        download_path.mkdir()
        
        # Download dataset
        logger.info("📥 Downloading dataset from Kaggle...")
        kaggle.api.dataset_download_files(
            dataset_slug,
            path=str(download_path),
            unzip=True
        )
        
        # Find augmented_resized_V2 folder
        augmented_folder = None
        for item in download_path.rglob("*"):
            if item.is_dir() and target_subfolder in item.name:
                augmented_folder = item
                break
        
        if not augmented_folder:
            raise ValueError(f"Could not find {target_subfolder} folder in downloaded dataset")
        
        logger.info(f"✅ Found target folder: {augmented_folder}")
        
        # Upload to GCS
        logger.info(f"📤 Uploading to gs://{bucket_name}/{target_folder}")
        
        total_files = sum(1 for _ in augmented_folder.rglob("*") if _.is_file())
        uploaded_files = 0
        
        for local_file in augmented_folder.rglob("*"):
            if local_file.is_file():
                # Calculate relative path
                relative_path = local_file.relative_to(augmented_folder)
                gcs_path = f"{target_folder}/{relative_path}"
                
                # Upload file
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(str(local_file))
                
                uploaded_files += 1
                if uploaded_files % 100 == 0:
                    progress = (uploaded_files / total_files) * 100
                    logger.info(f"Progress: {uploaded_files}/{total_files} files ({progress:.1f}%)")
        
        logger.info(f"✅ Successfully uploaded {uploaded_files} files to gs://{bucket_name}/{target_folder}")
    
    # Verify upload
    verify_dataset_upload(storage_client, bucket_name, target_folder)
    
    logger.info("🎉 Dataset download and upload completed successfully!")

def verify_dataset_upload(storage_client, bucket_name: str, target_folder: str):
    """Verify the uploaded dataset structure."""
    logger.info("🔍 Verifying uploaded dataset structure...")
    
    # Count files per class
    file_counts = {}
    for split in ['train', 'val', 'test']:
        file_counts[split] = {}
        for class_id in ['0', '1', '2', '3', '4']:
            prefix = f"{target_folder}/{split}/{class_id}/"
            blobs = list(storage_client.list_blobs(bucket_name, prefix=prefix))
            count = len([b for b in blobs if not b.name.endswith('/')])
            file_counts[split][class_id] = count
    
    # Print summary
    logger.info("📊 Dataset upload summary:")
    total_files = 0
    for split, classes in file_counts.items():
        split_total = sum(classes.values())
        total_files += split_total
        logger.info(f"  {split.upper()} ({split_total} files):")
        for class_id, count in classes.items():
            logger.info(f"    Class {class_id}: {count:,} files")
    
    logger.info(f"📈 Total files uploaded: {total_files:,}")
    
    # Verify expected distribution (approximately)
    expected_train_total = 115000  # From our earlier analysis
    if file_counts['train']['0'] + file_counts['train']['1'] + file_counts['train']['2'] + file_counts['train']['3'] + file_counts['train']['4'] >= expected_train_total * 0.9:
        logger.info("✅ Upload appears complete based on expected file counts")
    else:
        logger.warning("⚠️  Upload may be incomplete - fewer files than expected")

# Example usage functions for different environments
def run_in_colab():
    """Example function to run in Colab Enterprise."""
    print("=== Kaggle Dataset Downloader for Colab Enterprise ===")
    print()
    
    # Get credentials from user
    kaggle_username = input("Enter your Kaggle username: ")
    kaggle_key = input("Enter your Kaggle API key: ")
    bucket_name = input("Enter GCS bucket name (e.g., dr-data-2): ")
    target_folder = input("Enter target folder name [dataset3_augmented_resized]: ") or "dataset3_augmented_resized"
    
    print(f"\n🚀 Starting download to gs://{bucket_name}/{target_folder}")
    
    try:
        download_kaggle_dataset_to_gcs(
            kaggle_username=kaggle_username,
            kaggle_key=kaggle_key,
            bucket_name=bucket_name,
            target_folder=target_folder
        )
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise

def run_with_args():
    """Run with command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Kaggle dataset to GCS")
    parser.add_argument("--kaggle_username", required=True, help="Kaggle username")
    parser.add_argument("--kaggle_key", required=True, help="Kaggle API key")
    parser.add_argument("--bucket_name", required=True, help="GCS bucket name")
    parser.add_argument("--target_folder", default="dataset3_augmented_resized", 
                       help="Target folder in GCS bucket")
    
    args = parser.parse_args()
    
    download_kaggle_dataset_to_gcs(
        kaggle_username=args.kaggle_username,
        kaggle_key=args.kaggle_key,
        bucket_name=args.bucket_name,
        target_folder=args.target_folder
    )

if __name__ == "__main__":
    # Check if running in Colab (interactive)
    try:
        import google.colab
        run_in_colab()
    except ImportError:
        # Running from command line
        run_with_args()