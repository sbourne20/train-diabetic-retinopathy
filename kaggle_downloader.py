#!/usr/bin/env python3
"""
Vertex AI job to download Kaggle dataset directly to GCS
Downloads only augmented_resized_V2 folder from the diabetic retinopathy dataset
"""

import os
import sys
import zipfile
import tempfile
import shutil
import logging
from pathlib import Path
from google.cloud import storage
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KaggleToGCSDownloader:
    """Downloads Kaggle dataset directly to GCS using Vertex AI."""
    
    def __init__(self, bucket_name: str, target_folder: str = "dataset3_augmented_resized"):
        self.bucket_name = bucket_name
        self.target_folder = target_folder
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        
        # Kaggle dataset info
        self.dataset_slug = "ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy"
        self.target_subfolder = "augmented_resized_V2"
    
    def setup_kaggle_credentials(self, kaggle_username: str, kaggle_key: str):
        """Setup Kaggle API credentials."""
        # Try both standard locations to ensure compatibility
        kaggle_dirs = [
            Path.home() / ".kaggle",
            Path("/root/.config/kaggle"),  # Alternative location mentioned in error
            Path("/root/.kaggle")  # Standard location for root user
        ]
        
        for kaggle_dir in kaggle_dirs:
            kaggle_dir.mkdir(parents=True, exist_ok=True)
            kaggle_json = kaggle_dir / "kaggle.json"
            kaggle_json.write_text(f'{{"username":"{kaggle_username}","key":"{kaggle_key}"}}')
            kaggle_json.chmod(0o600)
            logger.info(f"Kaggle credentials configured in {kaggle_dir}")
        
        # Also set environment variables as backup
        os.environ['KAGGLE_USERNAME'] = kaggle_username
        os.environ['KAGGLE_KEY'] = kaggle_key
        logger.info("Kaggle credentials also set via environment variables")
    
    def install_kaggle_api(self):
        """Install Kaggle API if not available."""
        try:
            # Just check if kaggle package exists without triggering authentication
            import importlib.util
            spec = importlib.util.find_spec("kaggle")
            if spec is not None:
                logger.info("Kaggle API already available")
            else:
                raise ImportError("Kaggle not found")
        except ImportError:
            logger.info("Installing Kaggle API...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
            logger.info("Kaggle API installed successfully")
    
    def download_and_extract_dataset(self):
        """Download dataset from Kaggle and extract to GCS."""
        logger.info(f"Starting download of {self.dataset_slug}")
        
        # Import kaggle AFTER credentials are set up
        logger.info("Importing Kaggle API...")
        import kaggle
        logger.info("Kaggle API imported successfully")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            download_path = temp_path / "download"
            extract_path = temp_path / "extract"
            
            download_path.mkdir()
            extract_path.mkdir()
            
            # Download dataset
            logger.info("Downloading dataset from Kaggle...")
            kaggle.api.dataset_download_files(
                self.dataset_slug,
                path=str(download_path),
                unzip=True
            )
            
            # Find and extract only augmented_resized_V2
            augmented_folder = None
            for item in download_path.rglob("*"):
                if item.is_dir() and self.target_subfolder in item.name:
                    augmented_folder = item
                    break
            
            if not augmented_folder:
                raise ValueError(f"Could not find {self.target_subfolder} folder in downloaded dataset")
            
            logger.info(f"Found target folder: {augmented_folder}")
            
            # Upload to GCS
            self._upload_folder_to_gcs(augmented_folder, self.target_folder)
    
    def _upload_folder_to_gcs(self, local_folder: Path, gcs_prefix: str):
        """Upload a local folder to GCS."""
        logger.info(f"Uploading {local_folder} to gs://{self.bucket_name}/{gcs_prefix}")
        
        total_files = sum(1 for _ in local_folder.rglob("*") if _.is_file())
        uploaded_files = 0
        
        for local_file in local_folder.rglob("*"):
            if local_file.is_file():
                # Calculate relative path from the source folder
                relative_path = local_file.relative_to(local_folder)
                gcs_path = f"{gcs_prefix}/{relative_path}"
                
                # Upload file
                blob = self.bucket.blob(gcs_path)
                blob.upload_from_filename(str(local_file))
                
                uploaded_files += 1
                if uploaded_files % 100 == 0:
                    logger.info(f"Uploaded {uploaded_files}/{total_files} files")
        
        logger.info(f"Successfully uploaded {uploaded_files} files to gs://{self.bucket_name}/{gcs_prefix}")
    
    def verify_upload(self):
        """Verify the uploaded dataset structure."""
        logger.info("Verifying uploaded dataset structure...")
        
        # Check for required directories
        required_dirs = []
        for split in ['train', 'val', 'test']:
            for class_id in ['0', '1', '2', '3', '4']:
                required_dirs.append(f"{self.target_folder}/{split}/{class_id}/")
        
        existing_dirs = set()
        blobs = self.storage_client.list_blobs(self.bucket_name, prefix=f"{self.target_folder}/")
        
        for blob in blobs:
            dir_path = "/".join(blob.name.split("/")[:-1]) + "/"
            existing_dirs.add(dir_path)
        
        missing_dirs = []
        for required_dir in required_dirs:
            if required_dir not in existing_dirs:
                missing_dirs.append(required_dir)
        
        if missing_dirs:
            logger.warning(f"Missing directories: {missing_dirs}")
        else:
            logger.info("✅ All required directories found")
        
        # Count files per class
        file_counts = {}
        for split in ['train', 'val', 'test']:
            file_counts[split] = {}
            for class_id in ['0', '1', '2', '3', '4']:
                prefix = f"{self.target_folder}/{split}/{class_id}/"
                count = len(list(self.storage_client.list_blobs(self.bucket_name, prefix=prefix)))
                file_counts[split][class_id] = count
        
        # Print summary
        logger.info("Dataset upload summary:")
        for split, classes in file_counts.items():
            logger.info(f"  {split.upper()}:")
            for class_id, count in classes.items():
                logger.info(f"    Class {class_id}: {count} files")

def main():
    """Main function for running as Vertex AI job."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Kaggle dataset to GCS")
    parser.add_argument("--bucket_name", required=True, help="GCS bucket name")
    parser.add_argument("--target_folder", default="dataset3_augmented_resized", 
                       help="Target folder in GCS bucket")
    parser.add_argument("--kaggle_username", required=True, help="Kaggle username")
    parser.add_argument("--kaggle_key", required=True, help="Kaggle API key")
    
    args = parser.parse_args()
    
    try:
        # Initialize downloader
        downloader = KaggleToGCSDownloader(args.bucket_name, args.target_folder)
        
        # Install Kaggle API first
        downloader.install_kaggle_api()
        
        # Setup credentials BEFORE any kaggle import
        downloader.setup_kaggle_credentials(args.kaggle_username, args.kaggle_key)
        
        # Download and upload dataset
        downloader.download_and_extract_dataset()
        
        # Verify upload
        downloader.verify_upload()
        
        logger.info("✅ Dataset download and upload completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during dataset download: {str(e)}")
        raise

if __name__ == "__main__":
    main()