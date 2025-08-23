#!/usr/bin/env python3
"""
Vertex AI training script for diabetic retinopathy model
"""

import os
import argparse
import json
import tempfile
import shutil
from google.cloud import aiplatform
from google.cloud import storage
import tarfile
import logging

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not found. Environment variables from .env file won't be loaded.")

from vertex_ai_config import VertexAIConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VertexAITrainer:
    """Manages training jobs on Vertex AI."""
    
    def __init__(self, config: VertexAIConfig, dataset_name: str, dataset_type: int):
        self.config = config
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        
        # Validate dataset type
        if dataset_type not in [0, 1]:
            raise ValueError("dataset_type must be 0 (original dataset/) or 1 (dataset3_augmented_resized/)")
        
        # Initialize Vertex AI
        aiplatform.init(
            project=config.project_id,
            location=config.region,
            staging_bucket=f"gs://{config.bucket_name}/staging"
        )
        
        # Initialize GCS client
        self.storage_client = storage.Client(project=config.project_id)
        self.bucket = self.storage_client.bucket(config.bucket_name)
    
    def prepare_training_package(self, source_dir: str = ".") -> str:
        """Create and upload training package to GCS."""
        
        logger.info("Preparing training package...")
        
        # Create temporary directory for package
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = os.path.join(temp_dir, "training_package")
            os.makedirs(package_dir)
            
            # Files to include in training package
            files_to_include = [
                "main.py",
                "config.py", 
                "dataset.py",
                "models.py",
                "trainer.py",
                "evaluator.py",
                "utils.py",
                "requirements.txt"
            ]
            
            # Add appropriate medical terms file based on dataset type
            if self.dataset_type == 0:
                files_to_include.append("data/medical_terms.json")
            else:
                files_to_include.append("data/medical_terms_type1.json")
            
            # Copy files to package directory
            for file_path in files_to_include:
                if os.path.exists(os.path.join(source_dir, file_path)):
                    dest_path = os.path.join(package_dir, file_path)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.copy2(
                        os.path.join(source_dir, file_path),
                        dest_path
                    )
                else:
                    logger.warning(f"File not found: {file_path}")
            
            # Create setup.py for the package
            setup_py_content = '''
from setuptools import setup, find_packages

setup(
    name="diabetic-retinopathy-training",
    version="1.0.0",
    py_modules=["main", "config", "dataset", "models", "trainer", "evaluator", "utils"],
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "timm>=0.9.0",
        "pillow>=9.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "opencv-python>=4.5.0",
        "albumentations>=1.3.0",
        "wandb>=0.15.0",
        "tqdm>=4.64.0",
        "accelerate>=0.20.0",
        "datasets>=2.12.0",
        "evaluate>=0.4.0",
        "peft>=0.4.0",
        "bitsandbytes>=0.41.0",
        "google-cloud-aiplatform>=1.35.0",
        "google-cloud-storage>=2.10.0",
        "pydantic>=2.0.0,<3.0.0",
        "python-dotenv>=1.0.0"
    ]
)
'''
            with open(os.path.join(package_dir, "setup.py"), "w") as f:
                f.write(setup_py_content)
            
            # Create tar.gz package
            package_path = os.path.join(temp_dir, "training_package.tar.gz")
            with tarfile.open(package_path, "w:gz") as tar:
                tar.add(package_dir, arcname=".")
            
            # Upload to GCS
            blob_name = "models/training_package.tar.gz"
            blob = self.bucket.blob(blob_name)
            blob.upload_from_filename(package_path)
            
            logger.info(f"Training package uploaded to gs://{self.config.bucket_name}/{blob_name}")
            return f"gs://{self.config.bucket_name}/{blob_name}"
    
    def upload_dataset(self, local_dataset_path: str):
        """Upload dataset to GCS."""
        
        logger.info(f"Uploading dataset to GCS (type {self.dataset_type})...")
        
        # Validate dataset structure based on type
        if self.dataset_type == 0:
            self._validate_dataset_type0(local_dataset_path)
        else:
            self._validate_dataset_type1(local_dataset_path)
        
        for root, dirs, files in os.walk(local_dataset_path):
            for file in files:
                if file.endswith(('.tif', '.jpg', '.jpeg', '.png')):
                    local_path = os.path.join(root, file)
                    
                    # Create GCS path maintaining directory structure
                    relative_path = os.path.relpath(local_path, local_dataset_path)
                    gcs_path = f"{self.dataset_name}/{relative_path}"
                    
                    blob = self.bucket.blob(gcs_path)
                    if not blob.exists():
                        blob.upload_from_filename(local_path)
                        logger.info(f"Uploaded {relative_path}")
                    else:
                        logger.info(f"Skipped {relative_path} (already exists)")
    
    def _validate_dataset_type0(self, dataset_path: str):
        """Validate dataset structure for type 0 (RG/ME with 0,1 classes)."""
        required_dirs = ['RG/0', 'RG/1', 'ME/0', 'ME/1']
        for dir_path in required_dirs:
            full_path = os.path.join(dataset_path, dir_path)
            if not os.path.exists(full_path):
                raise ValueError(f"Dataset type 0 missing required directory: {dir_path}")
        logger.info("Dataset type 0 structure validated")
    
    def _validate_dataset_type1(self, dataset_path: str):
        """Validate dataset structure for type 1 (train/val/test with 0-4 classes)."""
        required_dirs = []
        for split in ['train', 'val', 'test']:
            for class_id in ['0', '1', '2', '3', '4']:
                required_dirs.append(f"{split}/{class_id}")
        
        for dir_path in required_dirs:
            full_path = os.path.join(dataset_path, dir_path)
            if not os.path.exists(full_path):
                raise ValueError(f"Dataset type 1 missing required directory: {dir_path}")
        logger.info("Dataset type 1 structure validated")
    
    def _get_training_args(self) -> list:
        """Get training arguments based on dataset type."""
        
        # Base arguments
        args = [
            "--mode", "train",
            "--output_dir", "/tmp/outputs",
            "--device", "cuda",
            "--no_wandb",
            "--save_to_gcs", f"gs://{self.config.bucket_name}/outputs"
        ]
        
        if self.dataset_type == 0:
            # Original dataset structure (RG/ME binary classification)
            args.extend([
                "--rg_path", f"gs://{self.config.bucket_name}/{self.dataset_name}/RG",
                "--me_path", f"gs://{self.config.bucket_name}/{self.dataset_name}/ME",
                "--epochs", "150",
                "--batch_size", "8",
                "--learning_rate", "5e-5",
                "--experiment_name", "enhanced_clinical_workflow_type0"
            ])
        else:
            # New dataset structure (5-class diabetic retinopathy classification)
            args.extend([
                "--dataset_path", f"gs://{self.config.bucket_name}/{self.dataset_name}",
                "--epochs", "200",  # More epochs for larger dataset
                "--batch_size", "16",  # Larger batch size for better convergence
                "--learning_rate", "3e-5",  # Lower learning rate for stability
                "--experiment_name", "medical_grade_dr_classification",
                "--num_classes", "5",
                "--class_weights",  # Enable class weighting for imbalanced data
                "--focal_loss",  # Use focal loss for better minority class performance
                "--medical_grade"  # Enable medical-grade validation metrics
            ])
        
        # Add medical terms file path
        medical_terms_file = "medical_terms.json" if self.dataset_type == 0 else "medical_terms_type1.json"
        args.extend(["--medical_terms", f"gs://{self.config.bucket_name}/{medical_terms_file}"])
        
        return args
    
    def _get_hyperparameter_tuning_args(self) -> list:
        """Get hyperparameter tuning arguments based on dataset type."""
        
        # Base arguments for hyperparameter tuning
        args = [
            "--mode", "train",
            "--output_dir", "/tmp/outputs",
            "--device", "cuda",
            "--no_wandb"
        ]
        
        if self.dataset_type == 0:
            # Original dataset structure
            args.extend([
                "--rg_path", f"gs://{self.config.bucket_name}/{self.dataset_name}/RG",
                "--me_path", f"gs://{self.config.bucket_name}/{self.dataset_name}/ME",
                "--epochs", "100",
                "--experiment_name", "hp_tuning_type0"
            ])
        else:
            # New dataset structure
            args.extend([
                "--dataset_path", f"gs://{self.config.bucket_name}/{self.dataset_name}",
                "--epochs", "120",
                "--experiment_name", "hp_tuning_medical_dr",
                "--num_classes", "5",
                "--class_weights",
                "--focal_loss",
                "--medical_grade"
            ])
        
        # Add medical terms file path
        medical_terms_file = "medical_terms.json" if self.dataset_type == 0 else "medical_terms_type1.json"
        args.extend(["--medical_terms", f"gs://{self.config.bucket_name}/{medical_terms_file}"])
        
        return args
    
    def create_custom_training_job(self) -> str:
        """Create and submit custom training job."""
        
        logger.info("Creating custom training job...")
        
        # Prepare training package
        package_uri = self.prepare_training_package()
        
        # Create job using CustomJob constructor with worker pool specs
        job = aiplatform.CustomJob(
            display_name=self.config.job_display_name,
            worker_pool_specs=[
                {
                    "machine_spec": {
                        "machine_type": self.config.machine_type,
                        "accelerator_type": self.config.accelerator_type,
                        "accelerator_count": self.config.accelerator_count,
                    },
                    "disk_spec": {
                        "boot_disk_type": self.config.boot_disk_type,
                        "boot_disk_size_gb": self.config.boot_disk_size_gb,
                    },
                    "replica_count": 1,
                    "python_package_spec": {
                        "executor_image_uri": self.config.container_uri,
                        "package_uris": [package_uri],
                        "python_module": "main",
                        "args": self._get_training_args()
                    },
                }
            ],
            base_output_dir=f"gs://{self.config.bucket_name}/outputs",
        )
        
        # Submit job
        job.submit()
        
        logger.info(f"Training job submitted: {job.resource_name}")
        return job.resource_name
    
    def create_hyperparameter_tuning_job(self) -> str:
        """Create hyperparameter tuning job."""
        
        logger.info("Creating hyperparameter tuning job...")
        
        # Prepare training package
        package_uri = self.prepare_training_package()
        
        # Create hyperparameter tuning job
        job = aiplatform.HyperparameterTuningJob(
            display_name=f"{self.config.job_display_name}-hp-tuning",
            custom_job_spec={
                "worker_pool_specs": [
                    {
                        "machine_spec": {
                            "machine_type": self.config.machine_type,
                            "accelerator_type": self.config.accelerator_type,
                            "accelerator_count": self.config.accelerator_count,
                        },
                        "disk_spec": {
                            "boot_disk_type": self.config.boot_disk_type,
                            "boot_disk_size_gb": self.config.boot_disk_size_gb,
                        },
                        "replica_count": 1,
                        "python_package_spec": {
                            "executor_image_uri": self.config.container_uri,
                            "package_uris": [package_uri],
                            "python_module": "main",
                            "args": self._get_hyperparameter_tuning_args()
                        },
                    }
                ]
            },
            metric_spec={
                "val_accuracy": "maximize"
            },
            parameter_spec={
                "learning_rate": aiplatform.hyperparameter_tuning.DoubleParameterSpec(
                    min_value=1e-5, max_value=1e-3, scale="log"
                ),
                "batch_size": aiplatform.hyperparameter_tuning.DiscreteParameterSpec(
                    values=[16, 32, 64]
                ),
                "dropout": aiplatform.hyperparameter_tuning.DoubleParameterSpec(
                    min_value=0.1, max_value=0.5
                )
            },
            max_trial_count=20,
            parallel_trial_count=4,
            base_output_dir=f"gs://{self.config.bucket_name}/outputs"
        )
        
        # Submit job
        job.submit()
        
        logger.info(f"Hyperparameter tuning job submitted: {job.resource_name}")
        return job.resource_name
    
    def monitor_job(self, job_resource_name: str):
        """Monitor training job progress."""
        
        logger.info(f"Monitoring job: {job_resource_name}")
        
        # Get job object using get method
        job = aiplatform.CustomJob.get(job_resource_name)
        
        # Wait for completion
        job.wait()
        
        logger.info(f"Job completed with state: {job.state}")
        
        if job.state == aiplatform.gapic.JobState.JOB_STATE_SUCCEEDED:
            logger.info("Training completed successfully!")
            logger.info(f"Outputs available at: {self.config.gcs_output_path}")
        else:
            logger.error(f"Training failed with state: {job.state}")
            if hasattr(job, 'error'):
                logger.error(f"Error: {job.error}")
    
    def download_results(self, local_output_dir: str = "vertex_ai_outputs"):
        """Download training results from GCS."""
        
        logger.info("Downloading training results...")
        
        os.makedirs(local_output_dir, exist_ok=True)
        
        # List and download output files
        blobs = self.storage_client.list_blobs(
            self.config.bucket_name,
            prefix="outputs/"
        )
        
        for blob in blobs:
            local_path = os.path.join(
                local_output_dir,
                os.path.relpath(blob.name, "outputs/")
            )
            
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded {blob.name}")
        
        logger.info(f"Results downloaded to {local_output_dir}")

def main():
    """Main function for Vertex AI training."""
    
    parser = argparse.ArgumentParser(description="Vertex AI Training Manager")
    parser.add_argument("--action", choices=["upload", "train", "tune", "monitor", "download"],
                       required=True, help="Action to perform")
    parser.add_argument("--dataset", required=True, 
                       help="Dataset folder name within GCS bucket (MANDATORY)")
    parser.add_argument("--dataset-type", type=int, choices=[0, 1], required=True,
                       help="Dataset type: 0=original dataset/ structure, 1=dataset3_augmented_resized/ structure (MANDATORY)")
    parser.add_argument("--dataset_path", help="Local dataset path for upload (required for upload action)")
    parser.add_argument("--job_id", help="Job ID for monitoring")
    parser.add_argument("--project_id", help="Google Cloud Project ID")
    parser.add_argument("--bucket_name", help="GCS bucket name")
    parser.add_argument("--region", default="us-central1", help="Training region")
    
    args = parser.parse_args()
    
    # Validate required parameters for upload action
    if args.action == "upload" and not args.dataset_path:
        parser.error("--dataset_path is required for upload action")
    
    # Convert dataset-type to dataset_type for internal use
    dataset_type = getattr(args, 'dataset_type')
    
    # Initialize config
    config = VertexAIConfig()
    if args.project_id:
        config.project_id = args.project_id
    if args.bucket_name:
        config.bucket_name = args.bucket_name
    if args.region:
        config.region = args.region
    
    # Initialize trainer with mandatory parameters
    trainer = VertexAITrainer(config, args.dataset, dataset_type)
    
    # Execute action
    if args.action == "upload":
        trainer.upload_dataset(args.dataset_path)
    
    elif args.action == "train":
        job_id = trainer.create_custom_training_job()
        print(f"Training job created: {job_id}")
        trainer.monitor_job(job_id)
    
    elif args.action == "tune":
        job_id = trainer.create_hyperparameter_tuning_job()
        print(f"Hyperparameter tuning job created: {job_id}")
        trainer.monitor_job(job_id)
    
    elif args.action == "monitor":
        if not args.job_id:
            print("Error: --job_id required for monitoring")
            return
        trainer.monitor_job(args.job_id)
    
    elif args.action == "download":
        trainer.download_results()

if __name__ == "__main__":
    main()