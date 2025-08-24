#!/usr/bin/env python3
"""
TensorFlow Vertex AI training script for diabetic retinopathy model
Compatible with existing vertex_ai_trainer.py pattern for GCS dataset loading
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

class TensorFlowVertexAITrainer:
    """Manages TensorFlow training jobs on Vertex AI with GCS dataset loading."""
    
    def __init__(self, config: VertexAIConfig, dataset_name: str, project_id: str, bucket_name: str, region: str):
        self.config = config
        self.dataset_name = dataset_name
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.region = region
        
        # Initialize Vertex AI
        aiplatform.init(
            project=project_id,
            location=region,
            staging_bucket=f"gs://{bucket_name}/staging"
        )
        
        # Initialize GCS client
        self.storage_client = storage.Client(project=project_id)
        self.bucket = self.storage_client.bucket(bucket_name)
    
    def prepare_tensorflow_training_package(self, source_dir: str = ".") -> str:
        """Create and upload TensorFlow training package to GCS."""
        
        logger.info("Preparing TensorFlow training package...")
        
        # Create temporary directory for package
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = os.path.join(temp_dir, "tensorflow_dr_package")
            os.makedirs(package_dir)
            
            # Files to include in TensorFlow training package
            files_to_include = [
                "tensorflow_dr_training.py",        # Main training script
                "comprehensive_medical_model.py",   # Medical model implementation
                "retinal_finding_detector.py",      # Finding detection
                "vertex_ai_config.py",             # Configuration
                ".env",                            # Environment variables (including HuggingFace token)
                "data/medical_terms_type1.json"    # Medical terminology
            ]
            
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
            
            # Create setup.py for TensorFlow package
            setup_py_content = '''
from setuptools import setup, find_packages

setup(
    name="tensorflow-diabetic-retinopathy-training",
    version="1.0.0",
    py_modules=[
        "tensorflow_dr_training", 
        "comprehensive_medical_model",
        "retinal_finding_detector",
        "vertex_ai_config"
    ],
    install_requires=[
        "tensorflow>=2.13.0",
        "pillow>=9.0.0",
        "numpy>=1.21.0,<2.0.0",  # Force NumPy 1.x for compatibility
        "pandas>=1.3.0,<2.0.0", 
        "scikit-learn>=1.0.0,<1.4.0",  # Compatible with NumPy 1.x
        "matplotlib>=3.5.0,<3.8.0",
        "opencv-python>=4.5.0,<4.9.0",
        "google-cloud-aiplatform>=1.35.0",
        "google-cloud-storage>=2.10.0",
        "python-dotenv>=0.19.0"  # Required for loading .env file
    ]
)
'''
            with open(os.path.join(package_dir, "setup.py"), "w") as f:
                f.write(setup_py_content)
            
            # Create tar.gz package
            package_path = os.path.join(temp_dir, "tensorflow_dr_package.tar.gz")
            with tarfile.open(package_path, "w:gz") as tar:
                tar.add(package_dir, arcname=".")
            
            # Upload to GCS
            blob_name = "models/tensorflow_dr_package.tar.gz"
            blob = self.bucket.blob(blob_name)
            blob.upload_from_filename(package_path)
            
            logger.info(f"TensorFlow training package uploaded to gs://{self.bucket_name}/{blob_name}")
            return f"gs://{self.bucket_name}/{blob_name}"
    
    def _get_tensorflow_training_args(self) -> list:
        """Get TensorFlow training arguments for medical-grade DR training."""
        
        # Medical-grade TensorFlow training arguments (matching tensorflow_dr_training.py)
        args = [
            "--dataset_path", f"gs://{self.bucket_name}/{self.dataset_name}",
            "--output_dir", "/tmp/outputs",
            "--epochs", "150",  # Medical-grade training
            "--batch_size", "4",  # Tesla P100 + 600x600 + RETFound optimized
            "--learning_rate", "0.0001",  # RETFound fine-tuning rate
            "--input_size", "600",
            "--num_classes", "5",  # 5-class DR grading
            "--medical_mode",  # Enable strict medical validation
            "--tesla_p100"  # Optimize for Tesla P100
        ]
        
        return args
    
    def create_tensorflow_training_job(self) -> str:
        """Create and submit TensorFlow custom training job."""
        
        logger.info("Creating TensorFlow medical-grade training job...")
        
        # Prepare training package
        package_uri = self.prepare_tensorflow_training_package()
        
        # Create job using CustomJob constructor
        job = aiplatform.CustomJob(
            display_name=f"tensorflow-medical-dr-{self.dataset_name}",
            worker_pool_specs=[
                {
                    "machine_spec": {
                        "machine_type": self.config.machine_type,  # n1-highmem-4
                        "accelerator_type": self.config.accelerator_type,  # NVIDIA_TESLA_P100
                        "accelerator_count": self.config.accelerator_count,  # 1
                    },
                    "disk_spec": {
                        "boot_disk_type": self.config.boot_disk_type,
                        "boot_disk_size_gb": self.config.boot_disk_size_gb,
                    },
                    "replica_count": 1,
                    "python_package_spec": {
                        "executor_image_uri": self.config.container_uri,  # TensorFlow container
                        "package_uris": [package_uri],
                        "python_module": "tensorflow_dr_training",  # Main training module
                        "args": self._get_tensorflow_training_args()
                    },
                }
            ],
            base_output_dir=f"gs://{self.bucket_name}/outputs",
        )
        
        # Submit job
        job.submit()
        
        logger.info(f"TensorFlow training job created: {job.resource_name}")
        return job.resource_name
    
    def create_tensorflow_hyperparameter_tuning_job(self) -> str:
        """Create TensorFlow hyperparameter tuning job."""
        
        logger.info("Creating TensorFlow hyperparameter tuning job...")
        
        # Prepare training package
        package_uri = self.prepare_tensorflow_training_package()
        
        # Base args for hyperparameter tuning (matching tensorflow_dr_training.py)
        base_args = [
            "--dataset_path", f"gs://{self.bucket_name}/{self.dataset_name}",
            "--output_dir", "/tmp/outputs",
            "--epochs", "100",  # Shorter for HP tuning
            "--input_size", "600",
            "--num_classes", "5",
            "--medical_mode",
            "--tesla_p100"
        ]
        
        # Create hyperparameter tuning job
        job = aiplatform.HyperparameterTuningJob(
            display_name=f"tensorflow-medical-dr-hp-tuning-{self.dataset_name}",
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
                            "python_module": "tensorflow_dr_training",
                            "args": base_args
                        },
                    }
                ]
            },
            metric_spec={
                "val_dr_accuracy": "maximize"  # Medical-grade metric
            },
            parameter_spec={
                "learning_rate": aiplatform.hyperparameter_tuning.DoubleParameterSpec(
                    min_value=1e-6, max_value=1e-3, scale="log"
                ),
                "batch_size": aiplatform.hyperparameter_tuning.DiscreteParameterSpec(
                    values=[8, 12, 16]  # Tesla P100 + n1-highmem-4 optimized
                ),
                "dropout_rate": aiplatform.hyperparameter_tuning.DoubleParameterSpec(
                    min_value=0.05, max_value=0.3
                )
            },
            max_trial_count=15,  # Reasonable for medical-grade tuning
            parallel_trial_count=3,  # Conservative for Tesla P100
            base_output_dir=f"gs://{self.bucket_name}/outputs"
        )
        
        # Submit job
        job.submit()
        
        logger.info(f"TensorFlow hyperparameter tuning job created: {job.resource_name}")
        return job.resource_name
    
    def monitor_job(self, job_resource_name: str):
        """Monitor training job progress."""
        
        logger.info(f"Monitoring job: {job_resource_name}")
        
        # Get job object
        job = aiplatform.CustomJob.get(job_resource_name)
        
        # Wait for completion
        job.wait()
        
        logger.info(f"Job completed with state: {job.state}")
        
        if job.state == aiplatform.gapic.JobState.JOB_STATE_SUCCEEDED:
            logger.info("TensorFlow medical-grade training completed successfully!")
            logger.info(f"Outputs available at: gs://{self.bucket_name}/outputs")
        else:
            logger.error(f"Training failed with state: {job.state}")
            if hasattr(job, 'error'):
                logger.error(f"Error: {job.error}")
    
    def download_results(self, local_output_dir: str = "tensorflow_vertex_ai_outputs"):
        """Download training results from GCS."""
        
        logger.info("Downloading TensorFlow training results...")
        
        os.makedirs(local_output_dir, exist_ok=True)
        
        # List and download output files
        blobs = self.storage_client.list_blobs(
            self.bucket_name,
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
        
        logger.info(f"TensorFlow results downloaded to {local_output_dir}")
    
    def validate_dataset_structure(self):
        """Validate GCS dataset structure for medical DR training."""
        
        logger.info(f"Validating dataset structure at gs://{self.bucket_name}/{self.dataset_name}")
        
        # Check for expected DR dataset structure (5-class)
        required_dirs = []
        for split in ['train', 'val', 'test']:
            for class_id in ['0', '1', '2', '3', '4']:
                required_dirs.append(f"{self.dataset_name}/{split}/{class_id}")
        
        missing_dirs = []
        for dir_path in required_dirs:
            # Check if any files exist in this directory path
            blobs = list(self.storage_client.list_blobs(
                self.bucket_name,
                prefix=f"{dir_path}/",
                max_results=1
            ))
            if not blobs:
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            logger.warning(f"Missing directories in dataset: {missing_dirs[:5]}...")  # Show first 5
        else:
            logger.info("Dataset structure validated successfully")
        
        # Check for medical terms file
        medical_terms_blob = self.bucket.blob("medical_terms_type1.json")
        if not medical_terms_blob.exists():
            logger.warning("medical_terms_type1.json not found in bucket root")
        else:
            logger.info("medical_terms_type1.json found")

def main():
    """Main function for TensorFlow Vertex AI training."""
    
    parser = argparse.ArgumentParser(description="TensorFlow Vertex AI Training Manager for Medical DR")
    parser.add_argument("--action", choices=["validate", "train", "tune", "monitor", "download"],
                       required=True, help="Action to perform")
    parser.add_argument("--dataset", required=True, 
                       help="Dataset folder name within GCS bucket (e.g., dataset3_augmented_resized)")
    parser.add_argument("--bucket_name", required=True, 
                       help="GCS bucket name (e.g., dr-data-2)")
    parser.add_argument("--project_id", required=True, 
                       help="Google Cloud Project ID (e.g., curalis-20250522)")
    parser.add_argument("--region", default="us-east1", 
                       help="Training region (e.g., us-east1)")
    parser.add_argument("--job_id", help="Job ID for monitoring")
    
    args = parser.parse_args()
    
    # Initialize config
    config = VertexAIConfig()
    
    # Initialize trainer with GCS parameters
    trainer = TensorFlowVertexAITrainer(
        config=config,
        dataset_name=args.dataset,
        project_id=args.project_id,
        bucket_name=args.bucket_name,
        region=args.region
    )
    
    # Execute action
    if args.action == "validate":
        trainer.validate_dataset_structure()
    
    elif args.action == "train":
        job_id = trainer.create_tensorflow_training_job()
        print(f"🏥 TensorFlow medical-grade training job created: {job_id}")
        print(f"📊 Dataset: gs://{args.bucket_name}/{args.dataset}")
        print(f"⚡ Hardware: n1-highmem-4 + Tesla P100")
        print(f"🎯 Target: 95%+ medical-grade accuracy")
        trainer.monitor_job(job_id)
    
    elif args.action == "tune":
        job_id = trainer.create_tensorflow_hyperparameter_tuning_job()
        print(f"🔧 TensorFlow hyperparameter tuning job created: {job_id}")
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