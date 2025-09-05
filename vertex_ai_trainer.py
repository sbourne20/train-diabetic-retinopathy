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
    env_loaded = load_dotenv()
    if env_loaded:
        print("✅ Environment variables loaded from .env file")
    else:
        print("⚠️  Warning: .env file not found or empty")
except ImportError:
    print("❌ Error: python-dotenv not found. Environment variables from .env file won't be loaded.")
    print("Install with: pip install python-dotenv")

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
            raise ValueError("dataset_type must be 0 (original dataset/ structure) or 1 (5-class DR dataset structure)")
        
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
            base_args = [
                "--dataset_path", f"gs://{self.config.bucket_name}/{self.dataset_name}",
                "--num_classes", "5"
            ]
            
            # Use command line parameters if provided, otherwise use defaults
            epochs = str(getattr(self, 'num_epochs', 50))
            batch_size = str(getattr(self, 'batch_size', 4))
            # CRITICAL FIX: Don't override None with default - preserve user's explicit setting
            learning_rate = str(self.learning_rate if hasattr(self, 'learning_rate') and self.learning_rate is not None else 3e-4)
            experiment_name = getattr(self, 'experiment_name', 'medsiglip_resume_optimized')
            
            base_args.extend([
                "--epochs", epochs,
                "--batch_size", batch_size,
                "--learning_rate", learning_rate,
                "--experiment_name", experiment_name
            ])
            
            # Add optional parameters if provided
            if getattr(self, 'freeze_backbone_epochs', None) is not None:
                base_args.extend(["--freeze_backbone_epochs", str(self.freeze_backbone_epochs)])
            
            if getattr(self, 'gradient_accumulation_steps', None) is not None:
                base_args.extend(["--gradient_accumulation_steps", str(self.gradient_accumulation_steps)])
                
            if getattr(self, 'warmup_epochs', None) is not None:
                base_args.extend(["--warmup_epochs", str(self.warmup_epochs)])
                
            if getattr(self, 'scheduler', None) is not None:
                base_args.extend(["--scheduler", self.scheduler])
                
            if getattr(self, 'validation_frequency', None) is not None:
                base_args.extend(["--validation_frequency", str(self.validation_frequency)])
                
            if getattr(self, 'patience', None) is not None:
                base_args.extend(["--patience", str(self.patience)])
                
            if getattr(self, 'min_delta', None) is not None:
                base_args.extend(["--min_delta", str(self.min_delta)])
                
            if getattr(self, 'weight_decay', None) is not None:
                base_args.extend(["--weight_decay", str(self.weight_decay)])
            
            # Add boolean flags
            if getattr(self, 'enable_class_weights', False):
                base_args.append("--class_weights")
                
            if getattr(self, 'enable_focal_loss', False):
                base_args.append("--focal_loss")
                
            if getattr(self, 'enable_medical_grade', False):
                base_args.append("--medical_grade")
            
            # Add LoRA parameters if enabled
            if getattr(self, 'use_lora', 'no') == 'yes':
                base_args.extend(["--use_lora", "yes"])
                if getattr(self, 'lora_r', None) is not None:
                    base_args.extend(["--lora_r", str(self.lora_r)])
                if getattr(self, 'lora_alpha', None) is not None:
                    base_args.extend(["--lora_alpha", str(self.lora_alpha)])
            
            # Add dropout parameter for medical-grade regularization
            if getattr(self, 'dropout', None) is not None:
                base_args.extend(["--dropout", str(self.dropout)])
            
            # Add advanced focal loss parameters
            if getattr(self, 'focal_loss_alpha', None) is not None:
                base_args.extend(["--focal_loss_alpha", str(self.focal_loss_alpha)])
            if getattr(self, 'focal_loss_gamma', None) is not None:
                base_args.extend(["--focal_loss_gamma", str(self.focal_loss_gamma)])
            
            # Add medical-grade class weighting parameters
            if getattr(self, 'class_weight_severe', None) is not None:
                base_args.extend(["--class_weight_severe", str(self.class_weight_severe)])
            if getattr(self, 'class_weight_pdr', None) is not None:
                base_args.extend(["--class_weight_pdr", str(self.class_weight_pdr)])
            
            # Add resume from checkpoint parameter
            if getattr(self, 'resume_from_checkpoint', None) is not None:
                base_args.extend(["--resume_from_checkpoint", str(self.resume_from_checkpoint)])
            
            args.extend(base_args)
        
        # Add medical terms file path
        medical_terms_file = "medical_terms.json" if self.dataset_type == 0 else "medical_terms_type1.json"
        args.extend(["--medical_terms", f"gs://{self.config.bucket_name}/{medical_terms_file}"])
        
        # Add debug mode arguments if enabled
        if getattr(self, 'debug_mode', False):
            args.extend(["--debug_mode"])
            if getattr(self, 'max_epochs', None):
                args.extend(["--max_epochs", str(self.max_epochs)])
            if getattr(self, 'eval_frequency', None):
                args.extend(["--eval_frequency", str(self.eval_frequency)])
            if getattr(self, 'checkpoint_frequency', None):
                args.extend(["--checkpoint_frequency", str(self.checkpoint_frequency)])
        
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
    
    def _get_training_env_vars(self) -> list:
        """Get environment variables required for training."""
        
        env_vars = []
        
        # HuggingFace token - REQUIRED for MedSigLIP-448 model
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            # Check if .env file exists and provide helpful debugging
            env_file_path = os.path.join(os.getcwd(), ".env")
            env_exists = os.path.exists(env_file_path)
            
            error_msg = (
                f"❌ HUGGINGFACE_TOKEN not found in environment variables.\n\n"
                f"🔍 DEBUG INFO:\n"
                f"- .env file exists: {env_exists}\n"
                f"- .env file path: {env_file_path}\n"
                f"- Current working directory: {os.getcwd()}\n"
                f"- Available env vars starting with 'HF' or 'HUGGINGFACE': "
                f"{[k for k in os.environ.keys() if k.startswith(('HF', 'HUGGINGFACE'))]}\n\n"
                f"🔧 SOLUTION:\n"
                f"1. Ensure .env file exists in project root\n"
                f"2. Add: HUGGINGFACE_TOKEN=hf_your_token_here\n"
                f"3. Get token from: https://huggingface.co/settings/tokens\n"
                f"4. Ensure access to: https://huggingface.co/google/medsiglip-448"
            )
            raise ValueError(error_msg)
        
        env_vars.append({"name": "HUGGINGFACE_TOKEN", "value": hf_token})
        
        # Optional: Add other environment variables if present
        optional_env_vars = [
            "WANDB_API_KEY",
            "OPENAI_API_KEY", 
            "GOOGLE_APPLICATION_CREDENTIALS"
        ]
        
        for env_var in optional_env_vars:
            value = os.getenv(env_var)
            if value:
                env_vars.append({"name": env_var, "value": value})
        
        # Add project configuration 
        env_vars.extend([
            {"name": "GOOGLE_CLOUD_PROJECT", "value": self.config.project_id},
            {"name": "GOOGLE_CLOUD_REGION", "value": self.config.region},
            {"name": "GCS_BUCKET", "value": self.config.bucket_name}
        ])
        
        logger.info(f"Setting {len(env_vars)} environment variables for training job")
        logger.info(f"Environment variables: {[var['name'] for var in env_vars]}")
        
        return env_vars
    
    def create_custom_training_job(self) -> str:
        """Create and submit custom training job."""
        
        logger.info("Creating custom training job...")
        
        # Prepare training package
        package_uri = self.prepare_training_package()
        
        # Get required environment variables for training
        env_vars = self._get_training_env_vars()
        
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
                        "args": self._get_training_args(),
                        "env": env_vars
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
        
        # Get required environment variables for training
        env_vars = self._get_training_env_vars()
        
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
                            "args": self._get_hyperparameter_tuning_args(),
                            "env": env_vars
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
                       help="Dataset type: 0=original dataset/ structure (RG/ME with 0,1 classes), 1=5-class DR structure (train/val/test with 0-4 classes) (MANDATORY)")
    parser.add_argument("--dataset_path", help="Local dataset path for upload (required for upload action)")
    parser.add_argument("--job_id", help="Job ID for monitoring")
    parser.add_argument("--project_id", help="Google Cloud Project ID")
    parser.add_argument("--bucket_name", help="GCS bucket name")
    parser.add_argument("--region", default="us-central1", help="Training region")
    
    # Debug and testing options
    parser.add_argument("--debug_mode", action="store_true", 
                       help="Enable debug mode for early testing (2 epochs, eval every epoch)")
    parser.add_argument("--max_epochs", type=int, help="Override maximum epochs for testing")
    parser.add_argument("--eval_frequency", type=int, help="Override evaluation frequency")
    parser.add_argument("--checkpoint_frequency", type=int, default=5, help="Save checkpoints every N epochs")
    
    # Training hyperparameters
    parser.add_argument("--num-epochs", type=int, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, help="Learning rate for training")
    parser.add_argument("--batch-size", type=int, help="Batch size for training")
    parser.add_argument("--freeze-backbone-epochs", type=int, help="Number of epochs to freeze backbone")
    parser.add_argument("--enable-focal-loss", action="store_true", help="Enable focal loss")
    parser.add_argument("--enable-medical-grade", action="store_true", help="Enable medical-grade validation")
    parser.add_argument("--enable-class-weights", action="store_true", help="Enable class weights")
    parser.add_argument("--gradient-accumulation-steps", type=int, help="Gradient accumulation steps")
    parser.add_argument("--warmup-epochs", type=int, help="Number of warmup epochs")
    parser.add_argument("--scheduler", help="Learning rate scheduler")
    parser.add_argument("--validation-frequency", type=int, help="Validation frequency")
    parser.add_argument("--patience", type=int, help="Early stopping patience")
    parser.add_argument("--min-delta", type=float, help="Minimum delta for early stopping")
    parser.add_argument("--weight-decay", type=float, help="Weight decay for optimizer")
    parser.add_argument("--experiment-name", help="Experiment name")
    
    # LoRA fine-tuning parameters
    parser.add_argument("--use-lora", type=str, default="no", choices=["yes", "no"],
                       help="Enable LoRA fine-tuning for memory efficiency")
    parser.add_argument("--lora-r", type=int, default=64,
                       help="LoRA rank parameter (default: 64 for maximum performance)")
    parser.add_argument("--lora-alpha", type=int, default=128,
                       help="LoRA alpha parameter (default: 128)")
    
    # Medical-grade regularization parameters
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate for medical-grade regularization")
    
    # Advanced focal loss parameters for medical-grade training
    parser.add_argument("--focal-loss-alpha", type=float, default=1.0,
                       help="Focal loss alpha parameter for class balancing")
    parser.add_argument("--focal-loss-gamma", type=float, default=2.0,
                       help="Focal loss gamma parameter for hard example focus")
    
    # Medical-grade class weighting parameters
    parser.add_argument("--class-weight-severe", type=float, default=3.0,
                       help="Class weight multiplier for Severe NPDR (Class 3)")
    parser.add_argument("--class-weight-pdr", type=float, default=2.5,
                       help="Class weight multiplier for PDR (Class 4)")
    
    # Resume from checkpoint parameter
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                       help="Resume training from specific checkpoint path")
    
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
    
    # Pass debug mode arguments to trainer
    trainer.debug_mode = getattr(args, 'debug_mode', False)
    trainer.max_epochs = getattr(args, 'max_epochs', None)
    trainer.eval_frequency = getattr(args, 'eval_frequency', None)
    trainer.checkpoint_frequency = getattr(args, 'checkpoint_frequency', 5)
    
    # Pass training hyperparameters to trainer
    trainer.num_epochs = getattr(args, 'num_epochs', None)
    trainer.learning_rate = getattr(args, 'learning_rate', None)
    trainer.batch_size = getattr(args, 'batch_size', None)
    trainer.freeze_backbone_epochs = getattr(args, 'freeze_backbone_epochs', None)
    trainer.enable_focal_loss = getattr(args, 'enable_focal_loss', False)
    trainer.enable_medical_grade = getattr(args, 'enable_medical_grade', False)
    trainer.enable_class_weights = getattr(args, 'enable_class_weights', False)
    trainer.gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', None)
    trainer.warmup_epochs = getattr(args, 'warmup_epochs', None)
    trainer.scheduler = getattr(args, 'scheduler', None)
    trainer.validation_frequency = getattr(args, 'validation_frequency', None)
    trainer.patience = getattr(args, 'patience', None)
    trainer.min_delta = getattr(args, 'min_delta', None)
    trainer.weight_decay = getattr(args, 'weight_decay', None)
    trainer.experiment_name = getattr(args, 'experiment_name', None)
    trainer.resume_from_checkpoint = getattr(args, 'resume_from_checkpoint', None)
    
    # Pass LoRA parameters to trainer
    trainer.use_lora = getattr(args, 'use_lora', 'no')
    trainer.lora_r = getattr(args, 'lora_r', 64)
    trainer.lora_alpha = getattr(args, 'lora_alpha', 128)
    trainer.dropout = getattr(args, 'dropout', 0.1)
    
    # Pass advanced medical-grade parameters to trainer
    trainer.focal_loss_alpha = getattr(args, 'focal_loss_alpha', 1.0)
    trainer.focal_loss_gamma = getattr(args, 'focal_loss_gamma', 2.0)
    trainer.class_weight_severe = getattr(args, 'class_weight_severe', 3.0)
    trainer.class_weight_pdr = getattr(args, 'class_weight_pdr', 2.5)
    
    # Execute action
    if args.action == "upload":
        trainer.upload_dataset(args.dataset_path)
    
    elif args.action == "train":
        # CRITICAL FIX: Parameters are now set BEFORE job creation
        print(f"🔧 DEBUG: Parameters set - LR: {trainer.learning_rate}, WD: {trainer.weight_decay}, GA: {trainer.gradient_accumulation_steps}")
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