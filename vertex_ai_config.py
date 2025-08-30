"""
Vertex AI configuration for diabetic retinopathy training
"""

import os
from typing import Dict, Any

class VertexAIConfig:
    """Configuration for Vertex AI training jobs."""
    
    def __init__(self):
        # Project Configuration
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "your-project-id")
        self.region = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
        self.bucket_name = os.getenv("GCS_BUCKET", "your-bucket-name")
        
        # Training Job Configuration
        self.job_display_name = "diabetic-retinopathy-training"
        self.container_uri = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-0.py310:latest"
        
        # Machine Configuration for V100 16GB (memory-efficient model)
        self.machine_type = "n1-highmem-4"  # 4 vCPUs, 26GB RAM
        self.accelerator_type = "NVIDIA_TESLA_V100"
        self.accelerator_count = 1
        
        # Disk Configuration
        self.boot_disk_type = os.getenv("BOOT_DISK_TYPE", "pd-ssd")
        self.boot_disk_size_gb = int(os.getenv("BOOT_DISK_SIZE_GB", "200"))
        
        # Storage paths
        self.gcs_dataset_path = f"gs://{self.bucket_name}/data"
        self.gcs_output_path = f"gs://{self.bucket_name}/outputs"
        self.gcs_model_path = f"gs://{self.bucket_name}/models"
        
        # Training configuration
        self.python_module = "main"
        self.package_path = "./training_package"
        
    def get_training_job_spec(self) -> Dict[str, Any]:
        """Get Vertex AI training job specification."""
        
        return {
            "display_name": self.job_display_name,
            "job_spec": {
                "worker_pool_specs": [
                    {
                        "machine_spec": {
                            "machine_type": self.machine_type,
                            "accelerator_type": self.accelerator_type,
                            "accelerator_count": self.accelerator_count
                        },
                        "disk_spec": {
                            "boot_disk_type": self.boot_disk_type,
                            "boot_disk_size_gb": self.boot_disk_size_gb
                        },
                        "replica_count": 1,
                        "python_package_spec": {
                            "executor_image_uri": self.container_uri,
                            "package_uris": [f"{self.gcs_model_path}/training_package.tar.gz"],
                            "python_module": self.python_module,
                            "args": [
                                "--mode", "train",
                                "--rg_path", f"{self.gcs_dataset_path}/RG",
                                "--me_path", f"{self.gcs_dataset_path}/ME",
                                "--output_dir", "/gcs/outputs",
                                "--epochs", "200",
                                "--batch_size", "8",  # Optimized batch size
                                "--learning_rate", "3e-4",  # Increased learning rate
                                "--device", "cuda",
                                "--no_wandb"  # Disable wandb for cloud training
                            ]
                        }
                    }
                ],
                "base_output_directory": {
                    "output_uri_prefix": self.gcs_output_path
                }
            }
        }
    
    def get_hyperparameter_tuning_spec(self) -> Dict[str, Any]:
        """Get hyperparameter tuning job specification."""
        
        return {
            "display_name": f"{self.job_display_name}-hp-tuning",
            "max_trial_count": 20,
            "parallel_trial_count": 4,
            "study_spec": {
                "metrics": [
                    {
                        "metric_id": "val_accuracy",
                        "goal": "MAXIMIZE"
                    }
                ],
                "parameters": [
                    {
                        "parameter_id": "learning_rate",
                        "double_value_spec": {
                            "min_value": 1e-5,
                            "max_value": 1e-3
                        },
                        "scale_type": "UNIT_LOG_SCALE"
                    },
                    {
                        "parameter_id": "batch_size",
                        "discrete_value_spec": {
                            "values": [16, 32, 64]
                        }
                    },
                    {
                        "parameter_id": "dropout",
                        "double_value_spec": {
                            "min_value": 0.1,
                            "max_value": 0.5
                        }
                    }
                ]
            },
            "trial_job_spec": {
                "worker_pool_specs": [
                    {
                        "machine_spec": {
                            "machine_type": self.machine_type,
                            "accelerator_type": self.accelerator_type,
                            "accelerator_count": self.accelerator_count
                        },
                        "disk_spec": {
                            "boot_disk_type": self.boot_disk_type,
                            "boot_disk_size_gb": self.boot_disk_size_gb
                        },
                        "replica_count": 1,
                        "python_package_spec": {
                            "executor_image_uri": self.container_uri,
                            "package_uris": [f"{self.gcs_model_path}/training_package.tar.gz"],
                            "python_module": self.python_module,
                            "args": [
                                "--mode", "train",
                                "--rg_path", f"{self.gcs_dataset_path}/RG",
                                "--me_path", f"{self.gcs_dataset_path}/ME",
                                "--output_dir", "/gcs/outputs",
                                "--epochs", "100",
                                "--learning_rate", "{{trial.parameters.learning_rate}}",
                                "--batch_size", "{{trial.parameters.batch_size}}",
                                "--dropout", "{{trial.parameters.dropout}}",
                                "--device", "cuda",
                                "--no_wandb"
                            ]
                        }
                    }
                ],
                "base_output_directory": {
                    "output_uri_prefix": self.gcs_output_path
                }
            }
        }