"""
Vertex AI configuration for diabetic retinopathy training with TensorFlow
Optimized for NVIDIA Tesla P100 GPUs and 5-class DR grading system
"""

import os
from typing import Dict, Any

class VertexAIConfig:
    """Configuration for Vertex AI training jobs with TensorFlow support."""
    
    def __init__(self):
        # Project Configuration
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "your-project-id")
        self.region = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
        self.bucket_name = os.getenv("GCS_BUCKET", "your-bucket-name")
        
        # Training Job Configuration
        self.job_display_name = "diabetic-retinopathy-tensorflow-training"
        # Updated to TensorFlow container for Tesla P100 compatibility
        self.container_uri = "us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-13.py310:latest"
        
        # Machine Configuration - Cost-effective Tesla P100 setup
        self.machine_type = "n1-highmem-4"  # 4 vCPUs, 26GB RAM - Cost effective with Tesla P100
        self.accelerator_type = "NVIDIA_TESLA_P100"
        self.accelerator_count = 1
        
        # Disk Configuration
        self.boot_disk_type = os.getenv("BOOT_DISK_TYPE", "pd-ssd")
        self.boot_disk_size_gb = int(os.getenv("BOOT_DISK_SIZE_GB", "300"))  # Increased for TensorFlow
        
        # Storage paths
        self.gcs_dataset_path = f"gs://{self.bucket_name}/data"
        self.gcs_output_path = f"gs://{self.bucket_name}/outputs"
        self.gcs_model_path = f"gs://{self.bucket_name}/models"
        
        # Training configuration
        self.python_module = "tensorflow_dr_training"
        self.package_path = "./tensorflow_dr_package"
        
        # DR-specific configurations
        self.dr_classes = {
            0: "No_DR",
            1: "Mild_NPDR", 
            2: "Moderate_NPDR",
            3: "Severe_NPDR",
            4: "PDR"
        }
        
        # Model configuration for DR grading - Medical-grade RETFound for retinal analysis
        self.model_config = {
            "input_shape": (600, 600, 3),  # Matches dataset3_augmented_resized images
            "num_classes": 5,  # 5-class DR grading
            "backbone": "RETFound",  # Medical foundation model for retinal diseases
            "learning_rate": 0.0001,  # RETFound fine-tuning learning rate
            "batch_size": 4,  # Tesla P100 + 600x600 + RETFound memory optimized
            "epochs": 150,  # More epochs for medical-grade accuracy
            "early_stopping_patience": 25,  # More patience for convergence
            "reduce_lr_patience": 12,
            "weight_decay": 0.0001,  # L2 regularization for better generalization
            "dropout_rate": 0.2,  # Lower dropout for higher accuracy
            "label_smoothing": 0.05  # Reduce overconfidence
        }
        
        # Medical requirements - Upgraded to 95%+ for medical use
        self.medical_config = {
            "minimum_accuracy": 0.95,  # Medical-grade requirement
            "minimum_sensitivity": 0.93,  # Critical for early detection
            "minimum_specificity": 0.95,  # Critical to avoid false positives
            "minimum_precision": 0.93,  # Added precision requirement
            "minimum_f1_score": 0.93,  # Added F1 score requirement
            "minimum_auc": 0.95,  # Area under curve requirement
            "confidence_threshold": 0.8,  # Higher confidence for medical decisions
            "per_class_minimum_sensitivity": 0.90,  # Each class must meet this
            "per_class_minimum_specificity": 0.93   # Each class must meet this
        }
        
    def get_training_job_spec(self) -> Dict[str, Any]:
        """Get Vertex AI training job specification for TensorFlow DR training."""
        
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
                            "package_uris": [f"{self.gcs_model_path}/tensorflow_dr_package.tar.gz"],
                            "python_module": self.python_module,
                            "args": [
                                "--mode", "train",
                                "--dataset_path", f"gs://{self.bucket_name}/{{dataset_name}}",  # Will be replaced dynamically
                                "--output_dir", "/tmp/outputs",
                                "--epochs", str(self.model_config["epochs"]),
                                "--batch_size", str(self.model_config["batch_size"]),
                                "--learning_rate", str(self.model_config["learning_rate"]),
                                "--input_size", "600",
                                "--num_classes", "5",
                                "--backbone", self.model_config["backbone"],
                                "--medical_mode",  # Enable medical safety checks
                                "--tesla_p100",  # Optimize for Tesla P100
                                "--save_to_gcs", f"gs://{self.bucket_name}/outputs",
                                "--medical_terms", f"gs://{self.bucket_name}/medical_terms_type1.json",
                                "--comprehensive",  # Use comprehensive medical model
                                "--medical_grade"  # Enable 95%+ medical requirements
                            ]
                        }
                    }
                ],
                "base_output_directory": {
                    "output_uri_prefix": self.gcs_output_path
                }
            }
        }
    
    def get_training_job_spec_for_dataset(self, dataset_name: str, project_id: str, bucket_name: str, region: str) -> Dict[str, Any]:
        """Get Vertex AI training job specification with specific dataset and project settings."""
        
        # Update configuration with provided parameters
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.region = region
        
        # Update GCS paths with new bucket
        self.gcs_dataset_path = f"gs://{bucket_name}/data"
        self.gcs_output_path = f"gs://{bucket_name}/outputs"
        self.gcs_model_path = f"gs://{bucket_name}/models"
        
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
                            "package_uris": [f"gs://{bucket_name}/models/tensorflow_dr_package.tar.gz"],
                            "python_module": "medical_grade_training",  # Use our medical grade training
                            "args": [
                                "--dataset_path", f"gs://{bucket_name}/{dataset_name}",
                                "--output_dir", "/tmp/outputs",
                                "--epochs", str(self.model_config["epochs"]),
                                "--batch_size", str(self.model_config["batch_size"]),
                                "--learning_rate", str(self.model_config["learning_rate"]),
                                "--input_size", "600",
                                "--comprehensive",  # Use comprehensive medical model
                                "--medical_grade",  # Enable 95%+ medical requirements
                                "--save_to_gcs", f"gs://{bucket_name}/outputs",
                                "--medical_terms_path", f"gs://{bucket_name}/medical_terms_type1.json"
                            ]
                        }
                    }
                ],
                "base_output_directory": {
                    "output_uri_prefix": f"gs://{bucket_name}/outputs"
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