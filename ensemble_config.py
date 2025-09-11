#!/usr/bin/env python3
"""
Configuration Management for Multi-Architecture Ensemble Training

This module provides configuration classes and utilities for the ensemble 
diabetic retinopathy classification system, maintaining compatibility with
existing training infrastructure while adding ensemble-specific settings.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

@dataclass
class EnsembleModelConfig:
    """Configuration for individual models in the ensemble."""
    
    # Model architecture settings
    efficientnet_dropout: float = 0.3
    resnet_dropout: float = 0.3
    densenet_dropout: float = 0.3
    
    # Ensemble weighting (must sum to 1.0)
    ensemble_weights: List[float] = field(default_factory=lambda: [0.4, 0.35, 0.25])  # EfficientNet, ResNet, DenseNet
    
    # Pre-trained weights
    use_pretrained: bool = True
    
    # Input image size
    img_size: int = 224  # Optimal for CNN architectures
    
    # Number of output classes
    num_classes: int = 5
    
    # Model-specific settings
    efficientnet_variant: str = "b2"  # EfficientNet-B2
    resnet_variant: str = "50"        # ResNet-50
    densenet_variant: str = "121"     # DenseNet-121
    
    def validate_weights(self):
        """Validate ensemble weights sum to 1.0."""
        if abs(sum(self.ensemble_weights) - 1.0) > 1e-6:
            raise ValueError(f"Ensemble weights must sum to 1.0, got {sum(self.ensemble_weights)}")
        if len(self.ensemble_weights) != 3:
            raise ValueError(f"Must provide exactly 3 ensemble weights, got {len(self.ensemble_weights)}")

@dataclass 
class EnsembleDataConfig:
    """Configuration for data processing and augmentation."""
    
    # Dataset paths
    dataset_path: str = "./dataset3_augmented_resized"
    medical_terms_path: str = "data/medical_terms_type1.json"
    
    # Data splits
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Batch settings
    batch_size: int = 6  # Optimized for V100 memory
    num_workers: int = 4
    pin_memory: bool = True
    
    # Enhanced preprocessing
    enable_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple = (8, 8)
    
    # SMOTE class balancing
    enable_smote: bool = True
    smote_k_neighbors: int = 5
    smote_sampling_strategy: str = "auto"  # Balance all minority classes
    
    # Medical-grade augmentation
    enable_medical_augmentation: bool = True
    rotation_range: float = 15.0  # ±15 degrees (preserves retinal anatomy)
    horizontal_flip_prob: float = 0.5
    zoom_range: tuple = (0.95, 1.05)  # Subtle zoom maintaining field of view
    brightness_range: float = 0.1  # ±10% brightness variation
    contrast_range: float = 0.1    # ±10% contrast variation
    
    # Normalization (ImageNet statistics for pre-trained models)
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Class balancing
    enable_class_weights: bool = True
    class_weight_severe: float = 8.0  # Severe NPDR multiplier
    class_weight_pdr: float = 6.0     # PDR multiplier
    auto_class_weights: bool = True   # Compute weights from data distribution

@dataclass
class EnsembleTrainingConfig:
    """Configuration for ensemble training process."""
    
    # Training hyperparameters
    num_epochs: int = 100
    learning_rate: float = 1e-4  # Conservative LR for ensemble stability
    weight_decay: float = 1e-4
    
    # Individual model learning rates (can be different)
    efficientnet_lr: float = 1e-4
    resnet_lr: float = 1e-4  
    densenet_lr: float = 1e-4
    
    # Optimization settings
    optimizer: str = "adamw"
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    
    # Learning rate scheduling
    scheduler: str = "cosine"  # cosine, linear, plateau, none
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    
    # Gradient handling
    gradient_accumulation_steps: int = 2  # Effective batch size = batch_size * accum_steps
    max_grad_norm: float = 1.0  # Gradient clipping
    
    # Loss configuration
    enable_focal_loss: bool = True
    focal_loss_alpha: float = 2.0  # Reduced for ensemble stability
    focal_loss_gamma: float = 3.0  # Reduced for ensemble stability
    
    # Multi-task loss weights
    dr_loss_weight: float = 2.0      # Primary DR classification
    referable_loss_weight: float = 1.0
    sight_threatening_loss_weight: float = 1.0
    confidence_loss_weight: float = 0.5
    
    # Class weighting (duplicated from data config for training access)
    class_weight_severe: float = 8.0  # Severe NPDR multiplier
    class_weight_pdr: float = 6.0     # PDR multiplier
    
    # Ensemble-specific training
    train_individual_models: bool = True  # Train each model separately first
    train_ensemble_jointly: bool = True   # Then fine-tune ensemble together
    ensemble_training_epochs: int = 20    # Additional epochs for joint training
    
    # Validation and checkpointing
    validation_frequency: int = 1  # Validate every epoch
    checkpoint_frequency: int = 5  # Save checkpoint every 5 epochs
    patience: int = 15             # Early stopping patience
    min_delta: float = 0.001       # Minimum improvement threshold
    
    # Mixed precision training
    use_mixed_precision: bool = True
    
    # Regularization
    use_dropout_scheduling: bool = True  # Gradually reduce dropout
    initial_dropout: float = 0.5
    final_dropout: float = 0.3

@dataclass
class EnsembleSystemConfig:
    """System-level configuration for ensemble training."""
    
    # Device settings
    device: str = "cuda"
    device_ids: List[int] = field(default_factory=lambda: [0])  # GPU IDs
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Output and logging
    output_dir: str = "./ensemble_results"
    experiment_name: str = "efficientnetb2_resnet50_densenet121_ensemble"
    
    # Checkpoint management
    save_best_individual_models: bool = True
    save_best_ensemble_model: bool = True
    save_checkpoint_gcs: Optional[str] = None  # GCS bucket for backup
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None
    resume_individual_models: Optional[Dict[str, str]] = None
    
    # Logging and monitoring
    use_wandb: bool = False
    wandb_project: str = "dr-ensemble-training"
    wandb_entity: Optional[str] = None
    
    # Debug settings
    debug_mode: bool = False
    max_debug_epochs: int = 2
    
    # Medical validation
    enable_medical_validation: bool = True
    medical_accuracy_threshold: float = 0.90   # 90% minimum
    medical_sensitivity_threshold: float = 0.85  # 85% minimum
    medical_specificity_threshold: float = 0.90  # 90% minimum
    target_ensemble_accuracy: float = 0.9696   # Research target: 96.96%

class EnsembleConfig:
    """Main configuration class combining all ensemble settings."""
    
    def __init__(self,
                 model_config: Optional[EnsembleModelConfig] = None,
                 data_config: Optional[EnsembleDataConfig] = None,
                 training_config: Optional[EnsembleTrainingConfig] = None,
                 system_config: Optional[EnsembleSystemConfig] = None):
        
        self.model = model_config or EnsembleModelConfig()
        self.data = data_config or EnsembleDataConfig()
        self.training = training_config or EnsembleTrainingConfig()
        self.system = system_config or EnsembleSystemConfig()
        
        # Validate configuration
        self.validate()
    
    def validate(self):
        """Validate configuration consistency."""
        # Validate ensemble weights
        self.model.validate_weights()
        
        # Validate data splits
        if abs(self.data.train_split + self.data.val_split + self.data.test_split - 1.0) > 1e-6:
            raise ValueError("Data splits must sum to 1.0")
        
        # Validate paths
        if not os.path.exists(self.data.dataset_path):
            raise ValueError(f"Dataset path does not exist: {self.data.dataset_path}")
        
        # Validate training settings
        if self.training.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        
        if self.training.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
    
    def save(self, filepath: Union[str, Path]):
        """Save configuration to JSON file."""
        config_dict = {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'system': self.system.__dict__
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'EnsembleConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        model_config = EnsembleModelConfig(**config_dict['model'])
        data_config = EnsembleDataConfig(**config_dict['data'])
        training_config = EnsembleTrainingConfig(**config_dict['training'])
        system_config = EnsembleSystemConfig(**config_dict['system'])
        
        return cls(model_config, data_config, training_config, system_config)
    
    def update_from_args(self, args) -> 'EnsembleConfig':
        """Update configuration from command line arguments."""
        # Update data config
        if hasattr(args, 'dataset_path') and args.dataset_path:
            self.data.dataset_path = args.dataset_path
        if hasattr(args, 'batch_size') and args.batch_size:
            self.data.batch_size = args.batch_size
        if hasattr(args, 'medical_terms') and args.medical_terms:
            self.data.medical_terms_path = args.medical_terms
        
        # Update training config
        if hasattr(args, 'epochs') and args.epochs:
            self.training.num_epochs = args.epochs
        if hasattr(args, 'learning_rate') and args.learning_rate:
            self.training.learning_rate = args.learning_rate
        if hasattr(args, 'weight_decay') and args.weight_decay:
            self.training.weight_decay = args.weight_decay
        if hasattr(args, 'validation_frequency') and args.validation_frequency:
            self.training.validation_frequency = args.validation_frequency
        if hasattr(args, 'checkpoint_frequency') and args.checkpoint_frequency:
            self.training.checkpoint_frequency = args.checkpoint_frequency
        
        # Update focal loss settings
        if hasattr(args, 'enable_focal_loss') and args.enable_focal_loss:
            self.training.enable_focal_loss = True
        if hasattr(args, 'focal_loss_alpha') and args.focal_loss_alpha:
            self.training.focal_loss_alpha = args.focal_loss_alpha
        if hasattr(args, 'focal_loss_gamma') and args.focal_loss_gamma:
            self.training.focal_loss_gamma = args.focal_loss_gamma
        
        # Update data augmentation settings
        if hasattr(args, 'enable_clahe') and args.enable_clahe:
            self.data.enable_clahe = True
        if hasattr(args, 'enable_smote') and args.enable_smote:
            self.data.enable_smote = True
        if hasattr(args, 'enable_class_weights') and args.enable_class_weights:
            self.data.enable_class_weights = True
        
        # Update system config
        if hasattr(args, 'output_dir') and args.output_dir:
            self.system.output_dir = args.output_dir
        if hasattr(args, 'experiment_name') and args.experiment_name:
            self.system.experiment_name = args.experiment_name
        if hasattr(args, 'resume_from_checkpoint') and args.resume_from_checkpoint:
            self.system.resume_from_checkpoint = args.resume_from_checkpoint
        if hasattr(args, 'device') and args.device:
            self.system.device = args.device
        if hasattr(args, 'seed') and args.seed:
            self.system.seed = args.seed
        
        # Debug mode
        if hasattr(args, 'debug_mode') and args.debug_mode:
            self.system.debug_mode = True
            self.training.num_epochs = min(2, self.training.num_epochs)
        
        return self
    
    def get_individual_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get configuration dictionaries for individual models."""
        base_config = {
            'num_classes': self.model.num_classes,
            'img_size': self.model.img_size,
            'pretrained': self.model.use_pretrained
        }
        
        return {
            'efficientnetb2': {
                **base_config,
                'dropout': self.model.efficientnet_dropout,
                'learning_rate': self.training.efficientnet_lr
            },
            'resnet50': {
                **base_config,
                'dropout': self.model.resnet_dropout,
                'learning_rate': self.training.resnet_lr
            },
            'densenet121': {
                **base_config,
                'dropout': self.model.densenet_dropout,
                'learning_rate': self.training.densenet_lr
            }
        }
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"""EnsembleConfig(
  Models: EfficientNetB2 ({self.model.ensemble_weights[0]:.2f}) + ResNet50 ({self.model.ensemble_weights[1]:.2f}) + DenseNet121 ({self.model.ensemble_weights[2]:.2f})
  Dataset: {self.data.dataset_path}
  Epochs: {self.training.num_epochs}
  LR: {self.training.learning_rate}
  Batch Size: {self.data.batch_size}
  Device: {self.system.device}
  CLAHE: {self.data.enable_clahe}
  SMOTE: {self.data.enable_smote}
  Target Accuracy: {self.system.target_ensemble_accuracy:.2%}
)"""

def create_default_config() -> EnsembleConfig:
    """Create default ensemble configuration."""
    return EnsembleConfig()

def create_medical_grade_config() -> EnsembleConfig:
    """Create medical-grade ensemble configuration with optimized settings."""
    model_config = EnsembleModelConfig(
        ensemble_weights=[0.45, 0.35, 0.20],  # Higher weight for EfficientNet (best performer)
        efficientnet_dropout=0.2,  # Lower dropout for best model
        resnet_dropout=0.3,
        densenet_dropout=0.4       # Higher dropout for weaker model
    )
    
    data_config = EnsembleDataConfig(
        enable_clahe=True,
        enable_smote=True,
        enable_class_weights=True,
        enable_medical_augmentation=True,
        batch_size=8  # Slightly larger for better gradient estimates
    )
    
    training_config = EnsembleTrainingConfig(
        learning_rate=5e-5,  # Conservative for stability
        focal_loss_alpha=1.5,  # Gentler focal loss
        focal_loss_gamma=2.5,
        num_epochs=80,
        patience=20,  # More patience for ensemble convergence
        ensemble_training_epochs=30
    )
    
    system_config = EnsembleSystemConfig(
        enable_medical_validation=True,
        target_ensemble_accuracy=0.9696
    )
    
    return EnsembleConfig(model_config, data_config, training_config, system_config)

if __name__ == "__main__":
    # Test configuration creation and validation
    print("🧪 Testing Ensemble Configuration")
    
    # Test default config
    config = create_default_config()
    print("✅ Default configuration created")
    print(config)
    
    # Test medical-grade config
    medical_config = create_medical_grade_config()
    print("\n✅ Medical-grade configuration created")
    print(medical_config)
    
    # Test save/load
    config.save("test_ensemble_config.json")
    loaded_config = EnsembleConfig.load("test_ensemble_config.json")
    print("\n✅ Configuration save/load test passed")
    
    # Clean up
    os.remove("test_ensemble_config.json")
    
    print("\n✅ Configuration testing completed successfully!")