import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    model_name: str = "RETFound_cfp"
    pretrained_path: str = "models/RETFound_cfp_weights.pth"
    img_size: int = 224
    patch_size: int = 16
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_classes_rg: int = 5  # RG grades: 0, 1, 2, 3, 4 (PDR restored)
    num_classes_me: int = 3  # ME grades: 0, 1, 2
    dropout: float = 0.1
    drop_path: float = 0.1
    
    # Enhanced grading features
    enable_referable_classification: bool = True
    enable_confidence_estimation: bool = True
    enable_feature_localization: bool = True
    
    # Multi-modal support
    support_oct: bool = False
    support_octa: bool = False
    support_faf: bool = False

@dataclass
class DataConfig:
    dataset_path: str = "dataset"
    rg_path: str = "dataset/RG"
    me_path: str = "dataset/ME"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    batch_size: int = 32  # Increased from 16 for V100 optimization
    num_workers: int = 6   # Reduced from 10 - will auto-adjust based on system
    pin_memory: bool = True
    
    # DataLoader optimizations
    prefetch_factor: int = 8  # Enhanced prefetch for better GPU utilization
    persistent_workers: bool = True  # Keep workers alive between epochs
    drop_last: bool = True  # Drop incomplete batches for consistent performance
    
    # Augmentation parameters
    augment_prob: float = 0.8
    rotate_limit: int = 15
    brightness_limit: float = 0.1
    contrast_limit: float = 0.1
    
    # Class weights for handling imbalance
    rg_class_weights: List[float] = None
    me_class_weights: List[float] = None
    
    # New dataset type 1 fields
    num_classes: int = 2
    class_weights: bool = False
    medical_terms: Optional[str] = None

@dataclass
class TrainingConfig:
    num_epochs: int = 100
    learning_rate: float = 2e-4  # Scaled up for larger batch size (32 vs 16)
    weight_decay: float = 1e-5
    warmup_epochs: int = 10
    scheduler_type: str = "cosine_restarts"  # cosine, cosine_restarts, linear, polynomial
    
    # Performance optimizations
    compile_model: bool = True  # Use torch.compile() for PyTorch 2.0+
    enable_amp: bool = True  # Automatic Mixed Precision training
    gradient_accumulation_steps: int = 2  # Accumulate gradients for effective batch size 80
    gradient_checkpointing: bool = True  # Trade compute for memory efficiency
    max_grad_norm: float = 1.0  # Gradient clipping norm
    
    # Training efficiency
    validation_frequency: int = 2  # Validate every N epochs (reduced for speed)
    log_frequency: int = 100  # Log every N batches (reduced for performance)
    save_frequency: int = 10  # Save checkpoint every N epochs
    
    # Loss function weights
    rg_loss_weight: float = 1.0
    me_loss_weight: float = 1.0
    referable_loss_weight: float = 0.5
    confidence_loss_weight: float = 0.3
    feature_localization_loss_weight: float = 0.4
    
    # Early stopping
    patience: int = 15
    min_delta: float = 0.001
    
    # Progressive unfreezing
    freeze_backbone_epochs: int = 20
    unfreeze_rate: int = 5  # unfreeze layers every N epochs
    
    # New dataset type 1 fields
    focal_loss: bool = False
    medical_grade: bool = False

@dataclass
class LanguageConfig:
    model_name: str = "microsoft/DialoGPT-medium"  # For medical reasoning
    max_length: int = 256
    temperature: float = 0.7
    do_sample: bool = True
    num_return_sequences: int = 1
    pad_token_id: int = 50256
    
    # Medical vocabulary enhancement
    medical_terms_path: str = "data/medical_terms.json"
    enhanced_schema_version: str = "dr_v2.0_enhanced"
    
    # Enhanced reporting features
    generate_structured_reports: bool = True
    include_confidence_scores: bool = True
    include_evidence_localization: bool = True
    include_grading_rules: bool = True
    
@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    language: LanguageConfig = LanguageConfig()
    
    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    logs_dir: str = "logs"
    
    # Experiment tracking
    experiment_name: str = "diabetic_retinopathy_reasoning"
    use_wandb: bool = True
    wandb_project: str = "dr-classification"
    
    # Hardware
    device: str = "cuda"
    mixed_precision: bool = True
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True

def get_config() -> Config:
    return Config()