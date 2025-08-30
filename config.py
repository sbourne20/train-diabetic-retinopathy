from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    model_name: str = "MedSigLIP_448"
    pretrained_path: str = "google/medsiglip-448"  # HuggingFace model path
    img_size: int = 448  # MedSigLIP-448 requires 448x448 images
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
    
    # LoRA fine-tuning configuration
    use_lora: bool = False
    lora_r: int = 64  # Maximum performance configuration
    lora_alpha: int = 128  # 2x rank for optimal scaling

@dataclass
class DataConfig:
    dataset_path: str = "dataset"
    rg_path: str = "dataset/RG"
    me_path: str = "dataset/ME"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    batch_size: int = 6   # Reduced for MedSigLIP-448 + premium dataset fine-tuning
    num_workers: int = 4   # Optimized for faster data loading
    pin_memory: bool = True
    
    # DataLoader optimizations (aggressive for speed)
    prefetch_factor: int = 16  # Doubled prefetch for maximum GPU utilization
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
    
    # New dataset type 1 fields - KEY FIXES
    num_classes: int = 5  # 5-class DR classification
    class_weights: bool = True  # Enable class weighting for imbalance
    medical_terms: Optional[str] = None

@dataclass
class TrainingConfig:
    num_epochs: int = 200  # Extended for 90%+ accuracy with premium dataset
    learning_rate: float = 1e-4  # Optimized for EyePACS/APTOS convergence
    weight_decay: float = 5e-6  # Reduced for better generalization
    warmup_epochs: int = 15  # Extended warmup for stable convergence
    scheduler: str = "cosine_restarts"  # cosine, cosine_restarts, linear, polynomial
    
    # Performance optimizations
    compile_model: bool = False  # Disable torch.compile() to avoid CUDA errors
    enable_amp: bool = True  # Automatic Mixed Precision training
    gradient_accumulation_steps: int = 2  # Reduced for more frequent updates
    gradient_checkpointing: bool = True  # Trade compute for memory efficiency
    max_grad_norm: float = 1.0  # Gradient clipping norm
    
    # Training efficiency (optimized for 90%+ accuracy)
    validation_frequency: int = 5  # Validate every N epochs (frequent monitoring)
    log_frequency: int = 50  # More frequent logging for fine-tuning
    save_frequency: int = 5  # Save checkpoints more frequently
    
    # Loss function weights
    rg_loss_weight: float = 1.0
    me_loss_weight: float = 1.0
    referable_loss_weight: float = 0.5
    confidence_loss_weight: float = 0.3
    feature_localization_loss_weight: float = 0.4
    
    # Early stopping (90%+ accuracy target with premium dataset)
    patience: int = 25  # Extended patience for 90%+ convergence
    min_delta: float = 0.0005  # Tighter threshold for premium dataset precision
    
    # Progressive unfreezing - KEY FIX
    freeze_backbone_epochs: int = 0  # No freezing - start fine-tuning immediately
    unfreeze_rate: int = 5  # unfreeze layers every N epochs
    
    # New dataset type 1 fields - KEY FIXES
    focal_loss: bool = True  # Enable focal loss for class imbalance
    medical_grade: bool = True  # Enable medical-grade validation

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