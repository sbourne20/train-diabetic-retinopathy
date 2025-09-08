#!/usr/bin/env python3
"""
Local V100 Training Script for Diabetic Retinopathy Classification
Mirrors vertex_ai_trainer.py functionality but runs directly on local V100 GPU
"""

import os
import argparse
import json
import torch
from datetime import datetime

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ Warning: wandb not available. Logging disabled.")

from config import get_config
from dataset import create_data_splits, create_dataloaders, compute_dataset_class_weights, save_data_splits
from models import DiabeticRetinopathyModel
from trainer import MedicalGradeDRTrainer
from evaluator import ModelEvaluator
from utils import set_seed, create_directories

def parse_args():
    """Parse command line arguments - mirrors vertex_ai_trainer.py parameters."""
    
    parser = argparse.ArgumentParser(description='Local V100 Diabetic Retinopathy Training')
    
    # Mode selection
    parser.add_argument('--mode', choices=['train', 'evaluate', 'inference'], 
                       default='train', help='Mode to run the script')
    
    # Dataset configuration
    parser.add_argument('--dataset_path', default='./dataset5',
                       help='Path to local dataset directory (train/val/test structure)')
    parser.add_argument('--num_classes', type=int, default=5,
                       help='Number of classes (5 for diabetic retinopathy)')
    
    # Model configuration (MedSigLIP-448 only)
    parser.add_argument('--pretrained_path', default='google/medsiglip-448',
                       help='Path to MedSigLIP-448 model (HuggingFace)')
    parser.add_argument('--img_size', type=int, default=448, 
                       help='Image size for MedSigLIP-448')
    
    # LoRA fine-tuning parameters (exact match with Vertex AI)
    parser.add_argument('--use_lora', choices=['yes', 'no'], default='yes',
                       help='Enable LoRA fine-tuning for memory efficiency')
    parser.add_argument('--lora_r', type=int, default=16,
                       help='LoRA rank parameter (16 for checkpoint compatibility)')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA alpha parameter (32 proven effective)')
    
    # Training hyperparameters (exact match with medical_grade_lora_antioverfitting.sh)
    parser.add_argument('--epochs', type=int, default=60, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=6, 
                       help='Batch size (6 for memory efficiency)')
    parser.add_argument('--learning_rate', type=float, default=2e-5, 
                       help='Learning rate (2e-5 exact from 81.76% success)')
    parser.add_argument('--freeze_backbone_epochs', type=int, default=0, 
                       help='Number of epochs to freeze backbone')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, 
                       help='Gradient accumulation steps')
    parser.add_argument('--warmup_epochs', type=int, default=30, 
                       help='Number of warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-5, 
                       help='Weight decay for optimizer')
    parser.add_argument('--dropout', type=float, default=0.4, 
                       help='Dropout rate for regularization')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, 
                       help='Maximum gradient norm for clipping')
    
    # Medical-grade loss and weighting (exact parameters)
    parser.add_argument('--enable_focal_loss', action='store_true',
                       help='Enable focal loss for imbalanced data')
    parser.add_argument('--focal_loss_alpha', type=float, default=4.0,
                       help='Focal loss alpha parameter (4.0 original)')
    parser.add_argument('--focal_loss_gamma', type=float, default=6.0,
                       help='Focal loss gamma parameter (6.0 original)')
    parser.add_argument('--enable_class_weights', action='store_true',
                       help='Enable class weights')
    parser.add_argument('--class_weight_severe', type=float, default=8.0,
                       help='Class weight for Severe NPDR (8.0 original)')
    parser.add_argument('--class_weight_pdr', type=float, default=6.0,
                       help='Class weight for PDR (6.0 original)')
    parser.add_argument('--enable_medical_grade', action='store_true',
                       help='Enable medical-grade validation')
    
    # Scheduler (exact match - none for fixed LR)
    parser.add_argument('--scheduler', 
                       choices=["none", "polynomial", "linear", "validation_plateau", "cosine_with_restarts"],
                       default='none', 
                       help='Learning rate scheduler (none for fixed LR)')
    
    # Validation and early stopping
    parser.add_argument('--validation_frequency', type=int, default=1, 
                       help='Validation frequency (every epoch)')
    parser.add_argument('--patience', type=int, default=15, 
                       help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=0.001, 
                       help='Minimum delta for early stopping')
    
    # Checkpointing (local only)
    parser.add_argument('--checkpoint_frequency', type=int, default=2, 
                       help='Save checkpoints every N epochs')
    parser.add_argument('--resume_from_checkpoint', default=None,
                       help='Resume from local checkpoint path')
    
    # Experiment settings
    parser.add_argument('--experiment_name', 
                       default='medsiglip_lora_local_v100_training',
                       help='Experiment name for logging')
    parser.add_argument('--output_dir', default='local_outputs', 
                       help='Output directory')
    parser.add_argument('--no_wandb', action='store_true', 
                       help='Disable wandb logging')
    
    # Device configuration
    parser.add_argument('--device', default='cuda', help='Device to use (cuda for V100)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Medical terms file
    parser.add_argument('--medical_terms', default='data/medical_terms_type1.json',
                       help='Path to medical terms JSON file')
    
    # Debug options
    parser.add_argument('--debug_mode', action='store_true',
                       help='Enable debug mode (2 epochs for testing)')
    parser.add_argument('--max_epochs', type=int, default=None,
                       help='Override maximum epochs for testing')
    
    return parser.parse_args()

def setup_local_experiment(args):
    """Setup experiment configuration for local V100 training."""
    
    config = get_config()
    
    # Update config with local training arguments
    config.data.dataset_path = args.dataset_path
    config.data.num_classes = args.num_classes
    config.data.batch_size = args.batch_size
    
    # Model configuration (MedSigLIP-448)
    config.model.pretrained_path = args.pretrained_path
    config.model.img_size = args.img_size
    config.model.num_classes = args.num_classes
    
    # LoRA configuration (exact match with Vertex AI)
    config.model.use_lora = (args.use_lora == 'yes')
    config.model.lora_r = args.lora_r
    config.model.lora_alpha = args.lora_alpha
    config.model.dropout = args.dropout
    
    # Training configuration (medical-grade parameters)
    config.training.num_epochs = args.epochs
    config.training.learning_rate = args.learning_rate
    config.training.freeze_backbone_epochs = args.freeze_backbone_epochs
    config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.training.warmup_epochs = args.warmup_epochs
    config.training.weight_decay = args.weight_decay
    config.training.max_grad_norm = args.max_grad_norm
    
    # Medical-grade loss configuration
    config.data.class_weights = args.enable_class_weights
    config.training.focal_loss = args.enable_focal_loss
    config.training.focal_loss_alpha = args.focal_loss_alpha
    config.training.focal_loss_gamma = args.focal_loss_gamma
    config.training.class_weight_severe = args.class_weight_severe
    config.training.class_weight_pdr = args.class_weight_pdr
    config.training.medical_grade = args.enable_medical_grade
    
    # Scheduler configuration
    config.training.scheduler = args.scheduler
    config.training.validation_frequency = args.validation_frequency
    config.training.patience = args.patience
    config.training.min_delta = args.min_delta
    config.training.checkpoint_frequency = args.checkpoint_frequency
    
    # Local settings
    config.output_dir = args.output_dir
    config.device = args.device
    config.seed = args.seed
    config.use_wandb = not args.no_wandb and WANDB_AVAILABLE
    
    # Medical terms
    if args.medical_terms and os.path.exists(args.medical_terms):
        config.data.medical_terms = args.medical_terms
        config.language.medical_terms_path = args.medical_terms
    
    # Debug mode
    if args.debug_mode:
        print("🐛 DEBUG MODE: 2 epochs for V100 testing")
        config.training.num_epochs = args.max_epochs or 2
        config.training.validation_frequency = 1
        config.training.checkpoint_frequency = 1
    elif args.max_epochs:
        config.training.num_epochs = args.max_epochs
    
    # Resume from checkpoint
    config.training.resume_from_checkpoint = args.resume_from_checkpoint
    if args.resume_from_checkpoint:
        print(f"🔄 Resume from checkpoint: {args.resume_from_checkpoint}")
    
    # Set experiment name
    config.experiment_name = args.experiment_name
    
    # Set reproducibility
    set_seed(config.seed)
    
    # Create directories
    create_directories(config)
    
    print(f"\n🎯 LOCAL V100 TRAINING CONFIGURATION:")
    print(f"   Dataset: {config.data.dataset_path}")
    print(f"   Model: {config.model.pretrained_path}")
    print(f"   LoRA: r={config.model.lora_r}, alpha={config.model.lora_alpha}")
    print(f"   LR: {config.training.learning_rate}, Epochs: {config.training.num_epochs}")
    print(f"   Batch Size: {config.data.batch_size}, Grad Accum: {config.training.gradient_accumulation_steps}")
    print(f"   Focal Loss: α={config.training.focal_loss_alpha}, γ={config.training.focal_loss_gamma}")
    print(f"   Class Weights: Severe={config.training.class_weight_severe}, PDR={config.training.class_weight_pdr}")
    print(f"   Device: {config.device}")
    
    return config

def prepare_local_data(config, args):
    """Prepare local dataset5 for training."""
    
    print(f"📁 Preparing dataset from: {config.data.dataset_path}")
    
    # Validate dataset structure
    required_dirs = ['train', 'val', 'test']
    for split in required_dirs:
        split_path = os.path.join(config.data.dataset_path, split)
        if not os.path.exists(split_path):
            raise ValueError(f"Missing required directory: {split_path}")
        
        # Check classes 0-4
        for class_id in range(5):
            class_path = os.path.join(split_path, str(class_id))
            if not os.path.exists(class_path):
                raise ValueError(f"Missing class directory: {class_path}")
    
    print("✅ Dataset5 structure validated")
    
    # Load dataset using type 1 structure
    from dataset import create_data_splits_type1
    train_data, val_data, test_data = create_data_splits_type1(
        config.data.dataset_path,
        num_classes=config.data.num_classes,
        seed=config.seed,
        max_train_samples=None  # Use full dataset5 (29k samples)
    )
    
    print(f"📊 Dataset splits created:")
    print(f"   Training: {len(train_data)} samples")
    print(f"   Validation: {len(val_data)} samples") 
    print(f"   Test: {len(test_data)} samples")
    
    # Save splits for reproducibility
    save_data_splits(train_data, val_data, test_data)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, config
    )
    
    # Compute medical-grade class weights
    if config.data.class_weights:
        from dataset import compute_dr_class_weights
        dr_weights = compute_dr_class_weights(
            train_data,
            config.data.num_classes,
            severe_multiplier=config.training.class_weight_severe,
            pdr_multiplier=config.training.class_weight_pdr
        )
        print(f"🏥 Medical-grade class weights computed: {dr_weights}")
    else:
        dr_weights = None
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'dr_weights': dr_weights,
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data
    }

def train_local_model(config, data_dict, args):
    """Train model locally on V100."""
    
    print("\n🚀 Initializing MedSigLIP-448 model...")
    
    # Check V100 memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
    
    model = DiabeticRetinopathyModel(
        img_size=config.model.img_size,
        num_classes=config.model.num_classes,
        dropout=config.model.dropout,
        enable_confidence=config.model.enable_confidence_estimation,
        use_lora=config.model.use_lora,
        lora_r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha
    )
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🎯 Model parameters: {trainable_params:,} trainable / {total_params:,} total")
    
    if config.model.use_lora:
        print(f"🔧 LoRA configuration: r={config.model.lora_r}, α={config.model.lora_alpha}")
        print(f"💾 Memory efficient: ~{trainable_params/1e6:.1f}M trainable params")
    
    # Initialize medical-grade trainer
    trainer = MedicalGradeDRTrainer(
        model=model,
        train_loader=data_dict['train_loader'],
        val_loader=data_dict['val_loader'],
        device=torch.device(config.device),
        learning_rate=config.training.learning_rate,
        num_epochs=config.training.num_epochs,
        patience=config.training.patience,
        use_mixed_precision=True,  # Enable for V100 efficiency
        class_weights=data_dict['dr_weights'],
        validation_frequency=config.training.validation_frequency,
        checkpoint_frequency=config.training.checkpoint_frequency,
        gcs_bucket=None,  # Disable GCS for local training
        resume_from_checkpoint=config.training.resume_from_checkpoint,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        # Medical-grade parameters (exact match)
        enable_focal_loss=config.training.focal_loss,
        focal_loss_alpha=config.training.focal_loss_alpha,
        focal_loss_gamma=config.training.focal_loss_gamma,
        weight_decay=config.training.weight_decay,
        scheduler=config.training.scheduler,
        max_grad_norm=config.training.max_grad_norm
    )
    
    # Load checkpoint if provided
    if config.training.resume_from_checkpoint and os.path.exists(config.training.resume_from_checkpoint):
        print(f"📥 Loading checkpoint: {config.training.resume_from_checkpoint}")
        start_epoch = trainer.load_checkpoint(config.training.resume_from_checkpoint)
        print(f"🔄 Resumed from epoch {start_epoch}")
    
    # Train model
    print("\n🎯 Starting local V100 training...")
    print("=" * 60)
    training_history = trainer.train()
    
    # Save final model locally
    final_model_path = os.path.join(config.checkpoint_dir, "final_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'training_history': training_history
    }, final_model_path)
    
    print(f"\n💾 Final model saved: {final_model_path}")
    
    return model, training_history

def evaluate_local_model(config, data_dict, model=None, args=None):
    """Evaluate trained model."""
    
    if model is None:
        print("📥 Loading model for evaluation...")
        model = DiabeticRetinopathyModel(
            img_size=config.model.img_size,
            num_classes=config.model.num_classes,
            dropout=config.model.dropout,
            enable_confidence=config.model.enable_confidence_estimation,
            use_lora=config.model.use_lora,
            lora_r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha
        )
        
        # Load best model
        best_model_path = os.path.join(config.checkpoint_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"📥 Loaded model from: {best_model_path}")
        else:
            print("⚠️ No trained model found. Using random initialization.")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model=model,
        test_loader=data_dict['test_loader'],
        config=config,
        device=config.device
    )
    
    # Run evaluation
    print("🏥 Running medical-grade evaluation...")
    evaluation_results = evaluator.evaluate_model(save_results=True)
    
    # Print summary
    print("\n" + "="*50)
    print("🏥 MEDICAL-GRADE EVALUATION RESULTS")
    print("="*50)
    
    if 'overall_summary' in evaluation_results:
        summary = evaluation_results['overall_summary']
        if 'overall_accuracy' in summary:
            print(f"🎯 Overall Accuracy: {summary['overall_accuracy'].get('combined', 'N/A'):.4f}")
        
        if 'clinical_relevance' in summary:
            clinical = summary['clinical_relevance']
            print(f"🏥 Clinical Utility: {clinical.get('overall_clinical_utility', 'N/A')}")
    
    return evaluation_results

def main():
    """Main function for local V100 training."""
    
    print("🎯 LOCAL V100 DIABETIC RETINOPATHY TRAINING")
    print("=" * 50)
    
    args = parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This script requires GPU.")
        return
    
    if args.device == 'cuda':
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Setup experiment
    config = setup_local_experiment(args)
    
    # Validate dataset path
    if not os.path.exists(args.dataset_path):
        print(f"❌ Dataset not found: {args.dataset_path}")
        print("Please ensure dataset5 exists in the current directory")
        return
    
    print(f"\n🎯 Experiment: {config.experiment_name}")
    print(f"📁 Dataset: {config.data.dataset_path}")
    print(f"🎮 Device: {config.device}")
    print(f"🔢 Random seed: {config.seed}")
    
    # Save config
    config_path = os.path.join(config.output_dir, "local_config.json")
    with open(config_path, 'w') as f:
        config_dict = {
            'model': config.model.__dict__,
            'data': config.data.__dict__,
            'training': config.training.__dict__,
            'experiment_name': config.experiment_name,
            'output_dir': config.output_dir,
            'device': config.device,
            'seed': config.seed
        }
        json.dump(config_dict, f, indent=2)
    
    # Prepare data
    try:
        data_dict = prepare_local_data(config, args)
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return
    
    # Execute based on mode
    if args.mode == 'train':
        try:
            # Train model
            model, training_history = train_local_model(config, data_dict, args)
            
            # Evaluate trained model
            print("\n🏥 Evaluating trained model...")
            evaluation_results = evaluate_local_model(config, data_dict, model, args)
            print("✅ Training and evaluation completed successfully!")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return
            
        # Close wandb if used
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
    
    elif args.mode == 'evaluate':
        # Only evaluate
        evaluation_results = evaluate_local_model(config, data_dict, args=args)
    
    elif args.mode == 'inference':
        print("For inference mode, use the inference.py script directly:")
        print("python inference.py --model_path path/to/model.pth --image_path path/to/image.jpg")
    
    print(f"\n✅ Experiment completed successfully!")
    print(f"📁 Results saved to: {config.output_dir}")
    
    # Show final memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / 1e9
        print(f"🎮 Peak GPU memory usage: {memory_used:.1f}GB")

if __name__ == "__main__":
    main()