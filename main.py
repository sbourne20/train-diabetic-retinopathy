#!/usr/bin/env python3
"""
Main training script for Diabetic Retinopathy Classification with Medical Reasoning
"""

import os
import argparse
import json
import torch
import wandb
from datetime import datetime

from config import get_config
from dataset import create_data_splits, create_dataloaders, compute_dataset_class_weights, save_data_splits
from models import DiabeticRetinopathyModel
from trainer import MedicalGradeDRTrainer
from evaluator import ModelEvaluator
from utils import set_seed, create_directories

def parse_args():
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(description='Train Diabetic Retinopathy Model')
    
    # Mode selection
    parser.add_argument('--mode', choices=['train', 'evaluate', 'inference'], 
                       default='train', help='Mode to run the script')
    
    # Data paths
    parser.add_argument('--rg_path', default='dataset/RG', 
                       help='Path to RG dataset directory')
    parser.add_argument('--me_path', default='dataset/ME', 
                       help='Path to ME dataset directory')
    parser.add_argument('--dataset_path', default=None,
                       help='Path to dataset directory (for type 1 structure)')
    
    # Dataset configuration
    parser.add_argument('--num_classes', type=int, default=2, 
                       help='Number of classes')
    parser.add_argument('--class_weights', action='store_true',
                       help='Enable class weighting for imbalanced data')
    parser.add_argument('--focal_loss', action='store_true',
                       help='Use focal loss for better minority class performance')
    parser.add_argument('--focal_loss_alpha', type=float, default=1.0,
                       help='Focal loss alpha parameter for class balancing')
    parser.add_argument('--focal_loss_gamma', type=float, default=2.0,
                       help='Focal loss gamma parameter for hard example focus')
    parser.add_argument('--class_weight_severe', type=float, default=3.0,
                       help='Class weight multiplier for Severe NPDR (Class 3)')
    parser.add_argument('--class_weight_pdr', type=float, default=2.5,
                       help='Class weight multiplier for PDR (Class 4)')
    parser.add_argument('--medical_grade', action='store_true',
                       help='Enable medical-grade validation metrics')
    parser.add_argument('--medical_terms', default=None,
                       help='Path to medical terms JSON file')
    
    # Model configuration
    parser.add_argument('--pretrained_path', default='google/medsiglip-448',
                       help='Path to MedSigLIP-448 model (HuggingFace)')
    parser.add_argument('--checkpoint_path', help='Path to model checkpoint')
    
    # LoRA fine-tuning parameters
    parser.add_argument('--use_lora', choices=['yes', 'no'], default='no',
                       help='Enable LoRA fine-tuning for memory efficiency')
    parser.add_argument('--lora_r', type=int, default=64,
                       help='LoRA rank parameter (default: 64 for maximum performance)')
    parser.add_argument('--lora_alpha', type=int, default=128,
                       help='LoRA alpha parameter (default: 128)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for MedSigLIP-448')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=448, help='Image size for MedSigLIP-448')
    
    # Advanced training parameters
    parser.add_argument('--freeze_backbone_epochs', type=int, default=0, help='Number of epochs to freeze backbone')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Number of warmup epochs')
    parser.add_argument('--scheduler', 
                        choices=["none", "polynomial", "linear", "validation_plateau", "cosine_with_restarts"],
                        default='polynomial', 
                        help='Learning rate scheduler: polynomial (gentle), linear (linear), validation_plateau (adaptive), cosine_with_restarts (aggressive), none (fixed)')
    parser.add_argument('--validation_frequency', type=int, default=5, help='Validation frequency')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum delta for early stopping')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for medical-grade regularization')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for clipping (prevents exploding gradients)')
    
    # Experiment settings
    parser.add_argument('--experiment_name', default=None, 
                       help='Experiment name for logging')
    parser.add_argument('--output_dir', default='outputs', 
                       help='Output directory')
    parser.add_argument('--no_wandb', action='store_true', 
                       help='Disable wandb logging')
    parser.add_argument('--save_to_gcs', default=None,
                       help='GCS path to save outputs to')
    
    # Device and reproducibility
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Data splits
    parser.add_argument('--use_existing_splits', action='store_true',
                       help='Use existing data splits if available')
    
    # Debug and testing options
    parser.add_argument('--debug_mode', action='store_true',
                       help='Enable debug mode for early testing (forces evaluation at epochs 1 & 2)')
    parser.add_argument('--max_epochs', type=int, default=None,
                       help='Override maximum epochs for testing (useful for debug mode)')
    parser.add_argument('--eval_frequency', type=int, default=None,
                       help='Override evaluation frequency (useful for debug mode)')
    parser.add_argument('--checkpoint_frequency', type=int, default=5,
                       help='Save checkpoints every N epochs')
    parser.add_argument('--resume_from_checkpoint', default=None,
                       help='Resume training from checkpoint (local path or gs:// path)')
    parser.add_argument('--save_checkpoint', default=None,
                       help='Save checkpoints to GCS bucket (e.g., gs://dr-data-2/checkpoints)')
    
    return parser.parse_args()

def setup_experiment(args):
    """Setup experiment configuration."""
    
    config = get_config()
    
    # Update config with command line arguments
    config.data.rg_path = args.rg_path
    config.data.me_path = args.me_path
    config.data.batch_size = args.batch_size
    config.model.pretrained_path = args.pretrained_path
    config.model.img_size = args.img_size
    
    # LoRA configuration
    config.model.use_lora = (args.use_lora == 'yes')
    config.model.lora_r = args.lora_r
    config.model.lora_alpha = args.lora_alpha
    config.model.dropout = args.dropout  # Medical-grade regularization
    config.training.num_epochs = args.epochs
    config.training.learning_rate = args.learning_rate
    config.output_dir = args.output_dir
    config.device = args.device
    config.seed = args.seed
    config.use_wandb = not args.no_wandb
    
    # Debug mode and testing configurations
    if args.debug_mode:
        print("🐛 DEBUG MODE ENABLED - Early pipeline testing")
        config.training.num_epochs = args.max_epochs or 2  # Default to 2 epochs in debug mode
        config.training.validation_frequency = args.eval_frequency or 1  # Evaluate every epoch
        config.training.checkpoint_frequency = 1  # Save checkpoint every epoch in debug
        print(f"   - Max epochs: {config.training.num_epochs}")
        print(f"   - Eval frequency: {config.training.validation_frequency}")
        print(f"   - Checkpoint frequency: {config.training.checkpoint_frequency}")
    else:
        # Override epochs and frequencies if specified
        if args.max_epochs:
            config.training.num_epochs = args.max_epochs
        if args.eval_frequency:
            config.training.validation_frequency = args.eval_frequency
        config.training.checkpoint_frequency = args.checkpoint_frequency
    
    # Resume from checkpoint
    config.training.resume_from_checkpoint = args.resume_from_checkpoint
    if args.resume_from_checkpoint:
        print(f"🔄 Resume from checkpoint: {args.resume_from_checkpoint}")
    
    # GCS checkpoint saving
    config.training.save_checkpoint = args.save_checkpoint
    if args.save_checkpoint:
        # Extract bucket name from gs:// URL
        if args.save_checkpoint.startswith('gs://'):
            # Extract bucket name (e.g., "dr-data-2" from "gs://dr-data-2/checkpoints")
            bucket_path = args.save_checkpoint[5:]  # Remove 'gs://'
            bucket_name = bucket_path.split('/')[0]
            config.training.gcs_bucket = bucket_name
            print(f"💾 GCS checkpoint saving enabled: {args.save_checkpoint}")
            print(f"🪣 GCS bucket: {bucket_name}")
        else:
            print(f"⚠️ Warning: Invalid GCS path format: {args.save_checkpoint}")
            print("   Expected format: gs://bucket-name/path")
            config.training.gcs_bucket = "dr-data-2"  # Fallback to default
    else:
        config.training.gcs_bucket = "dr-data-2"  # Default bucket
    
    # New dataset type 1 arguments
    if args.dataset_path:
        config.data.dataset_path = args.dataset_path
    config.data.num_classes = args.num_classes
    config.data.class_weights = args.class_weights
    config.training.focal_loss = args.focal_loss
    config.training.focal_loss_alpha = args.focal_loss_alpha
    config.training.focal_loss_gamma = args.focal_loss_gamma
    config.training.class_weight_severe = args.class_weight_severe
    config.training.class_weight_pdr = args.class_weight_pdr
    config.training.medical_grade = args.medical_grade
    if args.medical_terms:
        config.data.medical_terms = args.medical_terms
        config.language.medical_terms_path = args.medical_terms
    
    # Advanced training parameters
    config.training.freeze_backbone_epochs = args.freeze_backbone_epochs
    config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.training.warmup_epochs = args.warmup_epochs
    config.training.scheduler = args.scheduler
    config.training.validation_frequency = args.validation_frequency
    config.training.patience = args.patience
    config.training.min_delta = args.min_delta
    config.training.weight_decay = args.weight_decay
    config.training.max_grad_norm = args.max_grad_norm
    
    # Set experiment name
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.experiment_name = f"dr_model_{timestamp}"
    
    # Set reproducibility
    set_seed(config.seed)
    
    # Create directories
    create_directories(config)
    
    return config

def prepare_data(config, args):
    """Prepare dataset and dataloaders."""
    
    print("Preparing dataset...")
    
    # Check if we should use existing splits
    splits_exist = (
        os.path.exists("data/train_split.json") and 
        os.path.exists("data/val_split.json") and 
        os.path.exists("data/test_split.json")
    )
    
    if args.use_existing_splits and splits_exist:
        print("Loading existing data splits...")
        from dataset import load_data_splits
        train_data, val_data, test_data = load_data_splits()
    else:
        print("Creating new data splits...")
        
        # Handle different dataset structures
        if hasattr(config.data, 'dataset_path') and config.data.dataset_path:
            # Dataset type 1: structured train/val/test directories
            print("Using dataset type 1 structure (train/val/test directories)")
            from dataset import create_data_splits_type1
            train_data, val_data, test_data = create_data_splits_type1(
                config.data.dataset_path,
                num_classes=config.data.num_classes,
                seed=config.seed,
                max_train_samples=15000  # Aggressive speed optimization: reduce from ~115k to 15k samples
            )
        else:
            # Dataset type 0: RG/ME structure
            print("Using dataset type 0 structure (RG/ME directories)")
            train_data, val_data, test_data = create_data_splits(
                config.data.rg_path,
                config.data.me_path,
                train_split=config.data.train_split,
                val_split=config.data.val_split,
                test_split=config.data.test_split,
                seed=config.seed
            )
        
        # Save splits for reproducibility
        save_data_splits(train_data, val_data, test_data)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, config
    )
    
    # Compute class weights based on dataset type
    if hasattr(config.data, 'dataset_path') and config.data.dataset_path:
        # Dataset type 1: single DR classification with medical-grade weighting
        from dataset import compute_dr_class_weights
        dr_weights = compute_dr_class_weights(
            train_data, 
            config.data.num_classes,
            severe_multiplier=config.training.class_weight_severe,
            pdr_multiplier=config.training.class_weight_pdr
        )
        rg_weights, me_weights = None, None  # Not used for type 1
    else:
        # Dataset type 0: RG/ME classification
        rg_weights, me_weights = compute_dataset_class_weights(
            train_data, 
            config.model.num_classes_rg, 
            config.model.num_classes_me
        )
        dr_weights = None
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'rg_weights': rg_weights,
        'me_weights': me_weights,
        'dr_weights': dr_weights,
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data
    }

def train_model(config, data_dict, args):
    """Train the model."""
    
    print("Initializing model...")
    model = DiabeticRetinopathyModel(
        img_size=config.model.img_size,
        num_classes=5,  # 5-class ICDR classification
        dropout=config.model.dropout,
        enable_confidence=config.model.enable_confidence_estimation,
        use_lora=config.model.use_lora,
        lora_r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Initialize trainer with correct parameters including new checkpoint functionality
    trainer = MedicalGradeDRTrainer(
        model=model,
        train_loader=data_dict['train_loader'],
        val_loader=data_dict['val_loader'],
        device=torch.device(config.device),
        learning_rate=config.training.learning_rate,
        num_epochs=config.training.num_epochs,
        patience=config.training.patience,
        use_mixed_precision=config.mixed_precision,
        class_weights=data_dict['dr_weights'],
        validation_frequency=config.training.validation_frequency,
        checkpoint_frequency=config.training.checkpoint_frequency,
        gcs_bucket=config.training.gcs_bucket,  # Use configured GCS bucket
        resume_from_checkpoint=config.training.resume_from_checkpoint,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        # Medical-grade parameters from config
        enable_focal_loss=config.training.focal_loss,
        focal_loss_alpha=config.training.focal_loss_alpha,
        focal_loss_gamma=config.training.focal_loss_gamma,
        weight_decay=config.training.weight_decay,
        scheduler=config.training.scheduler,
        max_grad_norm=config.training.max_grad_norm
    )
    
    # Load checkpoint if provided
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint: {args.checkpoint_path}")
        start_epoch = trainer.load_checkpoint(args.checkpoint_path)
        print(f"Resumed from epoch {start_epoch}")
    
    # Train model
    print("Starting training...")
    training_history = trainer.train()
    
    # Save final model
    final_model_path = os.path.join(config.checkpoint_dir, "final_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'training_history': training_history
    }, final_model_path)
    
    print(f"Final model saved to: {final_model_path}")
    
    return model, training_history

def evaluate_model(config, data_dict, model=None, args=None):
    """Evaluate the model."""
    
    if model is None:
        print("Loading model for evaluation...")
        model = DiabeticRetinopathyModel(
            img_size=config.model.img_size,
            num_classes=5,  # 5-class ICDR classification
            dropout=config.model.dropout,
            enable_confidence=config.model.enable_confidence_estimation,
            use_lora=config.model.use_lora,
            lora_r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha
        )
        
        # Load best model
        best_model_path = os.path.join(config.checkpoint_dir, "best_model.pth")
        if not os.path.exists(best_model_path) and args and args.checkpoint_path:
            best_model_path = args.checkpoint_path
            
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from: {best_model_path}")
        else:
            print("Warning: No trained model found. Using random initialization.")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model=model,
        test_loader=data_dict['test_loader'],
        config=config,
        device=config.device
    )
    
    # Run evaluation
    print("Running comprehensive evaluation...")
    evaluation_results = evaluator.evaluate_model(save_results=True)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    summary = evaluation_results['overall_summary']
    print(f"RG Classification Accuracy: {summary['overall_accuracy']['rg']:.4f}")
    print(f"ME Classification Accuracy: {summary['overall_accuracy']['me']:.4f}")
    print(f"Combined Accuracy: {summary['overall_accuracy']['combined']:.4f}")
    
    if summary['overall_auc']['rg'] > 0:
        print(f"RG AUC (macro): {summary['overall_auc']['rg']:.4f}")
    if summary['overall_auc']['me'] > 0:
        print(f"ME AUC (macro): {summary['overall_auc']['me']:.4f}")
    
    clinical = summary['clinical_relevance']
    print(f"Severe DR Detection Rate: {clinical['severe_dr_detection']:.4f}")
    print(f"High ME Risk Detection Rate: {clinical['high_me_risk_detection']:.4f}")
    print(f"Clinical Utility: {clinical['overall_clinical_utility']}")
    
    return evaluation_results

def main():
    """Main function."""
    
    args = parse_args()
    
    # Setup experiment
    config = setup_experiment(args)
    
    print(f"Experiment: {config.experiment_name}")
    print(f"Device: {config.device}")
    print(f"Random seed: {config.seed}")
    
    # Save config
    config_path = os.path.join(config.output_dir, "config.json")
    with open(config_path, 'w') as f:
        # Convert config to dict for JSON serialization
        config_dict = {
            'model': config.model.__dict__,
            'data': config.data.__dict__,
            'training': config.training.__dict__,
            'language': config.language.__dict__,
            'experiment_name': config.experiment_name,
            'output_dir': config.output_dir,
            'device': config.device,
            'seed': config.seed
        }
        json.dump(config_dict, f, indent=2)
    
    # Prepare data
    if args.mode in ['train', 'evaluate']:
        data_dict = prepare_data(config, args)
    
    # Execute based on mode
    if args.mode == 'train':
        # Train model
        model, training_history = train_model(config, data_dict, args)
        
        # IMPORTANT: Upload model artifacts BEFORE evaluation to ensure they're saved even if evaluation fails
        print("\nUploading trained models to GCS...")
        if args.save_to_gcs:
            try:
                from google.cloud import storage
                
                client = storage.Client()
                bucket_name = args.save_to_gcs.replace('gs://', '').split('/')[0]
                bucket = client.bucket(bucket_name)
                
                # Upload final model
                final_model_local = os.path.join(config.checkpoint_dir, "final_model.pth")
                if os.path.exists(final_model_local):
                    final_model_blob = bucket.blob("models/final_model.pth")
                    final_model_blob.upload_from_filename(final_model_local)
                    print(f"✅ Uploaded final model to gs://{bucket_name}/models/final_model.pth")
                
                # Upload best model
                best_model_local = os.path.join(config.checkpoint_dir, "best_medical_model.pth")
                if os.path.exists(best_model_local):
                    best_model_blob = bucket.blob("models/best_medical_model.pth")
                    best_model_blob.upload_from_filename(best_model_local)
                    print(f"✅ Uploaded best model to gs://{bucket_name}/models/best_medical_model.pth")
                
                print("✅ Model artifacts safely uploaded to GCS!")
                
            except Exception as e:
                print(f"⚠️  Warning: Model upload failed: {e}")
                print("Models are still saved locally and in checkpoints/")
        
        # Evaluate trained model (with error handling to prevent loss of training results)
        print("\nEvaluating trained model...")
        try:
            evaluation_results = evaluate_model(config, data_dict, model, args)
            print("✅ Evaluation completed successfully!")
        except Exception as e:
            print(f"⚠️  Warning: Evaluation failed: {e}")
            print("This doesn't affect the trained model - it's still saved!")
            evaluation_results = None
        
        # Close wandb if used
        if config.use_wandb:
            wandb.finish()
    
    elif args.mode == 'evaluate':
        # Only evaluate
        evaluation_results = evaluate_model(config, data_dict, args=args)
    
    elif args.mode == 'inference':
        print("For inference mode, use the inference.py script directly:")
        print("python inference.py --model_path path/to/model.pth --image_path path/to/image.jpg")
    
    # Upload remaining results to GCS if specified (evaluation outputs, logs, etc.)
    if args.save_to_gcs and args.mode == 'train':
        print(f"\nUploading evaluation results and logs to GCS: {args.save_to_gcs}")
        try:
            from google.cloud import storage
            
            client = storage.Client()
            bucket_name = args.save_to_gcs.replace('gs://', '').split('/')[0]
            bucket = client.bucket(bucket_name)
            
            # Upload all files from output_dir to GCS (evaluation results, visualizations, etc.)
            if os.path.exists(config.output_dir):
                for root, dirs, files in os.walk(config.output_dir):
                    for file in files:
                        local_path = os.path.join(root, file)
                        relative_path = os.path.relpath(local_path, config.output_dir)
                        gcs_path = f"outputs/{relative_path}"
                        
                        blob = bucket.blob(gcs_path)
                        blob.upload_from_filename(local_path)
                        print(f"Uploaded {relative_path} to {args.save_to_gcs}/{gcs_path}")
                
                print(f"✅ All evaluation results uploaded to: {args.save_to_gcs}")
            else:
                print("No output directory found - only models were uploaded")
                
        except Exception as e:
            print(f"⚠️  Warning: Failed to upload evaluation results to GCS: {e}")
            print(f"Results are still available locally at: {config.output_dir}")
            print("✅ Models were already uploaded successfully!")
    
    print("\nExperiment completed successfully!")
    print(f"Results saved to: {config.output_dir}")
    if args.save_to_gcs:
        print(f"Results also uploaded to: {args.save_to_gcs}")

if __name__ == "__main__":
    main()