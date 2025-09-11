#!/usr/bin/env python3
"""
Multi-Architecture Ensemble Local Trainer for Diabetic Retinopathy Classification

Main training script for the ensemble approach achieving 96.96% accuracy using
EfficientNetB2, ResNet50, and DenseNet121. This script preserves all essential
functionality from the original local_trainer.py while implementing the 
enhanced ensemble methodology.

Key Features:
- Multi-architecture ensemble training (EfficientNetB2 + ResNet50 + DenseNet121)
- Enhanced preprocessing with CLAHE and medical-grade augmentation
- SMOTE class balancing for improved minority class performance  
- Preserved checkpoint saving/loading, validation, and parameter functionality
- Medical-grade validation with 96.96% accuracy target
- GCS backup support for checkpoint management
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np

# Load environment variables
try:
    from dotenv import load_dotenv
    env_loaded = load_dotenv()
    if env_loaded:
        print("✅ Environment variables loaded from .env file")
    else:
        print("⚠️ Warning: .env file not found or empty")
except ImportError:
    print("❌ Error: python-dotenv not found. Installing...")
    print("Run: pip install python-dotenv")

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ Warning: wandb not available. Logging disabled.")

from ensemble_config import EnsembleConfig, create_default_config, create_medical_grade_config
from ensemble_dataset import (
    create_data_splits_ensemble,
    create_ensemble_dataloaders,
    compute_ensemble_class_weights
)
from ensemble_models import create_ensemble_model, validate_medical_grade_ensemble
from ensemble_trainer import EnsembleTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments with preserved functionality from local_trainer.py."""
    
    parser = argparse.ArgumentParser(
        description='Multi-Architecture Ensemble Diabetic Retinopathy Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  # Basic ensemble training
  python ensemble_local_trainer.py --mode train --dataset_path ./dataset3_augmented_resized --epochs 50

  # Medical-grade training with full features
  python ensemble_local_trainer.py --mode train --dataset_path ./dataset3_augmented_resized \\
    --epochs 100 --enable_clahe --enable_smote --enable_focal_loss --enable_class_weights \\
    --validation_frequency 1 --checkpoint_frequency 5 --output_dir ./ensemble_results

  # Resume from checkpoint
  python ensemble_local_trainer.py --mode train --dataset_path ./dataset3_augmented_resized \\
    --resume_from_checkpoint ./ensemble_results/checkpoints/ensemble_best.pth

  # Debug mode (2 epochs)
  python ensemble_local_trainer.py --mode train --dataset_path ./dataset3_augmented_resized \\
    --debug_mode --epochs 2
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', choices=['train', 'evaluate', 'inference'], 
                       default='train', help='Mode to run the script')
    
    # Dataset configuration (PRESERVED)
    parser.add_argument('--dataset_path', default='./dataset3_augmented_resized',
                       help='Path to dataset directory (train/val/test structure)')
    parser.add_argument('--num_classes', type=int, default=5,
                       help='Number of classes (5 for diabetic retinopathy)')
    parser.add_argument('--medical_terms', default='data/medical_terms_type1.json',
                       help='Path to medical terms JSON file')
    
    # Model ensemble configuration
    parser.add_argument('--ensemble_weights', nargs=3, type=float, 
                       default=[0.4, 0.35, 0.25],
                       help='Ensemble weights for EfficientNetB2, ResNet50, DenseNet121')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size (224 optimal for CNNs)')
    parser.add_argument('--individual_dropout', nargs=3, type=float,
                       default=[0.3, 0.3, 0.3],
                       help='Dropout rates for EfficientNetB2, ResNet50, DenseNet121')
    
    # Training hyperparameters (PRESERVED)
    parser.add_argument('--epochs', type=int, default=80,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=6,
                       help='Batch size (optimized for V100 memory)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (conservative for ensemble stability)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                       help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping')
    
    # Individual model learning rates
    parser.add_argument('--efficientnet_lr', type=float, default=None,
                       help='Learning rate for EfficientNetB2 (default: same as --learning_rate)')
    parser.add_argument('--resnet_lr', type=float, default=None,
                       help='Learning rate for ResNet50 (default: same as --learning_rate)')
    parser.add_argument('--densenet_lr', type=float, default=None,
                       help='Learning rate for DenseNet121 (default: same as --learning_rate)')
    
    # Enhanced preprocessing (NEW)
    parser.add_argument('--enable_clahe', action='store_true',
                       help='Enable CLAHE preprocessing (+3-5% accuracy)')
    parser.add_argument('--clahe_clip_limit', type=float, default=2.0,
                       help='CLAHE clip limit')
    parser.add_argument('--clahe_tile_grid_size', nargs=2, type=int, default=[8, 8],
                       help='CLAHE tile grid size')
    
    # SMOTE class balancing (NEW)
    parser.add_argument('--enable_smote', action='store_true',
                       help='Enable SMOTE class balancing')
    parser.add_argument('--smote_k_neighbors', type=int, default=5,
                       help='SMOTE k-neighbors parameter')
    
    # Medical-grade augmentation (NEW)
    parser.add_argument('--enable_medical_augmentation', action='store_true', default=True,
                       help='Enable medical-grade augmentation')
    parser.add_argument('--rotation_range', type=float, default=15.0,
                       help='Rotation range in degrees (±15° preserves anatomy)')
    parser.add_argument('--brightness_range', type=float, default=0.1,
                       help='Brightness variation range (±10%)')
    parser.add_argument('--contrast_range', type=float, default=0.1,
                       help='Contrast variation range (±10%)')
    
    # Loss configuration (PRESERVED)
    parser.add_argument('--enable_focal_loss', action='store_true',
                       help='Enable focal loss for class imbalance')
    parser.add_argument('--focal_loss_alpha', type=float, default=2.0,
                       help='Focal loss alpha parameter (reduced for ensemble)')
    parser.add_argument('--focal_loss_gamma', type=float, default=3.0,
                       help='Focal loss gamma parameter (reduced for ensemble)')
    parser.add_argument('--enable_class_weights', action='store_true',
                       help='Enable class weights for imbalanced data')
    parser.add_argument('--class_weight_severe', type=float, default=8.0,
                       help='Class weight multiplier for severe NPDR')
    parser.add_argument('--class_weight_pdr', type=float, default=6.0,
                       help='Class weight multiplier for PDR')
    
    # Scheduler configuration
    parser.add_argument('--scheduler', choices=['cosine', 'linear', 'plateau', 'none'],
                       default='cosine', help='Learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                       help='Number of warmup epochs')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                       help='Minimum learning rate')
    
    # Training strategy (NEW)
    parser.add_argument('--train_individual_models', action='store_true', default=True,
                       help='Train individual models separately first')
    parser.add_argument('--train_ensemble_jointly', action='store_true', default=True,
                       help='Train ensemble jointly after individual training')
    parser.add_argument('--ensemble_training_epochs', type=int, default=20,
                       help='Additional epochs for ensemble joint training')
    
    # Validation and checkpointing (PRESERVED)
    parser.add_argument('--validation_frequency', type=int, default=1,
                       help='Validation frequency (every N epochs)')
    parser.add_argument('--checkpoint_frequency', type=int, default=5,
                       help='Checkpoint saving frequency (every N epochs)')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=0.001,
                       help='Minimum improvement threshold')
    
    # Checkpoint management (PRESERVED)
    parser.add_argument('--resume_from_checkpoint', default=None,
                       help='Resume from checkpoint path')
    parser.add_argument('--save_checkpoint_gcs', default=None,
                       help='GCS bucket for checkpoint backup (e.g., gs://dr-data-2/checkpoints)')
    
    # Experiment settings (PRESERVED)
    parser.add_argument('--experiment_name', default='ensemble_efficientnetb2_resnet50_densenet121',
                       help='Experiment name for logging')
    parser.add_argument('--output_dir', default='./ensemble_results',
                       help='Output directory for results and checkpoints')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable wandb logging')
    
    # Device configuration (PRESERVED)
    parser.add_argument('--device', default='cuda', help='Device to use (cuda for V100)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Debug and testing
    parser.add_argument('--debug_mode', action='store_true',
                       help='Enable debug mode (2 epochs for testing)')
    parser.add_argument('--max_epochs', type=int, default=None,
                       help='Override maximum epochs for testing')
    
    # Medical-grade validation
    parser.add_argument('--enable_medical_validation', action='store_true', default=True,
                       help='Enable medical-grade validation')
    parser.add_argument('--target_accuracy', type=float, default=0.9696,
                       help='Target ensemble accuracy (96.96% from research)')
    
    return parser.parse_args()

def setup_ensemble_experiment(args) -> EnsembleConfig:
    """Setup ensemble experiment configuration from arguments."""
    
    # Start with medical-grade config for optimal settings
    config = create_medical_grade_config()
    
    # Update from command line arguments
    config = config.update_from_args(args)
    
    # Model-specific updates
    if args.ensemble_weights:
        config.model.ensemble_weights = args.ensemble_weights
        config.model.validate_weights()
    
    if args.individual_dropout:
        config.model.efficientnet_dropout = args.individual_dropout[0]
        config.model.resnet_dropout = args.individual_dropout[1]
        config.model.densenet_dropout = args.individual_dropout[2]
    
    if args.img_size:
        config.model.img_size = args.img_size
    
    # Individual learning rates
    if args.efficientnet_lr:
        config.training.efficientnet_lr = args.efficientnet_lr
    else:
        config.training.efficientnet_lr = config.training.learning_rate
        
    if args.resnet_lr:
        config.training.resnet_lr = args.resnet_lr
    else:
        config.training.resnet_lr = config.training.learning_rate
        
    if args.densenet_lr:
        config.training.densenet_lr = args.densenet_lr
    else:
        config.training.densenet_lr = config.training.learning_rate
    
    # Enhanced preprocessing
    if hasattr(args, 'clahe_clip_limit'):
        config.data.clahe_clip_limit = args.clahe_clip_limit
    if hasattr(args, 'clahe_tile_grid_size'):
        config.data.clahe_tile_grid_size = tuple(args.clahe_tile_grid_size)
    
    # SMOTE configuration
    if hasattr(args, 'smote_k_neighbors'):
        config.data.smote_k_neighbors = args.smote_k_neighbors
    
    # Medical augmentation
    if hasattr(args, 'rotation_range'):
        config.data.rotation_range = args.rotation_range
    if hasattr(args, 'brightness_range'):
        config.data.brightness_range = args.brightness_range
    if hasattr(args, 'contrast_range'):
        config.data.contrast_range = args.contrast_range
    
    # Training strategy
    if hasattr(args, 'train_individual_models'):
        config.training.train_individual_models = args.train_individual_models
    if hasattr(args, 'train_ensemble_jointly'):
        config.training.train_ensemble_jointly = args.train_ensemble_jointly
    if hasattr(args, 'ensemble_training_epochs'):
        config.training.ensemble_training_epochs = args.ensemble_training_epochs
    
    # Scheduler configuration
    if hasattr(args, 'scheduler'):
        config.training.scheduler = args.scheduler
    if hasattr(args, 'warmup_epochs'):
        config.training.warmup_epochs = args.warmup_epochs
    if hasattr(args, 'min_lr'):
        config.training.min_lr = args.min_lr
    
    # Medical validation
    if hasattr(args, 'target_accuracy'):
        config.system.target_ensemble_accuracy = args.target_accuracy
    
    # GCS backup
    if args.save_checkpoint_gcs:
        config.system.save_checkpoint_gcs = args.save_checkpoint_gcs
    
    # Wandb configuration
    config.system.use_wandb = not args.no_wandb and WANDB_AVAILABLE
    
    # Create output directories
    output_path = Path(config.system.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "checkpoints").mkdir(exist_ok=True)
    (output_path / "logs").mkdir(exist_ok=True)
    (output_path / "results").mkdir(exist_ok=True)
    
    return config

def prepare_ensemble_data(config: EnsembleConfig) -> Dict[str, Any]:
    """Prepare dataset for ensemble training with enhanced preprocessing."""
    
    logger.info(f"📁 Preparing ensemble dataset from: {config.data.dataset_path}")
    
    # Validate dataset structure
    dataset_path = Path(config.data.dataset_path)
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    # Create data splits
    train_data, val_data, test_data = create_data_splits_ensemble(
        str(dataset_path),
        num_classes=config.model.num_classes,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        test_split=config.data.test_split,
        seed=config.system.seed
    )
    
    logger.info(f"📊 Dataset splits created:")
    logger.info(f"   Training: {len(train_data)} samples")
    logger.info(f"   Validation: {len(val_data)} samples")
    logger.info(f"   Test: {len(test_data)} samples")
    
    # Create data loaders with enhanced preprocessing
    train_loader, val_loader, test_loader = create_ensemble_dataloaders(
        train_data, val_data, test_data, config
    )
    
    # Compute class weights for medical-grade training
    class_weights = None
    if config.data.enable_class_weights:
        class_weights = compute_ensemble_class_weights(
            train_data,
            num_classes=config.model.num_classes,
            severe_multiplier=config.training.class_weight_severe,
            pdr_multiplier=config.training.class_weight_pdr
        )
        logger.info(f"🏥 Class weights computed: {class_weights}")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'class_weights': class_weights,
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data
    }

def train_ensemble_model(config: EnsembleConfig, data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Train multi-architecture ensemble model."""
    
    logger.info("\n🚀 Initializing Multi-Architecture Ensemble...")
    logger.info(f"   EfficientNetB2 weight: {config.model.ensemble_weights[0]:.2f}")
    logger.info(f"   ResNet50 weight: {config.model.ensemble_weights[1]:.2f}")
    logger.info(f"   DenseNet121 weight: {config.model.ensemble_weights[2]:.2f}")
    logger.info(f"   Target accuracy: {config.system.target_ensemble_accuracy:.2%}")
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
    
    # Initialize device
    device = torch.device(config.system.device if torch.cuda.is_available() else 'cpu')
    
    # Initialize trainer
    trainer = EnsembleTrainer(
        config=config,
        train_loader=data_dict['train_loader'],
        val_loader=data_dict['val_loader'],
        device=device,
        class_weights=data_dict['class_weights']
    )
    
    # Initialize wandb if enabled
    if config.system.use_wandb:
        wandb.init(
            project=config.system.wandb_project,
            entity=config.system.wandb_entity,
            name=config.system.experiment_name,
            config=config.__dict__
        )
    
    # Execute training pipeline
    logger.info("\n🎯 Starting Ensemble Training Pipeline...")
    logger.info("=" * 60)
    
    try:
        training_results = trainer.full_training_pipeline()
        
        # Log results
        logger.info("\n🎉 Training Pipeline Completed!")
        logger.info("=" * 60)
        
        if 'individual_models' in training_results:
            logger.info("📊 Individual Model Results:")
            for model_name, results in training_results['individual_models'].items():
                logger.info(f"   {model_name}: {results['final_accuracy']:.4f} "
                           f"({'✅ PASS' if results['medical_grade_pass'] else '❌ FAIL'})")
        
        if 'ensemble' in training_results:
            ensemble_results = training_results['ensemble']
            logger.info(f"\n🎯 Ensemble Results:")
            logger.info(f"   Final Accuracy: {ensemble_results['final_accuracy']:.4f}")
            logger.info(f"   Medical Grade: {'✅ PASS' if ensemble_results['medical_grade_pass'] else '❌ FAIL'}")
            logger.info(f"   Research Target: {'✅ ACHIEVED' if ensemble_results.get('achieves_research_target', False) else '❌ NOT ACHIEVED'}")
        
        return training_results
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    finally:
        # Clean up wandb
        if config.system.use_wandb and WANDB_AVAILABLE:
            wandb.finish()

def evaluate_ensemble_model(config: EnsembleConfig, data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate trained ensemble model."""
    
    logger.info("🏥 Evaluating trained ensemble model...")
    
    # Load best ensemble checkpoint
    checkpoint_path = Path(config.system.output_dir) / "checkpoints" / "ensemble_best.pth"
    
    if not checkpoint_path.exists():
        logger.error(f"❌ No trained ensemble model found at: {checkpoint_path}")
        return {}
    
    # Initialize device
    device = torch.device(config.system.device if torch.cuda.is_available() else 'cpu')
    
    # Load ensemble model
    ensemble_model = create_ensemble_model(
        num_classes=config.model.num_classes,
        dropout=config.model.efficientnet_dropout,
        pretrained=config.model.use_pretrained,
        model_weights=config.model.ensemble_weights
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ensemble_model.load_state_dict(checkpoint['ensemble_state_dict'])
    
    logger.info(f"📥 Loaded ensemble model from: {checkpoint_path}")
    
    # Evaluate on test set
    ensemble_model.eval()
    all_predictions = []
    all_targets = []
    individual_predictions = {name: [] for name in ['efficientnetb2', 'resnet50', 'densenet121']}
    
    with torch.no_grad():
        for batch in tqdm(data_dict['test_loader'], desc="Evaluating"):
            images = batch['image'].to(device)
            targets = batch['dr_grade'].to(device)
            
            # Get ensemble and individual predictions
            outputs = ensemble_model(images, return_individual=True)
            
            # Ensemble predictions
            _, ensemble_pred = torch.max(outputs['dr_logits'], 1)
            all_predictions.extend(ensemble_pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Individual predictions
            for model_name in individual_predictions.keys():
                _, individual_pred = torch.max(outputs['individual_predictions'][model_name]['dr_logits'], 1)
                individual_predictions[model_name].extend(individual_pred.cpu().numpy())
    
    # Calculate comprehensive metrics
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    ensemble_accuracy = accuracy_score(all_targets, all_predictions)
    individual_accuracies = {
        name: accuracy_score(all_targets, preds) 
        for name, preds in individual_predictions.items()
    }
    
    # Medical validation
    ensemble_metrics = {
        'ensemble_accuracy': ensemble_accuracy,
        'mean_sensitivity': 0.0,  # Would need per-class calculation
        'mean_specificity': 0.0   # Would need per-class calculation
    }
    medical_validation = validate_medical_grade_ensemble(ensemble_metrics)
    
    results = {
        'ensemble_accuracy': ensemble_accuracy,
        'individual_accuracies': individual_accuracies,
        'medical_validation': medical_validation,
        'classification_report': classification_report(all_targets, all_predictions),
        'confusion_matrix': confusion_matrix(all_targets, all_predictions).tolist()
    }
    
    # Save evaluation results
    results_path = Path(config.system.output_dir) / "results" / "evaluation_results.json"
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        # Convert numpy arrays for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = value
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"💾 Evaluation results saved: {results_path}")
    
    # Print summary
    logger.info("\n📊 Evaluation Summary:")
    logger.info(f"   Ensemble Accuracy: {ensemble_accuracy:.4f}")
    logger.info(f"   EfficientNetB2: {individual_accuracies['efficientnetb2']:.4f}")
    logger.info(f"   ResNet50: {individual_accuracies['resnet50']:.4f}")
    logger.info(f"   DenseNet121: {individual_accuracies['densenet121']:.4f}")
    logger.info(f"   Medical Grade: {'✅ PASS' if medical_validation['medical_grade_pass'] else '❌ FAIL'}")
    
    return results

def verify_environment():
    """Verify environment setup and dependencies."""
    
    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available. This script requires GPU.")
        return False
    
    # Check required packages
    try:
        import cv2
        import albumentations
        import imblearn
    except ImportError as e:
        logger.error(f"❌ Missing required package: {e}")
        logger.info("Install with: pip install opencv-python albumentations imbalanced-learn")
        return False
    
    logger.info(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    return True

def main():
    """Main function for ensemble local training."""
    
    print("🎯 MULTI-ARCHITECTURE ENSEMBLE DIABETIC RETINOPATHY TRAINING")
    print("=" * 70)
    print("Models: EfficientNetB2 + ResNet50 + DenseNet121")
    print("Target: 96.96% accuracy (research validated)")
    print("=" * 70)
    
    # Verify environment
    if not verify_environment():
        return
    
    # Parse arguments
    args = parse_args()
    
    # Setup configuration
    try:
        config = setup_ensemble_experiment(args)
    except Exception as e:
        logger.error(f"❌ Configuration setup failed: {e}")
        return
    
    # Set random seed
    torch.manual_seed(config.system.seed)
    np.random.seed(config.system.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.system.seed)
    
    # Save configuration
    config_path = Path(config.system.output_dir) / "ensemble_config.json"
    config.save(config_path)
    logger.info(f"💾 Configuration saved: {config_path}")
    
    # Prepare data
    try:
        data_dict = prepare_ensemble_data(config)
    except Exception as e:
        logger.error(f"❌ Data preparation failed: {e}")
        return
    
    # Execute based on mode
    try:
        if args.mode == 'train':
            # Train ensemble model
            training_results = train_ensemble_model(config, data_dict)
            
            # Evaluate trained model
            evaluation_results = evaluate_ensemble_model(config, data_dict)
            
            logger.info("✅ Training and evaluation completed successfully!")
            
        elif args.mode == 'evaluate':
            # Only evaluate existing model
            evaluation_results = evaluate_ensemble_model(config, data_dict)
            
        elif args.mode == 'inference':
            logger.info("For inference mode, use the inference script with ensemble model")
            logger.info("Example: python inference.py --model_path ./ensemble_results/checkpoints/ensemble_best.pth")
        
    except Exception as e:
        logger.error(f"❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    logger.info(f"\n✅ Ensemble experiment completed successfully!")
    logger.info(f"📁 Results saved to: {config.system.output_dir}")
    
    # Show final GPU memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"🎮 Peak GPU memory usage: {memory_used:.1f}GB")

if __name__ == "__main__":
    main()