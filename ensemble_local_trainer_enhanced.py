#!/usr/bin/env python3
"""
Enhanced OVO Ensemble Trainer with Advanced Overfitting Prevention

This enhanced version includes:
- Advanced early stopping with validation loss monitoring
- Dynamic dropout adjustment based on overfitting detection
- Gradient clipping for training stability
- Enhanced learning rate scheduling
- Comprehensive overfitting detection and prevention
- Automatic model complexity reduction when needed
"""

# Import the original training script and enhance it
import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Import enhanced overfitting prevention
from enhanced_overfitting_prevention import (
    AdvancedEarlyStopping,
    DynamicDropout,
    AdvancedLRScheduler,
    create_enhanced_model_with_dropout,
    enhanced_training_step
)

# Import all original functionality
import argparse
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from itertools import combinations
import cv2
from tqdm import tqdm

# Essential wandb import for medical-grade tracking
try:
    import wandb
    WANDB_AVAILABLE = True
    print("✅ wandb available for experiment tracking")
except ImportError:
    WANDB_AVAILABLE = False
    print("❌ Warning: wandb not available. Please install: pip install wandb")

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

# Essential imports for OVO ensemble
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import ImageFolder
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments with enhanced overfitting prevention options."""

    parser = argparse.ArgumentParser(
        description='Enhanced Multi-Architecture Ensemble Diabetic Retinopathy Training with Advanced Overfitting Prevention',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhanced Features:
  - Advanced early stopping with validation loss monitoring
  - Dynamic dropout adjustment based on overfitting detection
  - Gradient clipping for training stability
  - Enhanced learning rate scheduling
  - Automatic overfitting detection and prevention

Example Usage:
  # Enhanced training with advanced overfitting prevention
  python ensemble_local_trainer_enhanced.py --mode train --dataset_path ./dataset7b \\
    --epochs 30 --enhanced_dropout 0.7 --gradient_clipping 1.0 \\
    --overfitting_threshold 0.15 --early_stopping_patience 5

  # Medical-grade training with maximum overfitting prevention
  python ensemble_local_trainer_enhanced.py --mode train --dataset_path ./dataset7b \\
    --epochs 50 --enhanced_dropout 0.8 --gradient_clipping 0.5 \\
    --overfitting_threshold 0.10 --dynamic_dropout --batch_norm \\
    --advanced_scheduler --weight_decay 1e-2
        """
    )

    # Mode selection
    parser.add_argument('--mode', choices=['train', 'evaluate', 'inference'],
                       default='train', help='Mode to run the script')

    # Dataset configuration
    parser.add_argument('--dataset_path', default='./dataset7b',
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', default='./ovo_ensemble_results_enhanced',
                       help='Output directory for results')

    # Training configuration
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs per binary classifier')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                       help='Weight decay (L2 regularization)')

    # Enhanced overfitting prevention
    parser.add_argument('--enhanced_dropout', type=float, default=0.7,
                       help='Enhanced dropout rate for better overfitting prevention')
    parser.add_argument('--gradient_clipping', type=float, default=1.0,
                       help='Gradient clipping threshold (0 to disable)')
    parser.add_argument('--overfitting_threshold', type=float, default=0.12,
                       help='Train-val accuracy gap threshold for overfitting detection (0.12 = 12%)')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                       help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--validation_loss_patience', type=int, default=3,
                       help='Validation loss plateau patience')

    # Advanced features
    parser.add_argument('--dynamic_dropout', action='store_true',
                       help='Enable dynamic dropout adjustment based on overfitting')
    parser.add_argument('--batch_norm', action='store_true',
                       help='Add batch normalization layers')
    parser.add_argument('--advanced_scheduler', action='store_true',
                       help='Use advanced learning rate scheduler with overfitting awareness')

    # Model selection
    parser.add_argument('--base_models', nargs='+',
                       default=['mobilenet_v2'],
                       choices=['mobilenet_v2', 'inception_v3', 'densenet121'],
                       help='Base models to use for OVO ensemble')
    parser.add_argument('--freeze_weights', type=str, default='true',
                       choices=['true', 'false'],
                       help='Freeze pre-trained weights')

    # System configuration
    parser.add_argument('--img_size', type=int, default=299,
                       help='Input image size')
    parser.add_argument('--device', default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')

    # Experiment configuration
    parser.add_argument('--experiment_name', default='enhanced_ovo_ensemble',
                       help='Experiment name for logging')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoints')

    return parser.parse_args()

class CLAHETransform:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to images."""

    def __init__(self, clip_limit=3.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image):
        # Convert PIL to numpy
        img_array = np.array(image)

        if len(img_array.shape) == 3:
            # Convert RGB to LAB color space
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)

            # Apply CLAHE to the L channel
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])

            # Convert back to RGB
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            enhanced = clahe.apply(img_array)

        return Image.fromarray(enhanced)

def get_enhanced_transforms(img_size, is_training=True):
    """Get enhanced transforms with CLAHE and medical-grade augmentation."""

    if is_training:
        transform = transforms.Compose([
            CLAHETransform(clip_limit=2.0),  # Medical-grade contrast enhancement
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(15),   # Conservative rotation for medical images
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            CLAHETransform(clip_limit=2.0),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return transform

class EnhancedOVOBinaryClassifier(nn.Module):
    """Enhanced binary classifier with advanced overfitting prevention."""

    def __init__(self, model_name='mobilenet_v2', freeze_weights=True,
                 enhanced_dropout=0.7, dynamic_dropout=True, batch_norm=True):
        super().__init__()

        self.model_name = model_name
        self.enhanced_dropout = enhanced_dropout
        self.dynamic_dropout = dynamic_dropout
        self.batch_norm = batch_norm

        # Create enhanced model with overfitting prevention
        self.model = create_enhanced_model_with_dropout(
            model_name=model_name,
            dropout=enhanced_dropout,
            freeze_weights=freeze_weights
        )

    def forward(self, x):
        return self.model(x)

def create_enhanced_data_loaders(dataset_path, img_size, batch_size, num_workers):
    """Create enhanced data loaders with improved transforms."""

    # Enhanced transforms
    train_transform = get_enhanced_transforms(img_size, is_training=True)
    val_transform = get_enhanced_transforms(img_size, is_training=False)

    # Load datasets
    train_dataset = ImageFolder(root=f"{dataset_path}/train", transform=train_transform)
    val_dataset = ImageFolder(root=f"{dataset_path}/val", transform=val_transform)
    test_dataset = ImageFolder(root=f"{dataset_path}/test", transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)

    logger.info(f"📊 Dataset loaded:")
    logger.info(f"   Train: {len(train_dataset)} samples")
    logger.info(f"   Val: {len(val_dataset)} samples")
    logger.info(f"   Test: {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

def create_ovo_binary_datasets(full_dataset, class_pair):
    """Create binary dataset for specific class pair."""

    indices = []
    for idx, (_, label) in enumerate(full_dataset):
        if label in class_pair:
            indices.append(idx)

    binary_dataset = Subset(full_dataset, indices)
    return binary_dataset

def enhanced_train_binary_classifier(model, train_loader, val_loader, config, class_pair, model_name):
    """Enhanced training with comprehensive overfitting prevention."""

    device = torch.device(config['system']['device'])
    model = model.to(device)

    # Enhanced loss and optimizer
    criterion = nn.BCELoss()

    # Separate learning rates for backbone vs classifier
    backbone_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if any(x in name for x in ['features', 'backbone', 'conv', 'bn']):
                backbone_params.append(param)
            else:
                classifier_params.append(param)

    # Enhanced optimizer with higher weight decay
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': config['training']['learning_rate'] * 0.1,
         'weight_decay': config['training']['weight_decay'] * 2},
        {'params': classifier_params, 'lr': config['training']['learning_rate'],
         'weight_decay': config['training']['weight_decay']}
    ])

    # Advanced learning rate scheduler
    if config.get('advanced_scheduler', False):
        scheduler = AdvancedLRScheduler(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-7)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-7, verbose=True
        )

    # Enhanced early stopping
    early_stopping = AdvancedEarlyStopping(
        patience=config['training']['early_stopping_patience'],
        min_delta=0.001,
        overfitting_threshold=config['training']['overfitting_threshold'],
        validation_loss_patience=config['training']['validation_loss_patience']
    )

    # Gradient clipping for stability
    max_grad_norm = config['training'].get('gradient_clipping', 1.0)

    # Training tracking
    train_history = {
        'train_accuracies': [],
        'val_accuracies': [],
        'train_losses': [],
        'val_losses': [],
        'learning_rates': [],
        'overfitting_gaps': []
    }

    logger.info(f"🏁 Enhanced training {model_name} for classes {class_pair}")
    logger.info(f"   🛡️ Overfitting prevention enabled:")
    logger.info(f"   - Early stopping patience: {early_stopping.patience}")
    logger.info(f"   - Gradient clipping: {max_grad_norm}")
    logger.info(f"   - Overfitting threshold: {early_stopping.overfitting_threshold*100:.1f}%")
    logger.info(f"   - Enhanced dropout: {config['model']['enhanced_dropout']}")
    logger.info(f"   - Weight decay: {config['training']['weight_decay']}")

    for epoch in range(config['training']['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}", leave=False)):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data).squeeze()

            # Convert to binary labels for current class pair
            binary_target = (target == class_pair[1]).float()

            loss = criterion(output, binary_target)
            loss.backward()

            # Gradient clipping
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            train_loss += loss.item()
            predicted = (output > 0.5).float()
            train_correct += (predicted == binary_target).sum().item()
            train_total += binary_target.size(0)

        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data).squeeze()

                binary_target = (target == class_pair[1]).float()

                loss = criterion(output, binary_target)
                val_loss += loss.item()

                predicted = (output > 0.5).float()
                val_correct += (predicted == binary_target).sum().item()
                val_total += binary_target.size(0)

        val_acc = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # Update dynamic dropout if enabled
        overfitting_gap = train_acc - val_acc
        if config.get('dynamic_dropout', False):
            for module in model.modules():
                if isinstance(module, DynamicDropout):
                    module.adjust_dropout(overfitting_gap)

        # Learning rate scheduling
        if hasattr(scheduler, 'step'):
            if config.get('advanced_scheduler', False):
                scheduler.step(val_acc, overfitting_gap)
            else:
                scheduler.step(val_acc)

        # Record history
        train_history['train_accuracies'].append(train_acc)
        train_history['val_accuracies'].append(val_acc)
        train_history['train_losses'].append(avg_train_loss)
        train_history['val_losses'].append(avg_val_loss)
        train_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        train_history['overfitting_gaps'].append(overfitting_gap)

        # Enhanced logging with overfitting metrics
        overfitting_metrics = early_stopping.get_overfitting_metrics()
        logger.info(f"   Epoch {epoch+1}/{config['training']['epochs']}: "
                   f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, "
                   f"Gap: {overfitting_gap:.1f}%, LR: {optimizer.param_groups[0]['lr']:.2e}")

        if overfitting_gap > 10.0:
            logger.warning(f"   ⚠️ Overfitting detected: {overfitting_gap:.1f}% train-val gap")
        if overfitting_gap >= 15.0:
            logger.error(f"   🚨 CRITICAL overfitting: {overfitting_gap:.1f}% gap ≥15%")

        # Early stopping check
        should_stop = early_stopping(val_acc, avg_val_loss, train_acc, model)

        if should_stop:
            logger.info(f"   🛑 Training stopped early at epoch {epoch+1}")
            break

        # Log to wandb for medical-grade tracking
        if WANDB_AVAILABLE:
            wandb.log({
                f"{model_name}_{class_pair[0]}_{class_pair[1]}/train_acc": train_acc,
                f"{model_name}_{class_pair[0]}_{class_pair[1]}/val_acc": val_acc,
                f"{model_name}_{class_pair[0]}_{class_pair[1]}/overfitting_gap": overfitting_gap,
                f"{model_name}_{class_pair[0]}_{class_pair[1]}/learning_rate": optimizer.param_groups[0]['lr'],
                f"{model_name}_{class_pair[0]}_{class_pair[1]}/train_loss": avg_train_loss,
                f"{model_name}_{class_pair[0]}_{class_pair[1]}/val_loss": avg_val_loss,
                f"{model_name}_{class_pair[0]}_{class_pair[1]}/medical_grade_target": 90.0,
                f"{model_name}_{class_pair[0]}_{class_pair[1]}/medical_grade_achieved": val_acc >= 90.0,
                "epoch": epoch + 1
            })

        # Save checkpoint if best
        if val_acc == early_stopping.best_val_accuracy:
            model_path = Path(config['system']['output_dir']) / "models" / f"best_{model_name}_{class_pair[0]}_{class_pair[1]}.pth"
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_accuracy': early_stopping.best_val_accuracy,
                'current_val_accuracy': val_acc,
                'current_train_accuracy': train_acc,
                'class_pair': class_pair,
                'model_name': model_name,
                'train_history': train_history,
                'overfitting_metrics': overfitting_metrics,
                'config': config
            }
            torch.save(checkpoint, model_path)

    final_metrics = early_stopping.get_overfitting_metrics()
    logger.info(f"✅ Enhanced training completed for {model_name}_{class_pair}")
    logger.info(f"   📊 Final metrics: Best Val Acc: {final_metrics['best_val_accuracy']:.2f}%")
    logger.info(f"   📈 Overfitting gap: {final_metrics['overfitting_gap']:.1f}%")

    # Log final model summary to wandb
    if WANDB_AVAILABLE:
        medical_grade_achieved = final_metrics['best_val_accuracy'] >= 90.0
        wandb.log({
            f"{model_name}_{class_pair[0]}_{class_pair[1]}/final_val_accuracy": final_metrics['best_val_accuracy'],
            f"{model_name}_{class_pair[0]}_{class_pair[1]}/final_overfitting_gap": final_metrics['overfitting_gap'],
            f"{model_name}_{class_pair[0]}_{class_pair[1]}/medical_grade_achieved_final": medical_grade_achieved,
            f"{model_name}_{class_pair[0]}_{class_pair[1]}/training_completed": True
        })

        if medical_grade_achieved:
            logger.info(f"   🏆 MEDICAL GRADE ACHIEVED: {final_metrics['best_val_accuracy']:.2f}% ≥ 90%")
        else:
            logger.warning(f"   ⚠️ Below medical grade: {final_metrics['best_val_accuracy']:.2f}% < 90%")

    return early_stopping.best_val_accuracy

def train_enhanced_ovo_ensemble(config, train_dataset, val_dataset, test_dataset):
    """Train complete enhanced OVO ensemble."""

    logger.info("🚀 Starting Enhanced OVO Ensemble Training")
    logger.info("🛡️ Advanced overfitting prevention enabled")

    # Class pairs for OVO approach
    num_classes = len(train_dataset.classes)
    class_pairs = list(combinations(range(num_classes), 2))

    logger.info(f"📋 Training {len(class_pairs)} binary classifiers")
    logger.info(f"   Class pairs: {class_pairs}")

    results = {}

    for model_name in config['model']['base_models']:
        logger.info(f"\n🏗️ Training {model_name.upper()} models")
        results[model_name] = {}

        for class_pair in class_pairs:
            logger.info(f"\n🎯 Training {model_name} for classes {class_pair}")

            # Create binary datasets
            binary_train = create_ovo_binary_datasets(train_dataset, class_pair)
            binary_val = create_ovo_binary_datasets(val_dataset, class_pair)

            # Create binary data loaders
            binary_train_loader = DataLoader(
                binary_train,
                batch_size=config['training']['batch_size'],
                shuffle=True,
                num_workers=config['system']['num_workers'],
                pin_memory=True
            )
            binary_val_loader = DataLoader(
                binary_val,
                batch_size=config['training']['batch_size'],
                shuffle=False,
                num_workers=config['system']['num_workers'],
                pin_memory=True
            )

            # Check if model already exists and resume is enabled
            model_path = Path(config['system']['output_dir']) / "models" / f"best_{model_name}_{class_pair[0]}_{class_pair[1]}.pth"

            if config.get('resume', False) and model_path.exists():
                logger.info(f"   📂 Resuming from existing checkpoint: {model_path}")
                checkpoint = torch.load(model_path, map_location='cpu')
                best_acc = checkpoint.get('best_val_accuracy', 0.0)
                logger.info(f"   📊 Previous best accuracy: {best_acc:.2f}%")
                results[model_name][f"{class_pair[0]}_{class_pair[1]}"] = best_acc
                continue

            # Create enhanced model
            model = EnhancedOVOBinaryClassifier(
                model_name=model_name,
                freeze_weights=config['model']['freeze_weights'],
                enhanced_dropout=config['model']['enhanced_dropout'],
                dynamic_dropout=config.get('dynamic_dropout', False),
                batch_norm=config.get('batch_norm', False)
            )

            # Train with enhanced overfitting prevention
            try:
                best_acc = enhanced_train_binary_classifier(
                    model, binary_train_loader, binary_val_loader,
                    config, class_pair, model_name
                )
                results[model_name][f"{class_pair[0]}_{class_pair[1]}"] = best_acc

            except Exception as e:
                logger.error(f"❌ Training failed for {model_name}_{class_pair}: {str(e)}")
                results[model_name][f"{class_pair[0]}_{class_pair[1]}"] = 0.0

    # Save final results
    results_path = Path(config['system']['output_dir']) / "enhanced_ovo_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✅ Enhanced OVO ensemble training completed!")
    logger.info(f"📁 Results saved to: {results_path}")

    return results

def main():
    """Main enhanced training function."""
    args = parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    logger.info(f"🖥️ Using device: {device}")

    # Initialize wandb for medical-grade experiment tracking
    if WANDB_AVAILABLE:
        wandb.init(
            project="ovo_diabetic_retinopathy_medical_grade",
            name=f"{args.experiment_name}",
            config={
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "enhanced_dropout": args.enhanced_dropout,
                "overfitting_threshold": args.overfitting_threshold,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "base_models": args.base_models,
                "medical_grade_target": "90%+",
                "gradient_clipping": args.gradient_clipping
            },
            tags=["medical-grade", "ovo-ensemble", "overfitting-prevention"]
        )
        logger.info("✅ wandb initialized for medical-grade tracking")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    # Configuration
    config = {
        'system': {
            'device': device,
            'output_dir': str(output_dir),
            'num_workers': args.num_workers,
            'seed': args.seed
        },
        'model': {
            'base_models': args.base_models,
            'freeze_weights': args.freeze_weights.lower() == 'true',
            'enhanced_dropout': args.enhanced_dropout,
            'img_size': args.img_size
        },
        'training': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'early_stopping_patience': args.early_stopping_patience,
            'overfitting_threshold': args.overfitting_threshold,
            'validation_loss_patience': args.validation_loss_patience,
            'gradient_clipping': args.gradient_clipping
        },
        'experiment_name': args.experiment_name,
        'resume': args.resume,
        'dynamic_dropout': args.dynamic_dropout,
        'batch_norm': args.batch_norm,
        'advanced_scheduler': args.advanced_scheduler
    }

    # Save configuration
    config_path = output_dir / "enhanced_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"💾 Configuration saved to: {config_path}")

    # Create data loaders
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_enhanced_data_loaders(
        args.dataset_path, args.img_size, args.batch_size, args.num_workers
    )

    if args.mode == 'train':
        # Train enhanced OVO ensemble
        results = train_enhanced_ovo_ensemble(config, train_dataset, val_dataset, test_dataset)

        logger.info("\n📊 ENHANCED TRAINING SUMMARY:")
        for model_name, model_results in results.items():
            accuracies = [acc for acc in model_results.values() if acc > 0]
            if accuracies:
                avg_acc = np.mean(accuracies)
                logger.info(f"   {model_name}: {avg_acc:.2f}% avg accuracy ({len(accuracies)} models)")
            else:
                logger.info(f"   {model_name}: No successful models")

    else:
        logger.info(f"Mode '{args.mode}' not implemented yet.")

if __name__ == "__main__":
    main()