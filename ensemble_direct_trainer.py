#!/usr/bin/env python3
"""
Direct 5-Class Ensemble Trainer for Diabetic Retinopathy Classification

This script implements direct multi-class training instead of OVO approach to solve:
1. Critical overfitting issues (20%+ train-val gaps)
2. Poor accuracy on certain class pairs (60-70%)
3. Ensemble performance degradation

Key improvements:
- Direct 5-class training (no binary splitting)
- Comprehensive overfitting prevention
- Medical-grade regularization
- Enhanced data augmentation
- Proper learning rate scheduling
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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import cv2
from tqdm import tqdm

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

# Essential imports
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import ImageFolder
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments for direct ensemble training."""

    parser = argparse.ArgumentParser(
        description='Direct 5-Class Ensemble Diabetic Retinopathy Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  # Basic ensemble training with overfitting prevention
  python ensemble_direct_trainer.py --dataset_path ./dataset6 --epochs 50 --enable_overfitting_prevention

  # Medical-grade training with all features
  python ensemble_direct_trainer.py --dataset_path ./dataset6 --epochs 80 --enable_overfitting_prevention \
    --enable_clahe --enable_focal_loss --enable_class_weights --dropout 0.5 --weight_decay 1e-4

  # Debug mode (quick test)
  python ensemble_direct_trainer.py --dataset_path ./dataset6 --epochs 5 --debug_mode
        """
    )

    # Dataset configuration
    parser.add_argument('--dataset_path', default='./dataset6',
                       help='Path to dataset directory (train/val/test structure)')
    parser.add_argument('--num_classes', type=int, default=5,
                       help='Number of classes (5 for diabetic retinopathy)')

    # Model configuration
    parser.add_argument('--models', nargs='+',
                       default=['mobilenet_v2', 'inception_v3', 'densenet121'],
                       help='Models for ensemble')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size')

    # Training hyperparameters with overfitting prevention
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (increased for stability)')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate (reduced to prevent overfitting)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate for regularization')

    # Overfitting prevention (NEW)
    parser.add_argument('--enable_overfitting_prevention', action='store_true',
                       help='Enable comprehensive overfitting prevention')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--reduce_lr_patience', type=int, default=5,
                       help='Reduce LR on plateau patience')
    parser.add_argument('--min_lr', type=float, default=1e-7,
                       help='Minimum learning rate')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='Label smoothing for regularization')

    # Enhanced preprocessing
    parser.add_argument('--enable_clahe', action='store_true',
                       help='Enable CLAHE preprocessing')
    parser.add_argument('--augmentation_strength', type=float, default=0.3,
                       help='Data augmentation strength (0-1)')

    # Loss configuration
    parser.add_argument('--enable_focal_loss', action='store_true',
                       help='Enable focal loss for class imbalance')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                       help='Focal loss alpha parameter')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma parameter')
    parser.add_argument('--enable_class_weights', action='store_true',
                       help='Enable class weights for imbalanced data')

    # Experiment settings
    parser.add_argument('--output_dir', default='./ensemble_direct_results',
                       help='Output directory for results')
    parser.add_argument('--experiment_name', default='direct_ensemble_v1',
                       help='Experiment name')
    parser.add_argument('--device', default='cuda',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Debug mode
    parser.add_argument('--debug_mode', action='store_true',
                       help='Enable debug mode (reduced epochs)')

    return parser.parse_args()

class CLAHETransform:
    """CLAHE preprocessing for medical images."""

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, image):
        """Apply CLAHE to PIL image."""
        try:
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image

            if not isinstance(image_np, np.ndarray):
                return image

            if image_np.dtype != np.uint8:
                image_np = image_np.astype(np.uint8)

            # Apply CLAHE to LAB color space
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
                lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
                image_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                image_enhanced = self.clahe.apply(image_np)

            return Image.fromarray(image_enhanced.astype(np.uint8))

        except Exception as e:
            logger.warning(f"CLAHE failed: {e}")
            return image

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""

    def __init__(self, alpha=0.25, gamma=2.0, num_classes=5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class DirectClassifier(nn.Module):
    """Direct 5-class classifier with overfitting prevention."""

    def __init__(self, model_name='mobilenet_v2', num_classes=5, dropout=0.5):
        super(DirectClassifier, self).__init__()
        self.model_name = model_name

        # Load pre-trained model
        if model_name == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=True)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif model_name == 'inception_v3':
            self.backbone = models.inception_v3(pretrained=True, aux_logits=False)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == 'densenet121':
            self.backbone = models.densenet121(pretrained=True)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Freeze early layers, fine-tune later layers
        self._freeze_early_layers()

        # Enhanced classifier head with multiple dropout layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.7),  # Slightly less dropout
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.5),  # Even less dropout
            nn.Linear(256, num_classes)
        )

    def _freeze_early_layers(self):
        """Freeze early layers, fine-tune later layers."""
        if self.model_name == 'mobilenet_v2':
            # Freeze first 10 layers
            for i, param in enumerate(self.backbone.parameters()):
                if i < 50:  # Freeze more layers for stability
                    param.requires_grad = False
        elif self.model_name == 'inception_v3':
            # Freeze early conv layers
            for name, param in self.backbone.named_parameters():
                if any(x in name for x in ['Conv2d_1', 'Conv2d_2', 'Conv2d_3', 'Conv2d_4']):
                    param.requires_grad = False
        elif self.model_name == 'densenet121':
            # Freeze first 2 dense blocks
            for name, param in self.backbone.named_parameters():
                if 'denseblock1' in name or 'denseblock2' in name:
                    param.requires_grad = False

    def forward(self, x):
        """Forward pass with proper handling for different models."""
        # Handle InceptionV3 input size requirements
        if self.model_name == 'inception_v3' and x.size(-1) < 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        features = self.backbone(x)

        # Handle tuple outputs (e.g., from InceptionV3)
        if isinstance(features, tuple):
            features = features[0]

        # Global average pooling if needed
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)

        return self.classifier(features)

class DirectEnsemble(nn.Module):
    """Direct ensemble of 5-class classifiers."""

    def __init__(self, models=['mobilenet_v2', 'inception_v3', 'densenet121'],
                 num_classes=5, dropout=0.5):
        super(DirectEnsemble, self).__init__()
        self.models = models
        self.num_classes = num_classes

        # Create individual classifiers
        self.classifiers = nn.ModuleDict()
        for model_name in models:
            self.classifiers[model_name] = DirectClassifier(
                model_name=model_name,
                num_classes=num_classes,
                dropout=dropout
            )

        logger.info(f"✅ Direct ensemble created with {len(models)} models")

    def forward(self, x, return_individual=False):
        """Forward pass with ensemble averaging."""
        individual_outputs = {}
        ensemble_logits = torch.zeros(x.size(0), self.num_classes, device=x.device)

        for model_name, classifier in self.classifiers.items():
            logits = classifier(x)
            individual_outputs[model_name] = logits
            ensemble_logits += logits

        # Average the logits
        ensemble_logits /= len(self.classifiers)

        if return_individual:
            return ensemble_logits, individual_outputs
        return ensemble_logits

def create_transforms(img_size=224, enable_clahe=False, augmentation_strength=0.3):
    """Create training and validation transforms with medical-grade augmentation."""

    # Base transforms
    train_transforms = []
    val_transforms = []

    # Optional CLAHE
    if enable_clahe:
        train_transforms.append(CLAHETransform(clip_limit=2.0))
        val_transforms.append(CLAHETransform(clip_limit=2.0))

    # Resize
    train_transforms.extend([
        transforms.Resize((img_size, img_size)),
        # Conservative medical augmentation
        transforms.RandomRotation(degrees=15 * augmentation_strength),
        transforms.RandomHorizontalFlip(p=0.5 * augmentation_strength),
        transforms.ColorJitter(
            brightness=0.1 * augmentation_strength,
            contrast=0.1 * augmentation_strength,
            saturation=0.05 * augmentation_strength
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms.extend([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transforms.Compose(train_transforms), transforms.Compose(val_transforms)

def calculate_class_weights(dataset, num_classes=5):
    """Calculate class weights for imbalanced dataset."""
    class_counts = torch.zeros(num_classes)

    for _, label in dataset:
        class_counts[label] += 1

    total_samples = len(dataset)
    class_weights = total_samples / (num_classes * class_counts)

    # Normalize weights
    class_weights = class_weights / class_weights.sum() * num_classes

    logger.info(f"📊 Class weights: {class_weights.tolist()}")
    return class_weights

def train_single_model(model, train_loader, val_loader, config, model_name):
    """Train a single model with overfitting prevention."""

    device = torch.device(config['device'])
    model = model.to(device)

    # Setup loss function
    if config['enable_focal_loss']:
        criterion = FocalLoss(
            alpha=config['focal_alpha'],
            gamma=config['focal_gamma'],
            num_classes=config['num_classes']
        )
    else:
        # Use label smoothing for regularization
        criterion = nn.CrossEntropyLoss(
            weight=config.get('class_weights'),
            label_smoothing=config['label_smoothing']
        )

    # Setup optimizer with different learning rates
    optimizer = optim.AdamW(model.parameters(),
                           lr=config['learning_rate'],
                           weight_decay=config['weight_decay'])

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5,
        patience=config['reduce_lr_patience'],
        min_lr=config['min_lr'], verbose=True
    )

    # Training tracking
    best_val_acc = 0.0
    patience_counter = 0
    train_history = {
        'train_accuracies': [],
        'val_accuracies': [],
        'train_losses': [],
        'val_losses': [],
        'learning_rates': []
    }

    logger.info(f"🏁 Training {model_name}")

    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100.0 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100.0 * val_correct / val_total

        # Record history
        train_history['train_accuracies'].append(train_acc)
        train_history['val_accuracies'].append(val_acc)
        train_history['train_losses'].append(train_loss / len(train_loader))
        train_history['val_losses'].append(val_loss / len(val_loader))
        train_history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        # Step scheduler
        scheduler.step(val_acc)

        # Calculate overfitting metric
        overfitting_gap = train_acc - val_acc

        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # Save best model
            model_path = Path(config['output_dir']) / "models" / f"best_{model_name}.pth"
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_accuracy': best_val_acc,
                'current_val_accuracy': val_acc,
                'current_train_accuracy': train_acc,
                'train_history': train_history,
                'model_name': model_name,
                'config': config
            }
            torch.save(checkpoint, model_path)

            logger.info(f"🎯 New best {model_name}: {val_acc:.2f}% (gap: {overfitting_gap:.1f}%)")
        else:
            patience_counter += 1

        # Log progress with overfitting indicator
        overfitting_status = ""
        if overfitting_gap > 8.0:
            overfitting_status = " 🚨 CRITICAL OVERFITTING"
        elif overfitting_gap > 5.0:
            overfitting_status = " ⚠️ OVERFITTING"
        elif overfitting_gap > 3.0:
            overfitting_status = " 📈 MILD OVERFITTING"

        logger.info(f"   Epoch {epoch+1}: Train {train_acc:.2f}%, Val {val_acc:.2f}%{overfitting_status}")

        # Early stopping
        if config['enable_overfitting_prevention'] and patience_counter >= config['early_stopping_patience']:
            logger.info(f"   Early stopping at epoch {epoch+1}")
            break

    logger.info(f"✅ {model_name}: Best Val Acc = {best_val_acc:.2f}%")
    return best_val_acc, train_history

def train_ensemble(config):
    """Train the complete direct ensemble."""

    logger.info("🚀 Starting Direct Ensemble Training")
    logger.info("=" * 50)

    # Prepare data
    dataset_path = Path(config['dataset_path'])
    train_transform, val_transform = create_transforms(
        img_size=config['img_size'],
        enable_clahe=config['enable_clahe'],
        augmentation_strength=config['augmentation_strength']
    )

    # Load datasets
    train_dataset = ImageFolder(dataset_path / "train", transform=train_transform)
    val_dataset = ImageFolder(dataset_path / "val", transform=val_transform)
    test_dataset = ImageFolder(dataset_path / "test", transform=val_transform)

    logger.info(f"📊 Dataset loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Calculate class weights if enabled
    if config['enable_class_weights']:
        class_weights = calculate_class_weights(train_dataset, config['num_classes'])
        config['class_weights'] = class_weights.to(config['device'])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                           shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=4, pin_memory=True)

    # Train individual models
    trained_models = {}
    training_results = {}

    for model_name in config['models']:
        logger.info(f"\n🏗️ Training {model_name}")

        # Create model
        model = DirectClassifier(
            model_name=model_name,
            num_classes=config['num_classes'],
            dropout=config['dropout']
        )

        # Train model
        best_acc, history = train_single_model(
            model, train_loader, val_loader, config, model_name
        )

        trained_models[model_name] = model
        training_results[model_name] = {
            'best_accuracy': best_acc,
            'history': history
        }

    # Create ensemble
    logger.info("\n🎯 Creating ensemble")
    ensemble = DirectEnsemble(
        models=config['models'],
        num_classes=config['num_classes'],
        dropout=config['dropout']
    )

    # Load best weights into ensemble
    for model_name in config['models']:
        model_path = Path(config['output_dir']) / "models" / f"best_{model_name}.pth"
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location='cpu')
            ensemble.classifiers[model_name].load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"✅ Loaded {model_name} weights")

    # Save ensemble
    ensemble_path = Path(config['output_dir']) / "models" / "ensemble_best.pth"
    torch.save(ensemble.state_dict(), ensemble_path)
    logger.info(f"💾 Ensemble saved: {ensemble_path}")

    # Evaluate ensemble
    ensemble_acc = evaluate_ensemble(ensemble, test_loader, config)

    return ensemble, training_results, ensemble_acc

def evaluate_ensemble(ensemble, test_loader, config):
    """Evaluate ensemble on test set."""

    device = torch.device(config['device'])
    ensemble = ensemble.to(device)
    ensemble.eval()

    all_predictions = []
    all_targets = []
    individual_predictions = {model: [] for model in config['models']}

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            images, targets = images.to(device), targets.to(device)

            # Get ensemble predictions
            ensemble_logits, individual_outputs = ensemble(images, return_individual=True)

            # Ensemble predictions
            _, ensemble_pred = torch.max(ensemble_logits, 1)
            all_predictions.extend(ensemble_pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Individual predictions
            for model_name, logits in individual_outputs.items():
                _, pred = torch.max(logits, 1)
                individual_predictions[model_name].extend(pred.cpu().numpy())

    # Calculate accuracies
    ensemble_accuracy = accuracy_score(all_targets, all_predictions)
    individual_accuracies = {
        model: accuracy_score(all_targets, preds)
        for model, preds in individual_predictions.items()
    }

    logger.info(f"\n🏆 ENSEMBLE RESULTS:")
    logger.info(f"   Ensemble Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
    logger.info(f"   Medical Grade: {'✅ PASS' if ensemble_accuracy >= 0.90 else '❌ FAIL'}")

    logger.info(f"\n📊 INDIVIDUAL RESULTS:")
    for model, acc in individual_accuracies.items():
        logger.info(f"   {model}: {acc:.4f} ({acc*100:.2f}%)")

    return ensemble_accuracy

def main():
    """Main training function."""

    print("🎯 DIRECT 5-CLASS ENSEMBLE TRAINER")
    print("=" * 50)
    print("Solving OVO overfitting issues with direct multi-class training")
    print("=" * 50)

    # Parse arguments
    args = parse_args()

    # Set debug mode
    if args.debug_mode:
        args.epochs = 5
        logger.info("🐛 Debug mode: Using 5 epochs")

    # Create config
    config = {
        'dataset_path': args.dataset_path,
        'num_classes': args.num_classes,
        'models': args.models,
        'img_size': args.img_size,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'enable_overfitting_prevention': args.enable_overfitting_prevention,
        'early_stopping_patience': args.early_stopping_patience,
        'reduce_lr_patience': args.reduce_lr_patience,
        'min_lr': args.min_lr,
        'label_smoothing': args.label_smoothing,
        'enable_clahe': args.enable_clahe,
        'augmentation_strength': args.augmentation_strength,
        'enable_focal_loss': args.enable_focal_loss,
        'focal_alpha': args.focal_alpha,
        'focal_gamma': args.focal_gamma,
        'enable_class_weights': args.enable_class_weights,
        'output_dir': args.output_dir,
        'experiment_name': args.experiment_name,
        'device': args.device,
        'seed': args.seed
    }

    # Create output directories
    output_path = Path(config['output_dir'])
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "models").mkdir(exist_ok=True)
    (output_path / "logs").mkdir(exist_ok=True)

    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])

    logger.info(f"🎲 Random seed: {config['seed']}")
    logger.info(f"📁 Output directory: {output_path}")
    logger.info(f"⚙️ Overfitting prevention: {config['enable_overfitting_prevention']}")

    # Save config
    config_path = output_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    try:
        # Train ensemble
        ensemble, training_results, ensemble_acc = train_ensemble(config)

        logger.info(f"\n✅ Training completed!")
        logger.info(f"🎯 Final ensemble accuracy: {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")

        # Save results
        results = {
            'ensemble_accuracy': float(ensemble_acc),
            'training_results': training_results,
            'config': config
        }

        results_path = output_path / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"💾 Results saved: {results_path}")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()