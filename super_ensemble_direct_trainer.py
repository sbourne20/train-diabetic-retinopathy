#!/usr/bin/env python3
"""
Super-Ensemble Direct Trainer for Diabetic Retinopathy Classification

This script implements a medical-grade super-ensemble combining:
- MedSigLIP-448 (medical specialist)
- EfficientNet-B3 (efficient baseline)
- EfficientNet-B4 (optimal balance)
- EfficientNet-B5 (maximum accuracy)

Optimized for V100 16GB with memory-efficient training strategies.
Target: 92-96% medical-grade accuracy
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

# Memory optimization imports
from torch.utils.checkpoint import checkpoint
import gc

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

# Try to import timm for EfficientNet
try:
    import timm
    TIMM_AVAILABLE = True
    print("✅ TIMM available for EfficientNet models")
except ImportError:
    TIMM_AVAILABLE = False
    print("❌ TIMM not found. Install with: pip install timm")

# Try to import transformers for MedSigLIP
try:
    from transformers import AutoModel, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers available for MedSigLIP")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("❌ Transformers not found. Install with: pip install transformers")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments for super-ensemble training."""

    parser = argparse.ArgumentParser(
        description='Super-Ensemble Direct Trainer: MedSigLIP + EfficientNets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  # Medical-grade super-ensemble training
  python super_ensemble_direct_trainer.py --dataset_path ./dataset6 --epochs 40 --enable_memory_optimization

  # Quick test (debug mode)
  python super_ensemble_direct_trainer.py --dataset_path ./dataset6 --debug_mode --epochs 3
        """
    )

    # Dataset configuration
    parser.add_argument('--dataset_path', default='./dataset6',
                       help='Path to dataset directory (train/val/test structure)')
    parser.add_argument('--num_classes', type=int, default=5,
                       help='Number of classes (5 for diabetic retinopathy)')

    # Model configuration
    parser.add_argument('--models', nargs='+',
                       default=['medsiglip_448', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5'],
                       help='Models for super-ensemble')

    # Memory optimization for V100 16GB
    parser.add_argument('--enable_memory_optimization', action='store_true', default=True,
                       help='Enable memory optimization for V100 16GB')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True,
                       help='Enable gradient checkpointing to save memory')
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                       help='Enable mixed precision training (FP16)')

    # Training hyperparameters optimized for large models
    parser.add_argument('--epochs', type=int, default=40,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (optimized for V100 16GB)')
    parser.add_argument('--learning_rate', type=float, default=5e-6,
                       help='Learning rate (very conservative for large models)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate (reduced for large models)')

    # Advanced training strategies
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='Warmup epochs for learning rate')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                       help='Early stopping patience (longer for large models)')
    parser.add_argument('--reduce_lr_patience', type=int, default=8,
                       help='Reduce LR on plateau patience')
    parser.add_argument('--min_lr', type=float, default=1e-8,
                       help='Minimum learning rate')

    # Model-specific learning rates
    parser.add_argument('--medsiglip_lr_multiplier', type=float, default=0.1,
                       help='Learning rate multiplier for MedSigLIP (pre-trained)')
    parser.add_argument('--efficientnet_lr_multiplier', type=float, default=1.0,
                       help='Learning rate multiplier for EfficientNets')

    # Preprocessing and augmentation
    parser.add_argument('--enable_clahe', action='store_true', default=True,
                       help='Enable CLAHE preprocessing')
    parser.add_argument('--augmentation_strength', type=float, default=0.2,
                       help='Data augmentation strength (conservative for medical)')

    # Loss configuration
    parser.add_argument('--enable_focal_loss', action='store_true', default=True,
                       help='Enable focal loss for class imbalance')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                       help='Focal loss alpha parameter')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma parameter')
    parser.add_argument('--enable_class_weights', action='store_true', default=True,
                       help='Enable class weights for imbalanced data')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='Label smoothing for regularization')

    # Experiment settings
    parser.add_argument('--output_dir', default='./super_ensemble_results',
                       help='Output directory for results')
    parser.add_argument('--experiment_name', default='medsiglip_efficientnet_super_ensemble',
                       help='Experiment name')
    parser.add_argument('--device', default='cuda',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Debug mode
    parser.add_argument('--debug_mode', action='store_true',
                       help='Enable debug mode (reduced epochs and models)')

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

class MedSigLIPClassifier(nn.Module):
    """MedSigLIP-448 classifier optimized for V100 16GB."""

    def __init__(self, num_classes=5, dropout=0.3, enable_checkpointing=True):
        super(MedSigLIPClassifier, self).__init__()
        self.num_classes = num_classes
        self.enable_checkpointing = enable_checkpointing

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not available. Install with: pip install transformers")

        # Load MedSigLIP model (this is a placeholder - replace with actual model)
        # For now, using a vision transformer as proxy
        try:
            # Try to load actual MedSigLIP if available
            self.backbone = AutoModel.from_pretrained("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
            logger.info("✅ Loaded BiomedCLIP as MedSigLIP proxy")
        except:
            # Fallback to standard vision transformer
            try:
                self.backbone = AutoModel.from_pretrained("google/vit-base-patch16-224")
                logger.info("⚠️ Using ViT-Base as MedSigLIP fallback")
            except:
                # Final fallback to timm model
                if TIMM_AVAILABLE:
                    self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True)
                    logger.info("⚠️ Using TIMM ViT-Base as final fallback")
                else:
                    raise ImportError("No vision transformer available")

        # Get feature dimension
        if hasattr(self.backbone, 'config'):
            hidden_size = self.backbone.config.hidden_size
        else:
            hidden_size = 768  # Standard ViT hidden size

        # Medical-grade classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.7),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )

        # Freeze early layers for stability
        self._freeze_early_layers()

    def _freeze_early_layers(self):
        """Freeze early layers to prevent overfitting."""
        if hasattr(self.backbone, 'embeddings'):
            for param in self.backbone.embeddings.parameters():
                param.requires_grad = False

        # Freeze first few transformer blocks
        if hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layer'):
            for i in range(6):  # Freeze first 6 layers
                if i < len(self.backbone.encoder.layer):
                    for param in self.backbone.encoder.layer[i].parameters():
                        param.requires_grad = False

    def forward(self, x):
        """Forward pass with optional gradient checkpointing."""
        if self.enable_checkpointing and self.training:
            outputs = checkpoint(self.backbone, x, use_reentrant=False)
        else:
            outputs = self.backbone(x)

        # Handle different output formats from transformers
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state[:, 0]  # CLS token
        elif hasattr(outputs, 'pooler_output'):
            features = outputs.pooler_output
        elif hasattr(outputs, 'logits'):
            features = outputs.logits
        elif isinstance(outputs, torch.Tensor):
            features = outputs
        else:
            # Handle BaseModelOutputWithPooling and similar objects
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                features = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state[:, 0]  # Take CLS token
            else:
                # Last resort: try to get the first tensor-like attribute
                for attr_name in dir(outputs):
                    attr = getattr(outputs, attr_name)
                    if isinstance(attr, torch.Tensor) and len(attr.shape) >= 2:
                        if len(attr.shape) == 3:  # [batch, seq_len, hidden]
                            features = attr[:, 0]  # Take first token
                        else:
                            features = attr
                        break
                else:
                    raise RuntimeError(f"Cannot extract features from {type(outputs)}")

        # Handle different tensor shapes
        if isinstance(features, torch.Tensor):
            if len(features.shape) > 2:
                features = F.adaptive_avg_pool1d(features.transpose(1, 2), 1).squeeze(-1)
        else:
            raise RuntimeError(f"Features must be tensor, got {type(features)}")

        return self.classifier(features)

class EfficientNetClassifier(nn.Module):
    """EfficientNet classifier with memory optimization."""

    def __init__(self, model_name='efficientnet_b3', num_classes=5, dropout=0.3, enable_checkpointing=True):
        super(EfficientNetClassifier, self).__init__()
        self.model_name = model_name
        self.enable_checkpointing = enable_checkpointing

        if not TIMM_AVAILABLE:
            raise ImportError("timm not available. Install with: pip install timm")

        # Load EfficientNet model
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        num_features = self.backbone.num_features

        # Freeze early layers for stability
        self._freeze_early_layers()

        # Medical-grade classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.7),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )

    def _freeze_early_layers(self):
        """Freeze early layers to prevent overfitting."""
        # Freeze stem and first few blocks
        if hasattr(self.backbone, 'conv_stem'):
            for param in self.backbone.conv_stem.parameters():
                param.requires_grad = False

        if hasattr(self.backbone, 'blocks'):
            # Freeze first 2 blocks
            for i in range(min(2, len(self.backbone.blocks))):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = False

    def forward(self, x):
        """Forward pass with optional gradient checkpointing."""
        if self.enable_checkpointing and self.training:
            features = checkpoint(self.backbone, x)
        else:
            features = self.backbone(x)

        return self.classifier(features)

class SuperEnsemble(nn.Module):
    """Super-ensemble combining MedSigLIP + EfficientNets."""

    def __init__(self, models=['medsiglip_448', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5'],
                 num_classes=5, dropout=0.3, enable_checkpointing=True):
        super(SuperEnsemble, self).__init__()
        self.models = models
        self.num_classes = num_classes

        # Create individual classifiers
        self.classifiers = nn.ModuleDict()

        for model_name in models:
            if model_name == 'medsiglip_448':
                self.classifiers[model_name] = MedSigLIPClassifier(
                    num_classes=num_classes,
                    dropout=dropout,
                    enable_checkpointing=enable_checkpointing
                )
            elif model_name.startswith('efficientnet'):
                self.classifiers[model_name] = EfficientNetClassifier(
                    model_name=model_name,
                    num_classes=num_classes,
                    dropout=dropout,
                    enable_checkpointing=enable_checkpointing
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")

        logger.info(f"✅ Super-ensemble created with {len(models)} models")

    def forward(self, x, return_individual=False):
        """Forward pass with ensemble averaging."""
        individual_outputs = {}
        ensemble_logits = torch.zeros(x.size(0), self.num_classes, device=x.device)

        for model_name, classifier in self.classifiers.items():
            # Different input sizes for different models
            if model_name == 'medsiglip_448':
                input_x = F.interpolate(x, size=(448, 448), mode='bilinear', align_corners=False)
            elif model_name == 'efficientnet_b3':
                input_x = F.interpolate(x, size=(300, 300), mode='bilinear', align_corners=False)
            elif model_name == 'efficientnet_b4':
                input_x = F.interpolate(x, size=(380, 380), mode='bilinear', align_corners=False)
            elif model_name == 'efficientnet_b5':
                input_x = F.interpolate(x, size=(456, 456), mode='bilinear', align_corners=False)
            else:
                input_x = x

            logits = classifier(input_x)
            individual_outputs[model_name] = logits
            ensemble_logits += logits

        # Average the logits
        ensemble_logits /= len(self.classifiers)

        if return_individual:
            return ensemble_logits, individual_outputs
        return ensemble_logits

def create_transforms(img_size=224, enable_clahe=False, augmentation_strength=0.2):
    """Create training and validation transforms for super-ensemble."""

    # Base transforms
    train_transforms = []
    val_transforms = []

    # Optional CLAHE
    if enable_clahe:
        train_transforms.append(CLAHETransform(clip_limit=2.0))
        val_transforms.append(CLAHETransform(clip_limit=2.0))

    # Conservative medical augmentation
    train_transforms.extend([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=10 * augmentation_strength),
        transforms.RandomHorizontalFlip(p=0.3 * augmentation_strength),
        transforms.ColorJitter(
            brightness=0.08 * augmentation_strength,
            contrast=0.08 * augmentation_strength,
            saturation=0.04 * augmentation_strength
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
    """Train a single model with memory optimization."""

    device = torch.device(config['device'])
    model = model.to(device)

    # Enable memory optimization
    if config['enable_memory_optimization']:
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True

    # Setup mixed precision training
    scaler = torch.cuda.amp.GradScaler() if config['mixed_precision'] else None

    # Setup loss function
    if config['enable_focal_loss']:
        criterion = FocalLoss(
            alpha=config['focal_alpha'],
            gamma=config['focal_gamma'],
            num_classes=config['num_classes']
        )
    else:
        criterion = nn.CrossEntropyLoss(
            weight=config.get('class_weights'),
            label_smoothing=config['label_smoothing']
        )

    # Setup optimizer with model-specific learning rates
    base_lr = config['learning_rate']
    if model_name == 'medsiglip_448':
        lr = base_lr * config['medsiglip_lr_multiplier']
    else:
        lr = base_lr * config['efficientnet_lr_multiplier']

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config['weight_decay'])

    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * config['epochs']
    warmup_steps = len(train_loader) * config['warmup_epochs']

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return max(0.1, (total_steps - step) / (total_steps - warmup_steps))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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

    logger.info(f"🏁 Training {model_name} with LR: {lr:.2e}")

    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Mixed precision forward pass
            if config['mixed_precision']:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Memory cleanup
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        train_acc = 100.0 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                if config['mixed_precision']:
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
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
        train_history['learning_rates'].append(scheduler.get_last_lr()[0])

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

        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()

        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            logger.info(f"   Early stopping at epoch {epoch+1}")
            break

    logger.info(f"✅ {model_name}: Best Val Acc = {best_val_acc:.2f}%")
    return best_val_acc, train_history

def train_super_ensemble(config):
    """Train the complete super-ensemble."""

    logger.info("🚀 Starting Super-Ensemble Training")
    logger.info("=" * 60)
    logger.info(f"🏥 MedSigLIP + EfficientNet B3/B4/B5")
    logger.info(f"💾 Memory optimization: {config['enable_memory_optimization']}")
    logger.info(f"🔧 Mixed precision: {config['mixed_precision']}")
    logger.info("=" * 60)

    # Check GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
        if gpu_memory < 15:
            logger.warning("⚠️ GPU memory < 16GB - consider reducing batch size")

    # Prepare data
    dataset_path = Path(config['dataset_path'])
    train_transform, val_transform = create_transforms(
        img_size=224,  # Base size, models will resize internally
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

    # Create data loaders with memory optimization
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,  # Reduced for memory
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Train individual models
    trained_models = {}
    training_results = {}

    for model_name in config['models']:
        logger.info(f"\n🏗️ Training {model_name}")

        # Create model
        if model_name == 'medsiglip_448':
            model = MedSigLIPClassifier(
                num_classes=config['num_classes'],
                dropout=config['dropout'],
                enable_checkpointing=config['gradient_checkpointing']
            )
        elif model_name.startswith('efficientnet'):
            model = EfficientNetClassifier(
                model_name=model_name,
                num_classes=config['num_classes'],
                dropout=config['dropout'],
                enable_checkpointing=config['gradient_checkpointing']
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Train model
        best_acc, history = train_single_model(
            model, train_loader, val_loader, config, model_name
        )

        trained_models[model_name] = model
        training_results[model_name] = {
            'best_accuracy': best_acc,
            'history': history
        }

        # Memory cleanup between models
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Create super-ensemble
    logger.info("\n🎯 Creating Super-Ensemble")
    super_ensemble = SuperEnsemble(
        models=config['models'],
        num_classes=config['num_classes'],
        dropout=config['dropout'],
        enable_checkpointing=config['gradient_checkpointing']
    )

    # Load best weights into ensemble
    for model_name in config['models']:
        model_path = Path(config['output_dir']) / "models" / f"best_{model_name}.pth"
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location='cpu')
            super_ensemble.classifiers[model_name].load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"✅ Loaded {model_name} weights")

    # Save super-ensemble
    ensemble_path = Path(config['output_dir']) / "models" / "super_ensemble_best.pth"
    torch.save(super_ensemble.state_dict(), ensemble_path)
    logger.info(f"💾 Super-ensemble saved: {ensemble_path}")

    # Evaluate super-ensemble
    ensemble_acc = evaluate_super_ensemble(super_ensemble, test_loader, config)

    return super_ensemble, training_results, ensemble_acc

def evaluate_super_ensemble(ensemble, test_loader, config):
    """Evaluate super-ensemble on test set."""

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
            if config['mixed_precision']:
                with torch.amp.autocast('cuda'):
                    ensemble_logits, individual_outputs = ensemble(images, return_individual=True)
            else:
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

    logger.info(f"\n🏆 SUPER-ENSEMBLE RESULTS:")
    logger.info(f"   Ensemble Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
    logger.info(f"   Medical Grade: {'✅ PASS' if ensemble_accuracy >= 0.90 else '❌ FAIL'}")

    logger.info(f"\n📊 INDIVIDUAL RESULTS:")
    for model, acc in individual_accuracies.items():
        logger.info(f"   {model}: {acc:.4f} ({acc*100:.2f}%)")

    return ensemble_accuracy

def main():
    """Main training function."""

    print("🚀 SUPER-ENSEMBLE DIRECT TRAINER")
    print("=" * 60)
    print("🏥 MedSigLIP-448 + EfficientNet B3/B4/B5")
    print("💾 Optimized for V100 16GB")
    print("🎯 Target: 92-96% Medical-Grade Accuracy")
    print("=" * 60)

    # Parse arguments
    args = parse_args()

    # Debug mode adjustments
    if args.debug_mode:
        args.epochs = 3
        args.models = ['efficientnet_b3']  # Single model for testing
        logger.info("🐛 Debug mode: 3 epochs, single model")

    # Create config
    config = {
        'dataset_path': args.dataset_path,
        'num_classes': args.num_classes,
        'models': args.models,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'warmup_epochs': args.warmup_epochs,
        'early_stopping_patience': args.early_stopping_patience,
        'reduce_lr_patience': args.reduce_lr_patience,
        'min_lr': args.min_lr,
        'medsiglip_lr_multiplier': args.medsiglip_lr_multiplier,
        'efficientnet_lr_multiplier': args.efficientnet_lr_multiplier,
        'enable_memory_optimization': args.enable_memory_optimization,
        'gradient_checkpointing': args.gradient_checkpointing,
        'mixed_precision': args.mixed_precision,
        'enable_clahe': args.enable_clahe,
        'augmentation_strength': args.augmentation_strength,
        'enable_focal_loss': args.enable_focal_loss,
        'focal_alpha': args.focal_alpha,
        'focal_gamma': args.focal_gamma,
        'enable_class_weights': args.enable_class_weights,
        'label_smoothing': args.label_smoothing,
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

    # Save config
    config_path = output_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    try:
        # Train super-ensemble
        ensemble, training_results, ensemble_acc = train_super_ensemble(config)

        logger.info(f"\n✅ Super-ensemble training completed!")
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

        # Final memory cleanup
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        logger.error(f"❌ Super-ensemble training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()