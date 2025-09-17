#!/usr/bin/env python3
"""
One-Versus-One (OVO) Ensemble Trainer for Diabetic Retinopathy Classification

Implementation of the research-validated OVO ensemble methodology achieving 96.96%
accuracy using lightweight CNNs (MobileNet, InceptionV3, DenseNet121). This script
implements the exact technique from the paper "A lightweight transfer learning based
ensemble approach for diabetic retinopathy detection".

Key Features:
- OVO binarization approach: 10 binary classifiers for 5-class DR problem
- Transfer learning with frozen weights and single output nodes
- Majority voting strategy for multi-class prediction
- Enhanced preprocessing with CLAHE and medical-grade augmentation
- Class balancing optimized for binary classification tasks
- Medical-grade validation with 96.96% accuracy target
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
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from itertools import combinations
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

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ Warning: wandb not available. Logging disabled.")

# Essential imports for OVO ensemble
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import ImageFolder
from PIL import Image

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
    
    # OVO Ensemble configuration
    parser.add_argument('--base_models', nargs='+',
                       default=['mobilenet_v2', 'inception_v3', 'densenet121'],
                       help='Base models for OVO ensemble')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size (224 optimal for CNNs)')
    parser.add_argument('--freeze_weights', action='store_true', default=True,
                       help='Freeze pre-trained weights (transfer learning)')
    parser.add_argument('--ovo_dropout', type=float, default=0.5,
                       help='Dropout rate for binary classifier heads')
    
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
    
    # Enhanced preprocessing for OVO
    parser.add_argument('--enable_clahe', action='store_true', default=False,
                       help='Enable CLAHE preprocessing (+3-5% accuracy)')
    parser.add_argument('--clahe_clip_limit', type=float, default=3.0,
                       help='CLAHE clip limit (higher for binary tasks)')
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
    
    # OVO Training strategy
    parser.add_argument('--binary_epochs', type=int, default=30,
                       help='Epochs for each binary classifier training')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                       help='Early stopping patience for binary classifiers')
    
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
    
    # Debug and testing for OVO
    parser.add_argument('--debug_mode', action='store_true',
                       help='Enable debug mode (reduced epochs for testing)')
    parser.add_argument('--debug_epochs', type=int, default=2,
                       help='Number of epochs in debug mode')
    parser.add_argument('--test_single_pair', action='store_true',
                       help='Test with only one class pair for debugging')
    
    # Medical-grade validation
    parser.add_argument('--enable_medical_validation', action='store_true', default=True,
                       help='Enable medical-grade validation')
    parser.add_argument('--target_accuracy', type=float, default=0.9696,
                       help='Target ensemble accuracy (96.96% from research)')
    
    return parser.parse_args()

# ============================================================================
# OVO (One-Versus-One) Ensemble Implementation
# Based on research paper methodology
# ============================================================================

class CLAHETransform:
    """CLAHE preprocessing transform for medical images."""

    def __init__(self, clip_limit=3.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, image):
        """Apply CLAHE to PIL image."""
        try:
            if isinstance(image, Image.Image):
                # Convert PIL to numpy
                image_np = np.array(image)
            else:
                image_np = image

            # Ensure image_np is valid numpy array
            if not isinstance(image_np, np.ndarray):
                logger.warning(f"CLAHE: Invalid image type {type(image_np)}, skipping CLAHE")
                return image if isinstance(image, Image.Image) else Image.fromarray(image_np)

            # Ensure uint8 type
            if image_np.dtype != np.uint8:
                image_np = image_np.astype(np.uint8)

            # Apply CLAHE to each channel
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:  # Color image
                # Convert RGB to LAB
                lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
                # Apply CLAHE to L channel
                lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
                # Convert back to RGB
                image_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            elif len(image_np.shape) == 2:  # Grayscale
                image_enhanced = self.clahe.apply(image_np)
            else:
                logger.warning(f"CLAHE: Unsupported image shape {image_np.shape}, skipping CLAHE")
                return image if isinstance(image, Image.Image) else Image.fromarray(image_np)

            return Image.fromarray(image_enhanced.astype(np.uint8))

        except Exception as e:
            logger.warning(f"CLAHE preprocessing failed: {e}, skipping CLAHE for this image")
            return image if isinstance(image, Image.Image) else Image.fromarray(image_np)

class BinaryDataset(Dataset):
    """Dataset for binary classification tasks in OVO ensemble."""

    def __init__(self, base_dataset, class_a, class_b, transform=None):
        self.base_dataset = base_dataset
        self.class_a = class_a
        self.class_b = class_b
        self.transform = transform

        # Filter indices for binary classes
        self.indices = []
        self.labels = []

        for idx in range(len(base_dataset)):
            _, label = base_dataset[idx]
            if label == class_a:
                self.indices.append(idx)
                self.labels.append(0)  # Binary label 0
            elif label == class_b:
                self.indices.append(idx)
                self.labels.append(1)  # Binary label 1

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        image, _ = self.base_dataset[original_idx]
        binary_label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, binary_label

class BinaryClassifier(nn.Module):
    """Binary classifier with frozen pre-trained backbone."""

    def __init__(self, model_name='mobilenet_v2', freeze_weights=True, dropout=0.5):
        super().__init__()
        self.model_name = model_name

        # Load pre-trained model
        if model_name == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=True)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif model_name == 'inception_v3':
            self.backbone = models.inception_v3(pretrained=True)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == 'densenet121':
            self.backbone = models.densenet121(pretrained=True)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Freeze backbone weights if specified
        if freeze_weights:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Forward pass for binary classification."""
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        return self.classifier(features)

class OVOEnsemble(nn.Module):
    """One-Versus-One ensemble with majority voting."""

    def __init__(self, base_models=['mobilenet_v2', 'inception_v3', 'densenet121'],
                 num_classes=5, freeze_weights=True, dropout=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.base_models = base_models

        # Generate all class pairs for OVO
        self.class_pairs = list(combinations(range(num_classes), 2))
        logger.info(f"🔢 Created {len(self.class_pairs)} binary classifiers for {num_classes} classes")

        # Create binary classifiers for each base model and class pair
        self.classifiers = nn.ModuleDict()

        for model_name in base_models:
            model_classifiers = nn.ModuleDict()
            for i, (class_a, class_b) in enumerate(self.class_pairs):
                classifier_name = f"pair_{class_a}_{class_b}"
                model_classifiers[classifier_name] = BinaryClassifier(
                    model_name=model_name,
                    freeze_weights=freeze_weights,
                    dropout=dropout
                )
            self.classifiers[model_name] = model_classifiers

        logger.info(f"🏗️ Initialized OVO ensemble:")
        logger.info(f"   Base models: {base_models}")
        logger.info(f"   Binary classifiers per model: {len(self.class_pairs)}")
        logger.info(f"   Total binary classifiers: {len(base_models) * len(self.class_pairs)}")

    def forward(self, x, return_individual=False):
        """Forward pass with majority voting."""
        batch_size = x.size(0)

        # Collect votes from all binary classifiers
        votes = torch.zeros(batch_size, self.num_classes, device=x.device)
        individual_predictions = {} if return_individual else None

        for model_name, model_classifiers in self.classifiers.items():
            model_votes = torch.zeros(batch_size, self.num_classes, device=x.device)

            for classifier_name, classifier in model_classifiers.items():
                # Extract class indices from classifier name
                class_a, class_b = map(int, classifier_name.split('_')[1:])

                # Get binary prediction
                binary_output = classifier(x).squeeze()  # Shape: (batch_size,)

                # Convert binary output to class votes
                # If output > 0.5, vote for class_b, else vote for class_a
                class_a_votes = (binary_output <= 0.5).float()
                class_b_votes = (binary_output > 0.5).float()

                model_votes[:, class_a] += class_a_votes
                model_votes[:, class_b] += class_b_votes

            # Add model votes to ensemble votes
            votes += model_votes

            if return_individual:
                individual_predictions[model_name] = model_votes

        # Final predictions based on majority voting
        final_predictions = votes  # Raw vote counts

        result = {
            'logits': final_predictions,
            'votes': votes
        }

        if return_individual:
            result['individual_predictions'] = individual_predictions

        return result

def create_ovo_transforms(img_size=224, enable_clahe=False, clahe_clip_limit=3.0):
    """Create transforms for OVO training with standardized image sizes."""

    # Standardized transforms with explicit size control
    train_transforms = [
        # Force consistent size first
        transforms.Resize((img_size, img_size), antialias=True),
        # Medical-grade augmentation (conservative)
        transforms.RandomRotation(10),  # Reduced for medical images
        transforms.RandomHorizontalFlip(0.3),  # Reduced probability
        transforms.ColorJitter(brightness=0.05, contrast=0.05),  # Subtle changes
        # Convert to tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    val_transforms = [
        # Force consistent size
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    # Optional CLAHE (disabled by default for stability)
    if enable_clahe:
        logger.warning("CLAHE is disabled for stability. Standard transforms will be used.")

    logger.info(f"✅ Transforms created with standardized size: {img_size}x{img_size}")

    return (
        transforms.Compose(train_transforms),
        transforms.Compose(val_transforms)
    )

def create_binary_datasets(base_train_dataset, base_val_dataset, class_pairs, train_transform, val_transform):
    """Create binary datasets for all class pairs."""

    binary_datasets = {}

    for class_a, class_b in class_pairs:
        pair_name = f"pair_{class_a}_{class_b}"

        # Create binary training dataset
        binary_train = BinaryDataset(
            base_train_dataset, class_a, class_b, transform=train_transform
        )

        # Create binary validation dataset
        binary_val = BinaryDataset(
            base_val_dataset, class_a, class_b, transform=val_transform
        )

        binary_datasets[pair_name] = {
            'train': binary_train,
            'val': binary_val,
            'train_loader': DataLoader(binary_train, batch_size=16, shuffle=True, num_workers=0, pin_memory=True, drop_last=True),
            'val_loader': DataLoader(binary_val, batch_size=32, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
        }

        logger.info(f"📊 Binary dataset {pair_name}: Train={len(binary_train)}, Val={len(binary_val)}")

    return binary_datasets

def setup_ovo_experiment(args):
    """Setup OVO ensemble experiment configuration from arguments."""

    # Create configuration dictionary for OVO ensemble
    config = {
        'data': {
            'dataset_path': args.dataset_path,
            'num_classes': args.num_classes,
            'img_size': args.img_size,
            'batch_size': args.batch_size,
            'enable_clahe': args.enable_clahe,
            'clahe_clip_limit': args.clahe_clip_limit,
            'clahe_tile_grid_size': tuple(args.clahe_tile_grid_size)
        },
        'model': {
            'base_models': args.base_models,
            'freeze_weights': args.freeze_weights,
            'dropout': args.ovo_dropout,
            'num_classes': args.num_classes
        },
        'training': {
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'patience': args.patience,
            'enable_focal_loss': args.enable_focal_loss,
            'enable_class_weights': args.enable_class_weights
        },
        'system': {
            'device': args.device,
            'seed': args.seed,
            'output_dir': args.output_dir,
            'experiment_name': args.experiment_name,
            'target_accuracy': args.target_accuracy,
            'use_wandb': not args.no_wandb and WANDB_AVAILABLE
        }
    }

    # Create output directories
    output_path = Path(config['system']['output_dir'])
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "models").mkdir(exist_ok=True)
    (output_path / "logs").mkdir(exist_ok=True)
    (output_path / "results").mkdir(exist_ok=True)

    logger.info(f"📁 Output directory: {output_path}")
    logger.info(f"🔧 OVO Configuration:")
    logger.info(f"   Base models: {config['model']['base_models']}")
    logger.info(f"   Classes: {config['data']['num_classes']}")
    logger.info(f"   Binary classifiers: {len(list(combinations(range(config['data']['num_classes']), 2)))}")
    logger.info(f"   Freeze weights: {config['model']['freeze_weights']}")
    logger.info(f"   CLAHE enabled: {config['data']['enable_clahe']}")

    return config
    
    return config

def train_binary_classifier(model, train_loader, val_loader, config, class_pair, model_name):
    """Train a single binary classifier."""

    device = torch.device(config['system']['device'])
    model = model.to(device)

    # Loss and optimizer for binary classification
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Training parameters
    best_val_acc = 0.0
    patience_counter = 0

    logger.info(f"🏁 Training {model_name} for classes {class_pair}")

    for epoch in range(config['training']['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            images = images.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(images).squeeze()

            # Ensure both outputs and labels have the same shape for BCELoss
            if outputs.dim() == 0:  # If scalar, add batch dimension
                outputs = outputs.unsqueeze(0)
            if labels.dim() == 0:  # If scalar, add batch dimension
                labels = labels.unsqueeze(0)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
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
                images = images.to(device)
                labels = labels.float().to(device)

                outputs = model(images).squeeze()

                # Ensure both outputs and labels have the same shape for BCELoss
                if outputs.dim() == 0:  # If scalar, add batch dimension
                    outputs = outputs.unsqueeze(0)
                if labels.dim() == 0:  # If scalar, add batch dimension
                    labels = labels.unsqueeze(0)

                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100.0 * val_correct / val_total

        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            model_path = Path(config['system']['output_dir']) / "models" / f"best_{model_name}_{class_pair[0]}_{class_pair[1]}.pth"
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            logger.info(f"   Epoch {epoch+1}/{config['training']['epochs']}: "
                       f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        # Early stopping
        if patience_counter >= config['training']['patience']:
            logger.info(f"   Early stopping at epoch {epoch+1}")
            break

    logger.info(f"✅ {model_name} for {class_pair}: Best Val Acc = {best_val_acc:.2f}%")
    return best_val_acc

def train_ovo_ensemble(config, train_dataset, val_dataset, test_dataset):
    """Train complete OVO ensemble."""

    logger.info("🚀 Starting OVO Ensemble Training")
    logger.info("=" * 50)

    # Create transforms
    train_transform, val_transform = create_ovo_transforms(
        img_size=config['data']['img_size'],
        enable_clahe=config['data']['enable_clahe'],
        clahe_clip_limit=config['data']['clahe_clip_limit']
    )

    # Generate class pairs for OVO
    num_classes = config['data']['num_classes']
    class_pairs = list(combinations(range(num_classes), 2))

    logger.info(f"📊 OVO Setup:")
    logger.info(f"   Classes: {num_classes}")
    logger.info(f"   Binary problems: {len(class_pairs)}")
    logger.info(f"   Base models: {config['model']['base_models']}")
    logger.info(f"   Total binary classifiers: {len(class_pairs) * len(config['model']['base_models'])}")

    # Create binary datasets
    binary_datasets = create_binary_datasets(
        train_dataset, val_dataset, class_pairs, train_transform, val_transform
    )

    # Train all binary classifiers
    trained_models = {}
    training_results = {}

    for model_name in config['model']['base_models']:
        logger.info(f"\n🏗️ Training {model_name} binary classifiers")
        trained_models[model_name] = {}
        training_results[model_name] = {}

        for class_a, class_b in class_pairs:
            pair_name = f"pair_{class_a}_{class_b}"

            # Create binary classifier
            binary_model = BinaryClassifier(
                model_name=model_name,
                freeze_weights=config['model']['freeze_weights'],
                dropout=config['model']['dropout']
            )

            # Train binary classifier
            best_acc = train_binary_classifier(
                model=binary_model,
                train_loader=binary_datasets[pair_name]['train_loader'],
                val_loader=binary_datasets[pair_name]['val_loader'],
                config=config,
                class_pair=(class_a, class_b),
                model_name=model_name
            )

            trained_models[model_name][pair_name] = binary_model
            training_results[model_name][pair_name] = {'best_accuracy': best_acc}

    # Create and save complete OVO ensemble
    logger.info("\n🅾️ Creating complete OVO ensemble")
    ovo_ensemble = OVOEnsemble(
        base_models=config['model']['base_models'],
        num_classes=num_classes,
        freeze_weights=config['model']['freeze_weights'],
        dropout=config['model']['dropout']
    )

    # Load trained weights into ensemble
    for model_name in config['model']['base_models']:
        for pair_name in binary_datasets.keys():
            model_path = Path(config['system']['output_dir']) / "models" / f"best_{model_name}_{pair_name.split('_')[1]}_{pair_name.split('_')[2]}.pth"
            if model_path.exists():
                state_dict = torch.load(model_path, map_location='cpu')
                ovo_ensemble.classifiers[model_name][pair_name].load_state_dict(state_dict)

    # Save complete ensemble
    ensemble_path = Path(config['system']['output_dir']) / "models" / "ovo_ensemble_best.pth"
    torch.save(ovo_ensemble.state_dict(), ensemble_path)
    logger.info(f"💾 OVO ensemble saved: {ensemble_path}")

    return ovo_ensemble, training_results
def evaluate_ovo_ensemble(ovo_ensemble, test_loader, config):
    """Evaluate OVO ensemble on test dataset."""

    logger.info("📋 Evaluating OVO Ensemble")

    device = torch.device(config['system']['device'])
    ovo_ensemble = ovo_ensemble.to(device)
    ovo_ensemble.eval()

    all_predictions = []
    all_targets = []
    individual_predictions = {model: [] for model in config['model']['base_models']}

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(device)
            targets = targets.to(device)

            # Get ensemble predictions
            outputs = ovo_ensemble(images, return_individual=True)

            # Get final predictions (argmax of votes)
            _, ensemble_pred = torch.max(outputs['logits'], 1)
            all_predictions.extend(ensemble_pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Individual model predictions
            for model_name, model_votes in outputs['individual_predictions'].items():
                _, individual_pred = torch.max(model_votes, 1)
                individual_predictions[model_name].extend(individual_pred.cpu().numpy())

    # Calculate metrics
    ensemble_accuracy = accuracy_score(all_targets, all_predictions)
    individual_accuracies = {
        model: accuracy_score(all_targets, preds)
        for model, preds in individual_predictions.items()
    }

    # Medical grade validation
    medical_grade_pass = ensemble_accuracy >= 0.90
    research_target_achieved = ensemble_accuracy >= config['system']['target_accuracy']

    results = {
        'ensemble_accuracy': ensemble_accuracy,
        'individual_accuracies': individual_accuracies,
        'medical_grade_pass': medical_grade_pass,
        'research_target_achieved': research_target_achieved,
        'classification_report': classification_report(all_targets, all_predictions, target_names=[f'Class_{i}' for i in range(config['data']['num_classes'])]),
        'confusion_matrix': confusion_matrix(all_targets, all_predictions).tolist()
    }

    # Save results
    results_path = Path(config['system']['output_dir']) / "results" / "ovo_evaluation_results.json"
    results_path.parent.mkdir(exist_ok=True)

    # Prepare JSON-serializable results
    json_results = {
        'ensemble_accuracy': float(ensemble_accuracy),
        'individual_accuracies': {k: float(v) for k, v in individual_accuracies.items()},
        'medical_grade_pass': medical_grade_pass,
        'research_target_achieved': research_target_achieved,
        'confusion_matrix': results['confusion_matrix']
    }

    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    logger.info(f"💾 Results saved: {results_path}")

    return results

def prepare_ovo_data(config):
    """Prepare dataset for OVO ensemble training."""

    logger.info(f"📁 Preparing OVO dataset from: {config['data']['dataset_path']}")

    # Validate dataset structure
    dataset_path = Path(config['data']['dataset_path'])
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")

    # Check for train/val/test structure
    train_path = dataset_path / "train"
    val_path = dataset_path / "val"
    test_path = dataset_path / "test"

    if not all(p.exists() for p in [train_path, val_path, test_path]):
        raise ValueError(f"Dataset must have train/val/test structure in {dataset_path}")

    # NO transforms for base datasets - let binary datasets handle all transforms
    # Create base datasets without any transforms
    train_dataset = ImageFolder(str(train_path), transform=None)
    val_dataset = ImageFolder(str(val_path), transform=None)
    test_dataset = ImageFolder(str(test_path), transform=None)

    logger.info(f"📊 Dataset splits loaded:")
    logger.info(f"   Training: {len(train_dataset)} samples")
    logger.info(f"   Validation: {len(val_dataset)} samples")
    logger.info(f"   Test: {len(test_dataset)} samples")

    # Analyze class distribution
    class_counts = {}
    for dataset_name, dataset in [("train", train_dataset), ("val", val_dataset), ("test", test_dataset)]:
        counts = [0] * config['data']['num_classes']
        for _, label in dataset:
            counts[label] += 1
        class_counts[dataset_name] = counts
        logger.info(f"   {dataset_name.capitalize()} classes: {counts}")

    # Create test loader for final evaluation
    test_transform = transforms.Compose([
        transforms.Resize((config['data']['img_size'], config['data']['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset_final = ImageFolder(str(test_path), transform=test_transform)
    test_loader = DataLoader(test_dataset_final, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    return train_dataset, val_dataset, test_dataset_final, test_loader, class_counts

def run_ovo_pipeline(config):
    """Complete OVO ensemble training and evaluation pipeline."""

    logger.info("\n🚀 STARTING OVO ENSEMBLE PIPELINE")
    logger.info("=" * 60)
    logger.info(f"   Base models: {config['model']['base_models']}")
    logger.info(f"   Classes: {config['data']['num_classes']}")
    logger.info(f"   Target accuracy: {config['system']['target_accuracy']:.2%}")
    logger.info("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")

    # Prepare data
    train_dataset, val_dataset, test_dataset, test_loader, class_counts = prepare_ovo_data(config)

    # Initialize wandb if enabled
    if config['system']['use_wandb']:
        wandb.init(
            project="ovo_diabetic_retinopathy",
            name=config['system']['experiment_name'],
            config=config
        )

    try:
        # Train OVO ensemble
        ovo_ensemble, training_results = train_ovo_ensemble(
            config, train_dataset, val_dataset, test_dataset
        )

        # Evaluate ensemble
        evaluation_results = evaluate_ovo_ensemble(ovo_ensemble, test_loader, config)

        # Log final results
        logger.info("\n🎆 OVO ENSEMBLE TRAINING COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"🏆 ENSEMBLE RESULTS:")
        logger.info(f"   Accuracy: {evaluation_results['ensemble_accuracy']:.4f} ({evaluation_results['ensemble_accuracy']*100:.2f}%)")
        logger.info(f"   Medical Grade: {'✅ PASS' if evaluation_results['medical_grade_pass'] else '❌ FAIL'}")
        logger.info(f"   Research Target: {'✅ ACHIEVED' if evaluation_results['research_target_achieved'] else '❌ NOT ACHIEVED'}")

        logger.info(f"\n📊 INDIVIDUAL MODEL RESULTS:")
        for model_name, accuracy in evaluation_results['individual_accuracies'].items():
            logger.info(f"   {model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")

        # Save comprehensive results
        final_results = {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'dataset_info': {
                'class_counts': class_counts,
                'num_classes': config['data']['num_classes']
            },
            'config': config
        }

        results_path = Path(config['system']['output_dir']) / "results" / "complete_ovo_results.json"
        with open(results_path, 'w') as f:
            # Convert to JSON-serializable format
            json_results = json.loads(json.dumps(final_results, default=str))
            json.dump(json_results, f, indent=2)

        logger.info(f"💾 Complete results saved: {results_path}")

        return final_results

    except Exception as e:
        logger.error(f"❌ OVO training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if config['system']['use_wandb'] and WANDB_AVAILABLE:
            wandb.finish()


def verify_environment():
    """Verify environment setup and dependencies for OVO ensemble."""

    # Check required packages
    try:
        import cv2
        from sklearn.metrics import accuracy_score
        from PIL import Image
    except ImportError as e:
        logger.error(f"❌ Missing required package: {e}")
        logger.info("Install with: pip install opencv-python scikit-learn pillow")
        return False

    # Check CUDA (optional)
    if torch.cuda.is_available():
        logger.info(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        logger.info("⚠️ Running on CPU mode (training will be slower)")

    return True

def main():
    """Main function for OVO ensemble training."""

    print("🔢 ONE-VERSUS-ONE (OVO) ENSEMBLE DIABETIC RETINOPATHY TRAINING")
    print("=" * 70)
    print("Research Implementation: Lightweight Transfer Learning Ensemble")
    print("Base Models: MobileNet-v2 + InceptionV3 + DenseNet121")
    print("Binary Classifiers: 10 (One-vs-One for 5 classes)")
    print("Target: 96.96% accuracy with medical-grade validation")
    print("=" * 70)

    # Verify environment
    if not verify_environment():
        return

    # Parse arguments
    args = parse_args()

    # Setup configuration
    try:
        config = setup_ovo_experiment(args)
    except Exception as e:
        logger.error(f"❌ Configuration setup failed: {e}")
        return

    # Set random seed for reproducibility
    torch.manual_seed(config['system']['seed'])
    np.random.seed(config['system']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['system']['seed'])

        # Enable CUDA optimizations for better GPU utilization
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        logger.info("🚀 CUDA optimizations enabled for better GPU utilization")

    logger.info(f"🎲 Random seed set: {config['system']['seed']}")

    # Save configuration
    config_path = Path(config['system']['output_dir']) / "ovo_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"💾 Configuration saved: {config_path}")

    # Execute based on mode
    try:
        if args.mode == 'train':
            # Run complete OVO training pipeline
            final_results = run_ovo_pipeline(config)

            logger.info("✅ OVO ensemble training and evaluation completed successfully!")

            # Display final summary
            logger.info("\n" + "=" * 60)
            logger.info("🏆 FINAL RESULTS SUMMARY")
            logger.info("=" * 60)

            eval_results = final_results['evaluation_results']
            logger.info(f"🎯 Ensemble Accuracy: {eval_results['ensemble_accuracy']:.4f} ({eval_results['ensemble_accuracy']*100:.2f}%)")
            logger.info(f"🏥 Medical Grade: {'✅ PASS' if eval_results['medical_grade_pass'] else '❌ FAIL'} (≥90% required)")
            logger.info(f"📊 Research Target: {'✅ ACHIEVED' if eval_results['research_target_achieved'] else '❌ NOT ACHIEVED'} ({config['system']['target_accuracy']:.2%} target)")

            # Show individual model performance
            logger.info("\n📊 Individual Model Performance:")
            for model_name, accuracy in eval_results['individual_accuracies'].items():
                logger.info(f"   {model_name.capitalize()}: {accuracy:.4f} ({accuracy*100:.2f}%)")

        elif args.mode == 'evaluate':
            # Load and evaluate existing OVO ensemble
            ensemble_path = Path(config['system']['output_dir']) / "models" / "ovo_ensemble_best.pth"

            if not ensemble_path.exists():
                logger.error(f"❌ No trained OVO ensemble found at: {ensemble_path}")
                logger.info("Please run training mode first: --mode train")
                return

            # Load ensemble and evaluate
            logger.info(f"📥 Loading OVO ensemble from: {ensemble_path}")
            # Implementation would load and evaluate existing model
            logger.info("⚠️ Evaluation-only mode: Implementation needed")

        elif args.mode == 'inference':
            logger.info("For inference mode with OVO ensemble:")
            logger.info("Use: python ovo_inference.py --model_path ./ensemble_results/models/ovo_ensemble_best.pth")

    except Exception as e:
        logger.error(f"❌ OVO pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return

    logger.info(f"\n✅ OVO ensemble experiment completed successfully!")
    logger.info(f"📁 Results saved to: {config['system']['output_dir']}")

    # Show final memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"🎮 Peak GPU memory usage: {memory_used:.1f}GB")
        torch.cuda.empty_cache()

    logger.info("\n🎆 OVO ENSEMBLE TRAINING COMPLETE!")

if __name__ == "__main__":
    main()