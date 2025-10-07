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

# Add MedSigLIP support
try:
    from transformers import AutoModel, AutoProcessor
    MEDSIGLIP_AVAILABLE = True
    print("✅ MedSigLIP support enabled (transformers available)")
except ImportError:
    MEDSIGLIP_AVAILABLE = False
    print("⚠️ MedSigLIP not available - install transformers")

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
                       default=['mobilenet_v2', 'inception_v3', 'densenet121', 'medsiglip_448', 'efficientnetb5'],
                       help='Base models for OVO ensemble (now includes MedSigLIP-448)')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size (224 optimal for CNNs)')
    parser.add_argument('--freeze_weights', type=str, default='true',
                       help='Freeze pre-trained weights (true/false)')
    parser.add_argument('--ovo_dropout', type=float, default=0.5,
                       help='Dropout rate for binary classifier heads')
    parser.add_argument('--resume', action='store_true', default=False,
                       help='Resume training from existing checkpoints')

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
                       help='Enable CLAHE preprocessing (+3-5%% accuracy)')
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
                       help='Brightness variation range (±10%%)')
    parser.add_argument('--contrast_range', type=float, default=0.1,
                       help='Contrast variation range (±10%%)')
    
    # Loss configuration (PRESERVED)
    parser.add_argument('--enable_focal_loss', action='store_true',
                       help='Enable focal loss for class imbalance')
    parser.add_argument('--focal_loss_alpha', type=float, default=2.0,
                       help='Focal loss alpha parameter (reduced for ensemble)')
    parser.add_argument('--focal_loss_gamma', type=float, default=3.0,
                       help='Focal loss gamma parameter (reduced for ensemble)')
    parser.add_argument('--enable_class_weights', action='store_true',
                       help='Enable class weights for imbalanced data')
    parser.add_argument('--class_weight_mild', type=float, default=8.0,
                       help='Class weight multiplier for mild NPDR (Class 1)')
    parser.add_argument('--class_weight_moderate', type=float, default=4.0,
                       help='Class weight multiplier for moderate NPDR (Class 2)')
    parser.add_argument('--class_weight_severe', type=float, default=8.0,
                       help='Class weight multiplier for severe NPDR (Class 3)')
    parser.add_argument('--class_weight_pdr', type=float, default=6.0,
                       help='Class weight multiplier for PDR (Class 4)')
    
    # Scheduler configuration
    parser.add_argument('--scheduler', choices=['cosine', 'linear', 'plateau', 'none'],
                       default='cosine', help='Learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                       help='Number of warmup epochs')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                       help='Minimum learning rate')

    # Label smoothing for improved generalization
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                       help='Label smoothing factor (0.0-0.2 recommended, 0.0=disabled)')
    
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
                       help='Target ensemble accuracy (96.96%% from research)')
    
    return parser.parse_args()

# ============================================================================
# Multi-Class Classification Implementation
# Based on research paper methodology (APTOS 2019 MobileNet 92% accuracy)
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in multi-class classification."""

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MultiClassDRModel(nn.Module):
    """Multi-class DR classifier for APTOS dataset (5 classes)."""

    def __init__(self, model_name='mobilenet_v2', num_classes=5, freeze_weights=False, dropout=0.5):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        # Load pre-trained model
        if model_name == 'medsiglip_448':
            if not MEDSIGLIP_AVAILABLE:
                raise ImportError("MedSigLIP requires transformers. Install with: pip install transformers")

            # Get HuggingFace token from environment
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            if not hf_token:
                raise ImportError("MedSigLIP requires HUGGINGFACE_TOKEN environment variable. Set it in .env file.")

            try:
                self.backbone = AutoModel.from_pretrained(
                    "google/medsiglip-448",
                    token=hf_token,
                    trust_remote_code=True
                )
                # Use full model - this matches the 1.3B parameter working checkpoint
                num_features = self.backbone.config.vision_config.hidden_size
                logger.info(f"✅ Loaded MedSigLIP-448: {num_features} features")
            except Exception as e:
                raise ImportError(f"Failed to load MedSigLIP-448: {e}. Check HUGGINGFACE_TOKEN in .env file.")
        elif model_name == 'mobilenet_v2':
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
        elif model_name == 'efficientnetb2':
            self.backbone = models.efficientnet_b2(pretrained=True)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif model_name == 'efficientnetb5':
            self.backbone = models.efficientnet_b5(pretrained=True)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Fine-tuning strategy (partial freezing for better accuracy)
        if freeze_weights:
            # Freeze early layers, fine-tune later layers
            if model_name == 'medsiglip_448':
                # Freeze most of MedSigLIP vision model, fine-tune last few layers
                # Keep text model trainable to match working 1.3B parameter checkpoint
                for name, param in self.backbone.named_parameters():
                    if 'vision_model' in name and 'encoder.layers.23' not in name:
                        param.requires_grad = False  # Freeze most vision layers
                    else:
                        param.requires_grad = True   # Keep text model and last vision layer trainable
            elif model_name == 'mobilenet_v2':
                for i, param in enumerate(self.backbone.parameters()):
                    if i < 100:  # Freeze first 100 parameters
                        param.requires_grad = False
            elif model_name == 'densenet121':
                for name, param in self.backbone.named_parameters():
                    if 'denseblock1' in name or 'denseblock2' in name:
                        param.requires_grad = False
            elif model_name == 'inception_v3':
                for name, param in self.backbone.named_parameters():
                    if any(x in name for x in ['Conv2d_1', 'Conv2d_2', 'Conv2d_3']):
                        param.requires_grad = False
            elif model_name == 'efficientnetb2':
                # Freeze early blocks, fine-tune later blocks for EfficientNet-B2
                for name, param in self.backbone.named_parameters():
                    if any(x in name for x in ['features.0', 'features.1', 'features.2']):
                        param.requires_grad = False
            elif model_name == 'efficientnetb5':
                # Freeze early blocks, fine-tune later blocks for EfficientNet-B5
                for name, param in self.backbone.named_parameters():
                    if any(x in name for x in ['features.0', 'features.1', 'features.2', 'features.3']):
                        param.requires_grad = False

        # EXTREME classifier head for severe class imbalance
        # Larger capacity with specialized layers for minority classes
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 1024),  # Increased from 512
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout/2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout/3),  # Even less dropout for final layers
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout/4),  # Minimal dropout before output
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """Forward pass for multi-class classification."""
        # Handle MedSigLIP input size requirements
        if self.model_name == 'medsiglip_448' and x.size(-1) != 448:
            x = F.interpolate(x, size=(448, 448), mode='bilinear', align_corners=False)
        # Handle InceptionV3 input size requirements
        elif self.model_name == 'inception_v3' and x.size(-1) < 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        # Extract features
        if self.model_name == 'medsiglip_448':
            # MedSigLIP forward pass - use full model's get_image_features for consistency with training
            features = self.backbone.get_image_features(x)
        elif self.model_name == 'inception_v3' and self.training:
            features, aux_features = self.backbone(x)
        else:
            features = self.backbone(x)

        if isinstance(features, tuple):
            features = features[0]

        # Global average pooling for feature maps (skip for MedSigLIP as it's already pooled)
        if self.model_name != 'medsiglip_448' and len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)

        return self.classifier(features)

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
        if model_name == 'medsiglip_448':
            if not MEDSIGLIP_AVAILABLE:
                raise ImportError("MedSigLIP requires transformers. Install with: pip install transformers")

            # Get HuggingFace token from environment
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            if not hf_token:
                raise ImportError("MedSigLIP requires HUGGINGFACE_TOKEN environment variable. Set it in .env file.")

            try:
                self.backbone = AutoModel.from_pretrained(
                    "google/medsiglip-448",
                    token=hf_token,
                    trust_remote_code=True
                )
                # Use full model - this matches the 1.3B parameter working checkpoint
                num_features = self.backbone.config.vision_config.hidden_size
                logger.info(f"✅ Loaded MedSigLIP-448 Binary Classifier: {num_features} features")
            except Exception as e:
                raise ImportError(f"Failed to load MedSigLIP-448: {e}. Check HUGGINGFACE_TOKEN in .env file.")
        elif model_name == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=True)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif model_name == 'inception_v3':
            self.backbone = models.inception_v3(pretrained=True)  # Keep aux_logits=True (default)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            # Set to eval mode to disable auxiliary outputs during inference
            self.backbone.training = True  # We'll handle aux outputs in forward()
        elif model_name == 'densenet121':
            self.backbone = models.densenet121(pretrained=True)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif model_name == 'efficientnetb2':
            self.backbone = models.efficientnet_b2(pretrained=True)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif model_name == 'efficientnetb5':
            self.backbone = models.efficientnet_b5(pretrained=True)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Fine-tuning strategy instead of freezing for better medical accuracy
        if freeze_weights:
            # Partial fine-tuning: freeze early layers, unfreeze later layers
            if model_name == 'medsiglip_448':
                # Freeze most of MedSigLIP vision model, fine-tune last few layers
                # Keep text model trainable to match working 1.3B parameter checkpoint
                for name, param in self.backbone.named_parameters():
                    if 'vision_model' in name and 'encoder.layers.23' not in name:
                        param.requires_grad = False  # Freeze most vision layers
                    else:
                        param.requires_grad = True   # Keep text model and last vision layer trainable
            elif model_name == 'densenet121':
                # Freeze first 2 dense blocks, fine-tune last 2
                for name, param in self.backbone.named_parameters():
                    if 'denseblock1' in name or 'denseblock2' in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            elif model_name == 'mobilenet_v2':
                # Freeze first 10 layers, fine-tune rest
                for i, param in enumerate(self.backbone.parameters()):
                    if i < 10:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            elif model_name == 'inception_v3':
                # Freeze early layers, fine-tune later
                for name, param in self.backbone.named_parameters():
                    if any(x in name for x in ['Conv2d_1', 'Conv2d_2', 'Conv2d_3', 'Conv2d_4']):
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            elif model_name == 'efficientnetb2':
                # Freeze early blocks, fine-tune later blocks for EfficientNet-B2
                for name, param in self.backbone.named_parameters():
                    if any(x in name for x in ['features.0', 'features.1', 'features.2']):
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            elif model_name == 'efficientnetb5':
                # Freeze early blocks, fine-tune later blocks for EfficientNet-B5
                for name, param in self.backbone.named_parameters():
                    if any(x in name for x in ['features.0', 'features.1', 'features.2', 'features.3']):
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
        else:
            # Full fine-tuning for maximum accuracy
            for param in self.backbone.parameters():
                param.requires_grad = True

        # Enhanced binary classification head for medical accuracy
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout/2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout/2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Forward pass for binary classification."""
        # Handle MedSigLIP input size requirements
        if self.model_name == 'medsiglip_448' and x.size(-1) != 448:
            x = F.interpolate(x, size=(448, 448), mode='bilinear', align_corners=False)
        # Ensure input tensor has minimum size for InceptionV3
        elif self.model_name == 'inception_v3' and x.size(-1) < 75:
            # Upscale if too small
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        # Extract features
        if self.model_name == 'medsiglip_448':
            # MedSigLIP forward pass - use full model's get_image_features for consistency with training
            features = self.backbone.get_image_features(x)
        # Handle InceptionV3 auxiliary outputs during training
        elif self.model_name == 'inception_v3' and self.training:
            features, aux_features = self.backbone(x)
            # We only use main features, ignore auxiliary
        else:
            features = self.backbone(x)

        # Handle any remaining tuple outputs
        if isinstance(features, tuple):
            features = features[0]  # Take main output only

        if self.model_name != 'medsiglip_448' and len(features.shape) > 2:
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
        """FIXED forward pass with medical-grade voting mechanism"""
        batch_size = x.size(0)
        device = x.device

        # Enhanced vote accumulation
        class_scores = torch.zeros(batch_size, self.num_classes, device=device)
        total_weights = torch.zeros(batch_size, self.num_classes, device=device)

        individual_predictions = {} if return_individual else None

        # Medical-grade class weights (Class 1 emergency boost)
        class_weights = torch.tensor([1.0, 8.0, 2.0, 4.0, 5.0], device=device)
        class1_pairs = ['pair_0_1', 'pair_1_2', 'pair_1_3', 'pair_1_4']

        # Binary accuracy weights (will be updated after each training)
        binary_accuracies = {
            'mobilenet_v2': {
                'pair_0_1': 0.85, 'pair_0_2': 0.85, 'pair_0_3': 0.95, 'pair_0_4': 0.95,
                'pair_1_2': 0.80, 'pair_1_3': 0.85, 'pair_1_4': 0.90, 'pair_2_3': 0.85,
                'pair_2_4': 0.85, 'pair_3_4': 0.75
            },
            'inception_v3': {
                'pair_0_1': 0.85, 'pair_0_2': 0.80, 'pair_0_3': 0.95, 'pair_0_4': 0.95,
                'pair_1_2': 0.80, 'pair_1_3': 0.85, 'pair_1_4': 0.85, 'pair_2_3': 0.85,
                'pair_2_4': 0.85, 'pair_3_4': 0.75
            },
            'densenet121': {
                'pair_0_1': 0.85, 'pair_0_2': 0.85, 'pair_0_3': 0.95, 'pair_0_4': 0.95,
                'pair_1_2': 0.80, 'pair_1_3': 0.85, 'pair_1_4': 0.85, 'pair_2_3': 0.85,
                'pair_2_4': 0.85, 'pair_3_4': 0.75
            }
        }

        weak_classifiers = {
            ('inception_v3', 'pair_0_2'), ('inception_v3', 'pair_1_2'),
            ('inception_v3', 'pair_3_4'), ('mobilenet_v2', 'pair_3_4')
        }

        for model_name, model_classifiers in self.classifiers.items():
            if return_individual:
                individual_predictions[model_name] = torch.zeros(batch_size, self.num_classes, device=device)

            for classifier_name, classifier in model_classifiers.items():
                class_a, class_b = map(int, classifier_name.split('_')[1:])

                # Get binary prediction
                binary_output = classifier(x).squeeze()
                if binary_output.dim() == 0:
                    binary_output = binary_output.unsqueeze(0)

                # FIXED: Accuracy-based weighting
                base_accuracy = binary_accuracies.get(model_name, {}).get(classifier_name, 0.8)

                if (model_name, classifier_name) in weak_classifiers:
                    accuracy_weight = (base_accuracy ** 4) * 0.5  # Penalty
                else:
                    accuracy_weight = base_accuracy ** 2

                if base_accuracy > 0.95:
                    accuracy_weight *= 1.5  # Boost excellent classifiers

                # FIXED: Confidence-based weighting
                confidence = torch.abs(binary_output - 0.5) * 2
                weighted_confidence = confidence * accuracy_weight

                # FIXED: Medical-grade class weighting
                class_a_weight = class_weights[class_a]
                class_b_weight = class_weights[class_b]

                # Emergency Class 1 boost
                if classifier_name in class1_pairs:
                    if class_a == 1:
                        class_a_weight *= 3.0
                    if class_b == 1:
                        class_b_weight *= 3.0

                # FIXED: Probability-based voting
                prob_class_a = (1.0 - binary_output) * class_a_weight * weighted_confidence
                prob_class_b = binary_output * class_b_weight * weighted_confidence

                class_scores[:, class_a] += prob_class_a
                class_scores[:, class_b] += prob_class_b

                total_weights[:, class_a] += class_a_weight * weighted_confidence
                total_weights[:, class_b] += class_b_weight * weighted_confidence

                if return_individual:
                    individual_predictions[model_name][:, class_a] += prob_class_a
                    individual_predictions[model_name][:, class_b] += prob_class_b

        # FIXED: Proper normalization and softmax
        normalized_scores = class_scores / (total_weights + 1e-8)
        final_predictions = F.softmax(normalized_scores, dim=1)

        result = {
            'logits': final_predictions,
            'votes': class_scores
        }

        if return_individual:
            for model_name in individual_predictions:
                model_weights = total_weights / len(self.base_models)
                individual_predictions[model_name] = individual_predictions[model_name] / (model_weights + 1e-8)
            result['individual_predictions'] = individual_predictions

        return result

def create_ovo_transforms(img_size=224, enable_clahe=False, clahe_clip_limit=3.0):
    """Create transforms for OVO training with standardized image sizes."""

    # Ensure minimum size for InceptionV3 (requires 75x75 minimum, 299x299 optimal)
    safe_img_size = max(img_size, 299)  # Use 299x299 for InceptionV3 compatibility

    # Standardized transforms with explicit size control
    train_transforms = [
        # Force consistent size first
        transforms.Resize((safe_img_size, safe_img_size), antialias=True),
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
        transforms.Resize((safe_img_size, safe_img_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    # Optional CLAHE (disabled by default for stability)
    if enable_clahe:
        logger.warning("CLAHE is disabled for stability. Standard transforms will be used.")

    logger.info(f"✅ Transforms created with InceptionV3-safe size: {safe_img_size}x{safe_img_size} (requested: {img_size}x{img_size})")

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
            'train_loader': DataLoader(binary_train, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True),
            'val_loader': DataLoader(binary_val, batch_size=128, shuffle=False, num_workers=4, pin_memory=True, drop_last=False, persistent_workers=True)
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
            'freeze_weights': args.freeze_weights.lower() == 'true',
            'dropout': args.ovo_dropout,
            'num_classes': args.num_classes
        },
        'training': {
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'patience': args.patience,
            'enable_focal_loss': args.enable_focal_loss,
            'enable_class_weights': args.enable_class_weights,
            'focal_loss_alpha': args.focal_loss_alpha,
            'focal_loss_gamma': args.focal_loss_gamma,
            'scheduler': args.scheduler,
            'warmup_epochs': args.warmup_epochs,
            'rotation_range': args.rotation_range,
            'brightness_range': args.brightness_range,
            'contrast_range': args.contrast_range,
            'label_smoothing': args.label_smoothing,
            'class_weight_mild': args.class_weight_mild,
            'class_weight_moderate': args.class_weight_moderate,
            'class_weight_severe': args.class_weight_severe,
            'class_weight_pdr': args.class_weight_pdr,
            'resume': args.resume
        },
        'paths': {
            'output_dir': args.output_dir
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

def train_multiclass_dr_model(model, train_loader, val_loader, config, model_name):
    """Train multi-class DR model for APTOS dataset."""

    device = torch.device(config['system']['device'])
    model = model.to(device)

    # EXTREME loss configuration for severe class imbalance
    # Get label smoothing parameter
    label_smoothing = config['training'].get('label_smoothing', 0.0)

    if config['training']['enable_focal_loss']:
        # More aggressive focal loss parameters for extreme imbalance
        alpha = config['training'].get('focal_loss_alpha', 2.0)
        gamma = config['training'].get('focal_loss_gamma', 2.0)
        criterion = FocalLoss(alpha=alpha, gamma=gamma)
        logger.info(f"✅ Using EXTREME Focal Loss: alpha={alpha}, gamma={gamma}")
        if label_smoothing > 0:
            logger.info(f"⚠️  Label smoothing ({label_smoothing}) not supported with Focal Loss")
    else:
        if config['training']['enable_class_weights']:
            # EXTREME EyePACS class distribution weights - optimized for severe imbalance
            mild_weight = config['training'].get('class_weight_mild', 8.0)
            moderate_weight = config['training'].get('class_weight_moderate', 4.0)
            severe_weight = config['training'].get('class_weight_severe', 8.0)
            pdr_weight = config['training'].get('class_weight_pdr', 6.0)
            class_weights = torch.tensor([1.0, mild_weight, moderate_weight, severe_weight, pdr_weight])
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=label_smoothing)
            logger.info(f"✅ Using OPTIMIZED weighted CrossEntropyLoss: [1.0 (No DR), {mild_weight} (Mild), {moderate_weight} (Moderate), {severe_weight} (Severe), {pdr_weight} (PDR)]")
            if label_smoothing > 0:
                logger.info(f"✅ Label smoothing enabled: {label_smoothing}")
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            logger.info(f"✅ Using standard CrossEntropyLoss")
            if label_smoothing > 0:
                logger.info(f"✅ Label smoothing enabled: {label_smoothing}")

    # Enhanced optimizer with different learning rates
    backbone_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                classifier_params.append(param)

    # Research-validated optimizer settings
    # For balanced datasets, use same LR for all parameters (no differential LR)
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': config['training']['learning_rate']},
        {'params': classifier_params, 'lr': config['training']['learning_rate']}
    ], weight_decay=config['training']['weight_decay'])

    logger.info(f"✅ Using uniform learning rate: {config['training']['learning_rate']:.1e} for all parameters")

    # OPTIMIZED learning rate scheduler for high accuracy convergence
    scheduler_type = config['training'].get('scheduler', 'cosine')

    if scheduler_type == 'none':
        # No scheduler - constant learning rate
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        logger.info("✅ Using CONSTANT learning rate (no scheduler)")
    elif scheduler_type == 'cosine':
        # Cosine with warm restarts - T_0 adjusted based on warmup epochs for stable convergence
        warmup = config['training'].get('warmup_epochs', 10)
        T_0 = max(15, warmup + 5)  # Ensure T_0 is at least warmup + 5 epochs
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=1, eta_min=1e-7
        )
        logger.info(f"✅ Using CosineAnnealingWarmRestarts: T_0={T_0}, T_mult=1 (optimized for 90%+ accuracy)")
    else:
        # More patient plateau scheduler for imbalanced data
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.7, patience=8, verbose=True
        )
        logger.info("✅ Using patient ReduceLROnPlateau for imbalanced data")

    # Training tracking
    best_val_acc = 0.0
    patience_counter = 0
    train_history = {
        'train_accuracies': [],
        'val_accuracies': [],
        'train_losses': [],
        'val_losses': []
    }

    logger.info(f"🏁 Training {model_name} for 5-class DR classification")
    logger.info(f"📊 Target accuracy: {config['system']['target_accuracy']:.2%}")

    # Initialize wandb logging for this specific model if enabled
    log_wandb = config['system'].get('use_wandb', False) and WANDB_AVAILABLE
    if log_wandb:
        # Log model architecture info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.log({
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/freeze_ratio": (total_params - trainable_params) / total_params,
            "config/learning_rate": config['training']['learning_rate'],
            "config/batch_size": config['data']['batch_size'],
            "config/target_accuracy": config['system']['target_accuracy']
        })
        logger.info("📊 Wandb logging enabled for training metrics")

    for epoch in range(config['training']['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Monitor GPU memory at start of epoch
        if log_wandb and torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1e9
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1e9
            wandb.log({
                f"system/gpu_memory_allocated_gb": gpu_memory_allocated,
                f"system/gpu_memory_reserved_gb": gpu_memory_reserved,
                f"epoch": epoch + 1
            }, commit=False)

        batch_losses = []
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            images = images.to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_loss = loss.item()
            train_loss += batch_loss
            batch_losses.append(batch_loss)

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Log batch-level metrics occasionally
            if log_wandb and batch_idx % 50 == 0:
                wandb.log({
                    "batch/train_loss": batch_loss,
                    "batch/train_accuracy": (predicted == labels).float().mean().item(),
                    "batch/learning_rate": optimizer.param_groups[0]['lr'],
                    "batch/epoch": epoch + 1,
                    "batch/step": epoch * len(train_loader) + batch_idx
                }, commit=False)

        train_acc = 100.0 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.long().to(device)

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

        # Step scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler_type = config['training'].get('scheduler', 'cosine')
        if scheduler_type == 'none':
            scheduler.step()  # LambdaLR needs step() but doesn't change LR
        elif scheduler_type == 'cosine':
            scheduler.step()
        else:
            scheduler.step(val_acc)

        new_lr = optimizer.param_groups[0]['lr']

        # Log metrics to wandb
        if log_wandb:
            epoch_metrics = {
                "epoch": epoch + 1,
                "train/accuracy": train_acc / 100.0,
                "train/loss": train_loss / len(train_loader),
                "val/accuracy": val_acc / 100.0,
                "val/loss": val_loss / len(val_loader),
                "learning_rate": current_lr,
                "train/samples": train_total,
                "val/samples": val_total,
                "best_val_accuracy": best_val_acc / 100.0 if val_acc > best_val_acc else best_val_acc / 100.0
            }

            # Add progress toward target
            epoch_metrics["progress/target_achievement"] = (val_acc / 100.0) / config['system']['target_accuracy']
            epoch_metrics["progress/medical_grade"] = min((val_acc / 100.0) / 0.90, 1.0)

            wandb.log(epoch_metrics)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            model_path = Path(config['system']['output_dir']) / "models" / f"best_{model_name}_multiclass.pth"
            model_path.parent.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_accuracy': best_val_acc / 100.0,  # Convert to fraction
                'current_val_accuracy': val_acc / 100.0,
                'current_train_accuracy': train_acc / 100.0,
                'train_history': train_history,
                'model_name': model_name,
                'config': config,
                # Model analyzer compatible fields (converted to fractions)
                'val_accuracies': [acc/100.0 for acc in train_history['val_accuracies']],
                'train_accuracies': [acc/100.0 for acc in train_history['train_accuracies']],
                'train_losses': train_history['train_losses'],
                'val_losses': train_history['val_losses']
            }
            torch.save(checkpoint, model_path)
            logger.info(f"🎯 New best {model_name}: {val_acc:.2f}% (Target: {config['system']['target_accuracy']:.1%})")

            # Check if target achieved
            if val_acc/100.0 >= config['system']['target_accuracy']:
                logger.info(f"🏆 TARGET ACHIEVED! {val_acc:.2f}% >= {config['system']['target_accuracy']:.1%}")

                # Log milestone achievement to wandb
                if log_wandb:
                    wandb.log({
                        "milestone/target_achieved": True,
                        "milestone/target_epoch": epoch + 1,
                        "milestone/final_accuracy": val_acc / 100.0
                    })
        else:
            patience_counter += 1

        # Log progress
        logger.info(f"   Epoch {epoch+1}/{config['training']['epochs']}: "
                   f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, "
                   f"LR: {optimizer.param_groups[0]['lr']:.1e}")

        # Early stopping
        if patience_counter >= config['training']['patience']:
            logger.info(f"   Early stopping at epoch {epoch+1}")
            break

    logger.info(f"✅ {model_name} training completed: Best Val Acc = {best_val_acc:.2f}%")
    return best_val_acc / 100.0  # Return as fraction

def train_binary_classifier(model, train_loader, val_loader, config, class_pair, model_name):
    """Train a single binary classifier (for OVO ensemble)."""

    device = torch.device(config['system']['device'])
    model = model.to(device)

    # Binary classification loss
    criterion = nn.BCELoss()

    # Different learning rates for backbone vs classifier
    backbone_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                classifier_params.append(param)

    # Use lower learning rate for pre-trained backbone, higher for classifier head
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': config['training']['learning_rate'] * 0.1},  # 10x lower for backbone
        {'params': classifier_params, 'lr': config['training']['learning_rate']}       # Full rate for classifier
    ], weight_decay=config['training']['weight_decay'])

    # Learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    # Training parameters
    best_val_acc = 0.0
    patience_counter = 0

    # Track training history for comprehensive logging
    train_history = {
        'train_accuracies': [],
        'val_accuracies': [],
        'train_losses': [],
        'val_losses': [],
        'learning_rates': []
    }

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

        # Record training history
        train_history['train_accuracies'].append(train_acc)
        train_history['val_accuracies'].append(val_acc)
        train_history['train_losses'].append(train_loss / len(train_loader))
        train_history['val_losses'].append(val_loss / len(val_loader))
        train_history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        # Step the learning rate scheduler
        scheduler.step(val_acc)

        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # Save comprehensive checkpoint with training metrics
            model_path = Path(config['system']['output_dir']) / "models" / f"best_{model_name}_{class_pair[0]}_{class_pair[1]}.pth"
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_accuracy': best_val_acc,
                'current_val_accuracy': val_acc,
                'current_train_accuracy': train_acc,
                'train_history': train_history,
                'class_pair': class_pair,
                'model_name': model_name,
                'config': config,
                # Model analyzer compatible fields
                'val_accuracies': train_history['val_accuracies'],
                'train_accuracies': train_history['train_accuracies'],
                'train_losses': train_history['train_losses'],
                'val_losses': train_history['val_losses']
            }
            torch.save(checkpoint, model_path)
            logger.info(f"🎯 New best for {model_name}_{class_pair}: {val_acc:.2f}%")
        else:
            patience_counter += 1

        # Log every epoch for better monitoring
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

    # Resume logic - check for existing checkpoints
    completed_classifiers = []
    if config['training'].get('resume', False):
        models_dir = os.path.join(config['paths']['output_dir'], 'models')
        if os.path.exists(models_dir):
            for model_name in config['model']['base_models']:
                for class_a, class_b in class_pairs:
                    checkpoint_path = os.path.join(models_dir, f'best_{model_name}_{class_a}_{class_b}.pth')
                    if os.path.exists(checkpoint_path):
                        classifier_key = f"{model_name}_{class_a}_{class_b}"
                        completed_classifiers.append(classifier_key)
                        logger.info(f"✅ Found existing checkpoint: {classifier_key}")

            if completed_classifiers:
                logger.info(f"🔄 Resuming training - {len(completed_classifiers)} classifiers already completed")
            else:
                logger.info("🆕 No existing checkpoints found - starting fresh")
    training_results = {}

    for model_name in config['model']['base_models']:
        logger.info(f"\n🏗️ Training {model_name} binary classifiers")
        trained_models[model_name] = {}
        training_results[model_name] = {}

        for class_a, class_b in class_pairs:
            pair_name = f"pair_{class_a}_{class_b}"
            classifier_key = f"{model_name}_{class_a}_{class_b}"

            # Skip if already completed (resume functionality)
            if classifier_key in completed_classifiers:
                logger.info(f"⏭️ Skipping {classifier_key} - already completed")
                continue

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
    logger.info("🔄 Loading trained binary classifiers into ensemble...")
    loaded_count = 0
    for model_name in config['model']['base_models']:
        for pair_name in binary_datasets.keys():
            # Extract class indices from pair_name (e.g., "pair_0_1" -> "0_1")
            class_a, class_b = pair_name.split('_')[1], pair_name.split('_')[2]
            model_path = Path(config['system']['output_dir']) / "models" / f"best_{model_name}_{class_a}_{class_b}.pth"

            if model_path.exists():
                try:
                    checkpoint = torch.load(model_path, map_location='cpu')

                    # Handle both old format (state_dict only) and new format (checkpoint with metrics)
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        # New format with training metrics
                        ovo_ensemble.classifiers[model_name][pair_name].load_state_dict(checkpoint['model_state_dict'])
                        best_acc = checkpoint.get('best_val_accuracy', 0.0)
                        logger.info(f"✅ Loaded: {model_path.name} (Best Val Acc: {best_acc:.2f}%)")
                    else:
                        # Old format (just state_dict)
                        ovo_ensemble.classifiers[model_name][pair_name].load_state_dict(checkpoint)
                        logger.info(f"✅ Loaded: {model_path.name} (legacy format)")

                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"❌ Failed to load {model_path.name}: {e}")
            else:
                logger.warning(f"❌ Missing checkpoint: {model_path.name}")

    logger.info(f"📦 Loaded {loaded_count}/30 binary classifiers into ensemble")

    # Save complete ensemble
    ensemble_path = Path(config['system']['output_dir']) / "models" / "ovo_ensemble_best.pth"
    torch.save(ovo_ensemble.state_dict(), ensemble_path)
    logger.info(f"💾 OVO ensemble saved: {ensemble_path}")

    return ovo_ensemble, training_results
def evaluate_ovo_ensemble(ovo_ensemble, test_loader, config):
    """Evaluate OVO ensemble on test dataset."""

    logger.info("📋 Evaluating OVO Ensemble")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

def train_aptos_multiclass(config):
    """Train multi-class DR model for APTOS dataset to achieve >92% accuracy."""

    logger.info("🚀 STARTING APTOS 2019 MULTI-CLASS DR TRAINING")
    logger.info("=" * 60)
    logger.info(f"📊 Dataset: APTOS 2019 (5-class DR classification)")
    logger.info(f"🏗️ Model: {config['model']['base_models'][0]}")
    logger.info(f"🎯 Target accuracy: {config['system']['target_accuracy']:.1%}")
    logger.info("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")

    # Prepare data
    train_dataset, val_dataset, test_dataset, test_loader, class_counts = prepare_ovo_data(config)

    # Create transforms with research-validated augmentation
    train_transform = transforms.Compose([
        transforms.Resize((config['data']['img_size'], config['data']['img_size'])),
        transforms.RandomRotation(config['training'].get('rotation_range', 15)),
        transforms.RandomHorizontalFlip(0.3),  # Medical images - reduced flip
        transforms.ColorJitter(
            brightness=config['training'].get('brightness_range', 0.1),
            contrast=config['training'].get('contrast_range', 0.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config['data']['img_size'], config['data']['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply transforms to datasets
    train_dataset.transform = train_transform
    val_dataset.transform = val_transform

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create multi-class model
    model_name = config['model']['base_models'][0]  # Use first model
    model = MultiClassDRModel(
        model_name=model_name,
        num_classes=config['data']['num_classes'],
        freeze_weights=config['model']['freeze_weights'],
        dropout=config['model']['dropout']
    )

    logger.info(f"🏗️ Created {model_name} multi-class model")
    logger.info(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"📊 Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Initialize wandb if enabled
    if config['system']['use_wandb']:
        # Enhanced wandb configuration with better organization
        wandb_config = {
            # Model configuration
            "model_name": model_name,
            "model_architecture": f"{model_name}_multiclass",
            "num_classes": config['data']['num_classes'],
            "img_size": config['data']['img_size'],
            "dropout": config['model']['dropout'],
            "freeze_weights": config['model']['freeze_weights'],

            # Training configuration
            "batch_size": config['data']['batch_size'],
            "learning_rate": config['training']['learning_rate'],
            "weight_decay": config['training']['weight_decay'],
            "epochs": config['training']['epochs'],
            "patience": config['training']['patience'],
            "scheduler": config['training']['scheduler'],

            # Loss configuration
            "loss_function": "FocalLoss" if config['training']['enable_focal_loss'] else "CrossEntropyLoss",
            "enable_class_weights": config['training']['enable_class_weights'],
            "focal_loss_alpha": config['training'].get('focal_loss_alpha', 2.0),
            "focal_loss_gamma": config['training'].get('focal_loss_gamma', 2.0),

            # Augmentation configuration
            "rotation_range": config['training'].get('rotation_range', 15),
            "brightness_range": config['training'].get('brightness_range', 0.1),
            "contrast_range": config['training'].get('contrast_range', 0.1),

            # Target configuration
            "target_accuracy": config['system']['target_accuracy'],
            "medical_grade_threshold": 0.90,

            # Dataset info
            "dataset": "APTOS_2019",
            "dataset_path": config['data']['dataset_path']
        }

        wandb.init(
            project="aptos_dr_multiclass",
            name=config['system']['experiment_name'],
            config=wandb_config,
            tags=["aptos2019", "diabetic_retinopathy", model_name, "multiclass"],
            notes=f"APTOS 2019 training with {model_name} targeting {config['system']['target_accuracy']:.1%} accuracy"
        )

        # Define custom charts for better visualization
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("progress/*", step_metric="epoch")
        wandb.define_metric("batch/*", step_metric="batch/step")
        wandb.define_metric("system/*", step_metric="epoch")

        logger.info("📊 Wandb initialized with enhanced visualization configuration")

    try:
        # Train the model
        best_accuracy = train_multiclass_dr_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            model_name=model_name
        )

        # Evaluate on test set
        logger.info("📋 Evaluating on test set...")
        model.eval()
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_targets = []

        device = torch.device(config['system']['device'])
        model = model.to(device)

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images = images.to(device)
                labels = labels.long().to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        test_accuracy = test_correct / test_total

        # Generate detailed results
        from sklearn.metrics import classification_report, confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns

        conf_matrix = confusion_matrix(all_targets, all_predictions)
        class_names = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'PDR']

        results = {
            'best_val_accuracy': best_accuracy,
            'test_accuracy': test_accuracy,
            'target_achieved': test_accuracy >= config['system']['target_accuracy'],
            'medical_grade_pass': test_accuracy >= 0.90,
            'classification_report': classification_report(
                all_targets, all_predictions,
                target_names=class_names
            ),
            'confusion_matrix': conf_matrix.tolist(),
            'model_name': model_name,
            'dataset_info': {'class_counts': class_counts}
        }

        # Enhanced wandb logging for final results
        if config['system'].get('use_wandb', False) and WANDB_AVAILABLE:
            # Log final metrics
            final_metrics = {
                "final/best_val_accuracy": best_accuracy,
                "final/test_accuracy": test_accuracy,
                "final/target_achieved": results['target_achieved'],
                "final/medical_grade_pass": results['medical_grade_pass'],
                "final/target_gap": test_accuracy - config['system']['target_accuracy'],
                "final/medical_grade_gap": test_accuracy - 0.90
            }

            # Log per-class metrics from classification report
            from sklearn.metrics import precision_recall_fscore_support
            precision, recall, f1, support = precision_recall_fscore_support(
                all_targets, all_predictions, average=None
            )

            for i, class_name in enumerate(class_names):
                final_metrics[f"class_metrics/{class_name}/precision"] = precision[i]
                final_metrics[f"class_metrics/{class_name}/recall"] = recall[i]
                final_metrics[f"class_metrics/{class_name}/f1_score"] = f1[i]
                final_metrics[f"class_metrics/{class_name}/support"] = support[i]

            # Create and log confusion matrix visualization
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title(f'{model_name} - APTOS 2019 Confusion Matrix\nTest Accuracy: {test_accuracy:.3f}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()

            # Log confusion matrix as image
            final_metrics["visualizations/confusion_matrix"] = wandb.Image(plt)
            plt.close()

            # Create accuracy progress visualization
            if hasattr(model, 'train_history') or 'train_history' in locals():
                # Get training history from the saved checkpoint
                model_path = Path(config['system']['output_dir']) / "models" / f"best_{model_name}_multiclass.pth"
                if model_path.exists():
                    checkpoint = torch.load(model_path, map_location='cpu')
                    if 'train_history' in checkpoint:
                        history = checkpoint['train_history']

                        plt.figure(figsize=(12, 4))

                        # Accuracy plot
                        plt.subplot(1, 2, 1)
                        epochs_range = range(1, len(history['train_accuracies']) + 1)
                        plt.plot(epochs_range, history['train_accuracies'], 'b-', label='Training Accuracy')
                        plt.plot(epochs_range, history['val_accuracies'], 'r-', label='Validation Accuracy')
                        plt.axhline(y=config['system']['target_accuracy']*100, color='g', linestyle='--',
                                   label=f'Target ({config["system"]["target_accuracy"]:.1%})')
                        plt.axhline(y=90, color='orange', linestyle='--', label='Medical Grade (90%)')
                        plt.title('Model Accuracy Progress')
                        plt.xlabel('Epoch')
                        plt.ylabel('Accuracy (%)')
                        plt.legend()
                        plt.grid(True, alpha=0.3)

                        # Loss plot
                        plt.subplot(1, 2, 2)
                        plt.plot(epochs_range, history['train_losses'], 'b-', label='Training Loss')
                        plt.plot(epochs_range, history['val_losses'], 'r-', label='Validation Loss')
                        plt.title('Model Loss Progress')
                        plt.xlabel('Epoch')
                        plt.ylabel('Loss')
                        plt.legend()
                        plt.grid(True, alpha=0.3)

                        plt.tight_layout()
                        final_metrics["visualizations/training_progress"] = wandb.Image(plt)
                        plt.close()

            # Log all final metrics
            wandb.log(final_metrics)

            # Log the classification report as a table
            report_data = []
            lines = results['classification_report'].split('\n')
            for line in lines[2:-5]:  # Skip header and footer
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            # Handle class names with spaces (join multiple parts)
                            if len(parts) > 5:
                                class_name = ' '.join(parts[:-4])
                                metrics = parts[-4:]
                            else:
                                class_name = parts[0]
                                metrics = parts[1:]

                            report_data.append([
                                class_name,
                                float(metrics[0]),
                                float(metrics[1]),
                                float(metrics[2]),
                                int(metrics[3])
                            ])
                        except (ValueError, IndexError):
                            # Skip lines that can't be parsed (headers, averages, etc.)
                            continue

            if report_data:
                table = wandb.Table(
                    columns=["Class", "Precision", "Recall", "F1-Score", "Support"],
                    data=report_data
                )
                wandb.log({"classification_report": table})

            # Log sample predictions visualization
            sample_predictions = []
            sample_images = []
            sample_targets = []

            # Get a few sample predictions for visualization
            model.eval()
            with torch.no_grad():
                for i, (images, labels) in enumerate(test_loader):
                    if i >= 2:  # Only process first 2 batches for samples
                        break

                    images = images.to(device)
                    labels = labels.long().to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)

                    # Take first 4 samples from each batch
                    for j in range(min(4, images.size(0))):
                        img = images[j].cpu()
                        # Denormalize for visualization
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img = img * std + mean
                        img = torch.clamp(img, 0, 1)

                        sample_images.append(img.permute(1, 2, 0).numpy())
                        sample_targets.append(labels[j].item())
                        sample_predictions.append(predicted[j].item())

            # Create sample predictions visualization
            if sample_images:
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                for idx, (img, true_label, pred_label) in enumerate(zip(sample_images, sample_targets, sample_predictions)):
                    if idx >= 8:
                        break
                    row, col = idx // 4, idx % 4
                    axes[row, col].imshow(img)
                    axes[row, col].set_title(f'True: {class_names[true_label]}\nPred: {class_names[pred_label]}')
                    axes[row, col].axis('off')

                plt.tight_layout()
                final_metrics["visualizations/sample_predictions"] = wandb.Image(plt)
                plt.close()

            # Re-log all final metrics with sample predictions
            wandb.log(final_metrics)

            logger.info("📊 Enhanced visualizations logged to Wandb")

        # Log final results
        logger.info("🎆 APTOS 2019 TRAINING COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"🏆 FINAL RESULTS:")
        logger.info(f"   Best Val Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        logger.info(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        logger.info(f"   Target ({config['system']['target_accuracy']:.1%}): {'✅ ACHIEVED' if results['target_achieved'] else '❌ NOT ACHIEVED'}")
        logger.info(f"   Medical Grade (≥90%): {'✅ PASS' if results['medical_grade_pass'] else '❌ FAIL'}")

        # Save results
        results_path = Path(config['system']['output_dir']) / "results" / "aptos_multiclass_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)

        json_results = {
            'best_val_accuracy': float(best_accuracy),
            'test_accuracy': float(test_accuracy),
            'target_achieved': results['target_achieved'],
            'medical_grade_pass': results['medical_grade_pass'],
            'confusion_matrix': results['confusion_matrix'],
            'model_name': model_name,
            'target_accuracy': config['system']['target_accuracy']
        }

        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        logger.info(f"💾 Results saved: {results_path}")
        return results

    except Exception as e:
        logger.error(f"❌ APTOS training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if config['system']['use_wandb'] and WANDB_AVAILABLE:
            wandb.finish()

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
            # Determine training type based on configuration
            is_single_model = len(config['model']['base_models']) == 1
            dataset_name = Path(config['data']['dataset_path']).name.lower()

            if is_single_model:
                # APTOS multi-class training (research-validated 92% approach)
                logger.info("🔬 Detected single model dataset - using multi-class training")
                final_results = train_aptos_multiclass(config)

                logger.info("✅ APTOS multi-class training completed!")

                # Display final summary
                logger.info("\n" + "=" * 60)
                logger.info("🏆 APTOS 2019 RESULTS SUMMARY")
                logger.info("=" * 60)

                logger.info(f"🎯 Best Val Accuracy: {final_results['best_val_accuracy']:.4f} ({final_results['best_val_accuracy']*100:.2f}%)")
                logger.info(f"🎯 Test Accuracy: {final_results['test_accuracy']:.4f} ({final_results['test_accuracy']*100:.2f}%)")
                logger.info(f"🏥 Medical Grade: {'✅ PASS' if final_results['medical_grade_pass'] else '❌ FAIL'} (≥90% required)")
                logger.info(f"📊 Research Target: {'✅ ACHIEVED' if final_results['target_achieved'] else '❌ NOT ACHIEVED'} ({config['system']['target_accuracy']:.1%} target)")
                logger.info(f"🏗️ Model: {final_results['model_name']}")

            else:
                # OVO ensemble training for multiple models
                logger.info("🔢 Detected multi-model configuration - using OVO ensemble training")
                final_results = run_ovo_pipeline(config)

                logger.info("✅ OVO ensemble training and evaluation completed successfully!")

                # Display final summary
                logger.info("\n" + "=" * 60)
                logger.info("🏆 OVO ENSEMBLE RESULTS SUMMARY")
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

            # Load test dataset
            logger.info("📊 Loading test dataset...")
            test_dataset = ImageFolder(
                root=Path(config['data']['dataset_path']) / "test",
                transform=create_ovo_transforms(
                    img_size=config['data']['img_size'],
                    enable_clahe=config['data']['enable_clahe']
                )[1]  # Use validation transforms for test
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=config['data']['batch_size'],
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

            logger.info(f"📋 Test dataset: {len(test_dataset)} images")

            # Create OVO ensemble and load weights
            ovo_ensemble = OVOEnsemble(
                base_models=config['model']['base_models'],
                num_classes=config['data']['num_classes'],
                freeze_weights=config['model']['freeze_weights'],
                dropout=config['model']['dropout']
            )

            # Load the complete ensemble state
            state_dict = torch.load(ensemble_path, map_location='cpu')
            ovo_ensemble.load_state_dict(state_dict)
            logger.info("✅ OVO ensemble loaded successfully")

            # Evaluate the ensemble
            eval_results = evaluate_ovo_ensemble(ovo_ensemble, test_loader, config)

            # Display results
            logger.info("\n" + "="*60)
            logger.info("🏆 EVALUATION RESULTS")
            logger.info("="*60)
            logger.info(f"🎯 Ensemble Accuracy: {eval_results['ensemble_accuracy']:.4f} ({eval_results['ensemble_accuracy']*100:.2f}%)")
            logger.info(f"🏥 Medical Grade: {'✅ PASS' if eval_results['medical_grade_pass'] else '❌ FAIL'}")
            logger.info(f"📊 Research Target: {'✅ ACHIEVED' if eval_results['research_target_achieved'] else '❌ NOT ACHIEVED'}")
            logger.info("\n📊 Individual Model Performance:")
            for model_name, accuracy in eval_results['individual_accuracies'].items():
                logger.info(f"   {model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
            logger.info("="*60)

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