#!/usr/bin/env python3
"""
Specialized Dataset Trainer for OVO Binary Classification
Simplified version of ensemble_local_trainer.py for dataset-specific training
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

# Essential imports for OVO ensemble
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import ImageFolder
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments for specialized training."""

    parser = argparse.ArgumentParser(
        description='Specialized Dataset OVO Training',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Core parameters
    parser.add_argument('--mode', choices=['train', 'evaluate'],
                       default='train', help='Mode to run the script')
    parser.add_argument('--dataset_path', required=True,
                       help='Path to specialized dataset directory')
    parser.add_argument('--output_dir', default='./specialized_results',
                       help='Output directory for results')
    parser.add_argument('--base_models', nargs='+',
                       default=['mobilenet_v2'],
                       choices=['mobilenet_v2', 'inception_v3', 'densenet121'],
                       help='Base models for training')

    # Training parameters
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    return parser.parse_args()

class BinaryClassifier(nn.Module):
    """Simple binary classifier for specialized training."""

    def __init__(self, model_name='mobilenet_v2', dropout=0.5):
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

        # Freeze backbone for transfer learning
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Simple binary classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout * 0.5),
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Forward pass for binary classification."""
        # Handle InceptionV3 size requirements
        if self.model_name == 'inception_v3' and x.size(-1) < 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        # Handle InceptionV3 auxiliary outputs during training
        if self.model_name == 'inception_v3' and self.training:
            features, aux_features = self.backbone(x)
        else:
            features = self.backbone(x)

        # Handle any remaining tuple outputs
        if isinstance(features, tuple):
            features = features[0]

        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)

        return self.classifier(features)

class BinaryDataset(Dataset):
    """Dataset for binary classification tasks."""

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

def create_transforms(img_size=224):
    """Create transforms for training."""

    # Ensure minimum size for InceptionV3
    safe_img_size = max(img_size, 299) if img_size != 224 else img_size

    train_transform = transforms.Compose([
        transforms.Resize((safe_img_size, safe_img_size)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((safe_img_size, safe_img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def train_binary_classifier(model, train_loader, val_loader, config, class_pair, model_name):
    """Train a single binary classifier."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'],
                          weight_decay=config['weight_decay'])

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    best_val_acc = 0.0
    patience_counter = 0
    patience = 10

    logger.info(f"🏁 Training {model_name} for classes {class_pair}")

    for epoch in range(config['epochs']):
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

            # Ensure both outputs and labels have the same shape
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            if labels.dim() == 0:
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

                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)

                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100.0 * val_correct / val_total

        # Step scheduler
        scheduler.step(val_acc)

        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # Save checkpoint
            model_path = Path(config['output_dir']) / "models" / f"best_{model_name}_{class_pair[0]}_{class_pair[1]}.pth"
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_accuracy': best_val_acc,
                'current_val_accuracy': val_acc,
                'current_train_accuracy': train_acc,
                'class_pair': class_pair,
                'model_name': model_name,
                'config': config
            }
            torch.save(checkpoint, model_path)
            logger.info(f"🎯 New best for {model_name}_{class_pair}: {val_acc:.2f}%")
        else:
            patience_counter += 1

        logger.info(f"   Epoch {epoch+1}/{config['epochs']}: "
                   f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"   Early stopping at epoch {epoch+1}")
            break

    logger.info(f"✅ {model_name} for {class_pair}: Best Val Acc = {best_val_acc:.2f}%")
    return best_val_acc

def main():
    """Main function for specialized training."""

    print("🎯 SPECIALIZED DATASET OVO TRAINING")
    print("=" * 50)

    # Parse arguments
    args = parse_args()

    # Setup configuration
    config = {
        'dataset_path': args.dataset_path,
        'output_dir': args.output_dir,
        'base_models': args.base_models,
        'img_size': args.img_size,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'seed': args.seed
    }

    # Create output directories
    output_path = Path(config['output_dir'])
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "models").mkdir(exist_ok=True)

    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])

    logger.info(f"📁 Dataset: {config['dataset_path']}")
    logger.info(f"🏗️ Models: {config['base_models']}")
    logger.info(f"📁 Output: {config['output_dir']}")

    # Load datasets
    dataset_path = Path(config['dataset_path'])
    train_path = dataset_path / "train"
    val_path = dataset_path / "val"

    if not train_path.exists() or not val_path.exists():
        logger.error(f"❌ Dataset must have train/val structure in {dataset_path}")
        return

    # Create transforms
    train_transform, val_transform = create_transforms(config['img_size'])

    # Load base datasets
    train_dataset = ImageFolder(str(train_path), transform=None)
    val_dataset = ImageFolder(str(val_path), transform=None)

    logger.info(f"📊 Dataset loaded:")
    logger.info(f"   Train: {len(train_dataset)} samples")
    logger.info(f"   Val: {len(val_dataset)} samples")

    # Generate class pairs for OVO
    num_classes = 5
    class_pairs = list(combinations(range(num_classes), 2))

    logger.info(f"🔢 Training {len(class_pairs)} binary classifiers per model")

    # Train all binary classifiers
    results = {}

    for model_name in config['base_models']:
        logger.info(f"\n🏗️ Training {model_name} binary classifiers")
        results[model_name] = {}

        for class_a, class_b in class_pairs:
            # Create binary datasets
            binary_train = BinaryDataset(train_dataset, class_a, class_b, transform=train_transform)
            binary_val = BinaryDataset(val_dataset, class_a, class_b, transform=val_transform)

            # Skip if not enough samples
            if len(binary_train) < 10 or len(binary_val) < 5:
                logger.warning(f"⏭️ Skipping {model_name} ({class_a}, {class_b}) - insufficient data")
                continue

            # Create data loaders
            train_loader = DataLoader(binary_train, batch_size=config['batch_size'],
                                    shuffle=True, num_workers=2, pin_memory=True)
            val_loader = DataLoader(binary_val, batch_size=config['batch_size'],
                                  shuffle=False, num_workers=2, pin_memory=True)

            logger.info(f"📊 Binary dataset ({class_a}, {class_b}): Train={len(binary_train)}, Val={len(binary_val)}")

            # Create and train binary classifier
            binary_model = BinaryClassifier(model_name=model_name, dropout=0.5)

            best_acc = train_binary_classifier(
                model=binary_model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                class_pair=(class_a, class_b),
                model_name=model_name
            )

            results[model_name][f"{class_a}_{class_b}"] = best_acc

    # Save results
    results_path = output_path / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✅ Specialized training completed!")
    logger.info(f"📁 Results saved to: {config['output_dir']}")
    logger.info(f"💾 Training results: {results_path}")

    # Display summary
    logger.info(f"\n📊 TRAINING SUMMARY:")
    for model_name, model_results in results.items():
        avg_acc = np.mean(list(model_results.values())) if model_results else 0
        logger.info(f"   {model_name}: {len(model_results)} pairs trained, avg accuracy: {avg_acc:.2f}%")

if __name__ == "__main__":
    main()