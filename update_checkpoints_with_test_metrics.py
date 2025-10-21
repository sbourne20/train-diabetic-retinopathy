#!/usr/bin/env python3
"""
Update existing checkpoints with test metrics.

This script loads each trained binary classifier checkpoint,
evaluates it on the test set, and saves test metrics back to the checkpoint.

Usage:
    python update_checkpoints_with_test_metrics.py \
        --dataset_path ./dataset_eyepacs_5class_balanced_enhanced \
        --results_dir ./densenet_5class_v4_enhanced_results
"""

import os
import sys
import argparse
import json
from pathlib import Path
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torchvision import models
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinaryClassifier(nn.Module):
    """Simple binary classifier head for pretrained models."""

    def __init__(self, backbone_name='densenet121', num_features=1024, dropout=0.5):
        super().__init__()
        self.backbone_name = backbone_name

        # Load pretrained backbone
        if backbone_name == 'densenet121':
            self.backbone = models.densenet121(pretrained=True)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Binary classification head
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
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class BinaryDataset(torch.utils.data.Dataset):
    """Binary dataset for two classes."""

    def __init__(self, base_dataset, class_a, class_b, transform=None):
        self.base_dataset = base_dataset
        self.class_a = class_a
        self.class_b = class_b
        self.transform = transform

        # Filter indices for classes A and B
        self.indices = []
        self.labels = []

        for idx in range(len(base_dataset)):
            _, label = base_dataset[idx]
            if label == class_a:
                self.indices.append(idx)
                self.labels.append(0)  # Class A = 0
            elif label == class_b:
                self.indices.append(idx)
                self.labels.append(1)  # Class B = 1

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        image, _ = self.base_dataset[base_idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def evaluate_checkpoint_on_test(checkpoint_path, test_loader, device):
    """Evaluate a checkpoint on test set and return metrics."""

    logger.info(f"📊 Evaluating: {checkpoint_path.name}")

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Check if test metrics already exist
        if 'test_accuracy' in checkpoint:
            logger.info(f"   ✅ Already has test metrics (Test Acc: {checkpoint['test_accuracy']:.2f}%)")
            return checkpoint

        # Extract model name and create model
        model_name = checkpoint.get('model_name', 'densenet121')
        config = checkpoint.get('config', {})
        dropout = config.get('model', {}).get('dropout', 0.5)

        # Create model
        model = BinaryClassifier(
            backbone_name=model_name,
            dropout=dropout
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        # Evaluate on test set
        test_correct = 0
        test_total = 0
        all_test_predictions = []
        all_test_labels = []
        all_test_probabilities = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.float().to(device)

                outputs = model(images).squeeze()

                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)

                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities > 0.5).float()

                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                all_test_predictions.extend(predicted.cpu().numpy())
                all_test_labels.extend(labels.cpu().numpy())
                all_test_probabilities.extend(probabilities.cpu().numpy())

        # Calculate metrics
        test_accuracy = 100.0 * test_correct / test_total

        all_test_predictions = np.array(all_test_predictions)
        all_test_labels = np.array(all_test_labels)
        all_test_probabilities = np.array(all_test_probabilities)

        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
            all_test_labels, all_test_predictions, average='binary', zero_division=0
        )
        test_auc = roc_auc_score(all_test_labels, all_test_probabilities)

        logger.info(f"   Test Acc: {test_accuracy:.2f}%, Precision: {test_precision:.4f}, "
                   f"Recall: {test_recall:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")

        # Update checkpoint
        checkpoint['test_accuracy'] = test_accuracy
        checkpoint['test_precision'] = test_precision
        checkpoint['test_recall'] = test_recall
        checkpoint['test_f1'] = test_f1
        checkpoint['test_auc'] = test_auc

        # Save updated checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"   💾 Updated checkpoint with test metrics")

        return checkpoint

    except Exception as e:
        logger.error(f"   ❌ Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Update checkpoints with test metrics')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to dataset')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Path to results directory with checkpoints')
    parser.add_argument('--img_size', type=int, default=384,
                       help='Image size (default: 384)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 Device: {device}")

    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load test dataset
    test_path = Path(args.dataset_path) / 'test'
    if not test_path.exists():
        logger.error(f"❌ Test directory not found: {test_path}")
        return

    base_test_dataset = ImageFolder(test_path, transform=None)
    logger.info(f"📁 Loaded test dataset: {len(base_test_dataset)} images")

    # Find all checkpoint files
    models_dir = Path(args.results_dir) / 'models'
    checkpoint_files = list(models_dir.glob('best_densenet121_*_*.pth'))

    if not checkpoint_files:
        logger.error(f"❌ No checkpoint files found in {models_dir}")
        return

    logger.info(f"📦 Found {len(checkpoint_files)} checkpoints")

    # Process each checkpoint
    updated_count = 0
    for checkpoint_path in sorted(checkpoint_files):
        # Extract class pair from filename
        filename = checkpoint_path.stem  # e.g., "best_densenet121_0_1"
        parts = filename.split('_')
        if len(parts) < 4:
            logger.warning(f"⚠️  Skipping {filename} (invalid format)")
            continue

        class_a = int(parts[-2])
        class_b = int(parts[-1])

        # Create binary test dataset
        binary_test = BinaryDataset(base_test_dataset, class_a, class_b, transform=transform)
        test_loader = DataLoader(
            binary_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        logger.info(f"\n📊 Processing pair ({class_a}, {class_b}): {len(binary_test)} test samples")

        # Evaluate and update
        result = evaluate_checkpoint_on_test(checkpoint_path, test_loader, device)
        if result is not None:
            updated_count += 1

    logger.info(f"\n✅ Updated {updated_count}/{len(checkpoint_files)} checkpoints")


if __name__ == '__main__':
    main()
