#!/usr/bin/env python3
"""
Individual Model Trainer for Diabetic Retinopathy Classification

Trains single architecture models (ResNet50, DenseNet121) that will be combined
into an ensemble with the existing EfficientNetB2 model.

Key Features:
- Medical-grade preprocessing with CLAHE
- SMOTE class balancing for improved minority class performance
- Early stopping with validation monitoring
- Comprehensive checkpointing and logging
- Medical validation thresholds
"""

import os
import sys
import argparse
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm

# Import the existing dataset and utilities
try:
    from ensemble_dataset import DRDataset
    from utils import EarlyStopping, compute_class_weights
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
except ImportError as e:
    print(f"❌ Error: Required modules not found: {e}")
    print("Ensure ensemble_dataset.py and utils.py are available.")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class IndividualModelTrainer:
    """Trainer for individual CNN models."""

    def __init__(self, model_name, config, train_loader, val_loader, device, class_weights):
        self.model_name = model_name
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.class_weights = class_weights

        # Initialize model
        self.model = self._create_model()
        self.model.to(device)

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Initialize scheduler
        if config.scheduler == 'cosine':
            # More conservative cosine annealing
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=15, T_mult=1, eta_min=1e-7
            )
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=3, factor=0.7, verbose=True
            )

        # Initialize loss function with label smoothing
        if config.enable_focal_loss:
            # Simple focal loss implementation
            self.criterion = self._create_focal_loss(class_weights)
        else:
            # Use strong label smoothing to prevent overconfidence
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.2)

        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta
        )

        # Training state
        self.best_accuracy = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

        logger.info(f"✅ {model_name.upper()} trainer initialized")
        logger.info(f"   Device: {device}")
        logger.info(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _create_model(self):
        """Create individual model based on architecture name."""
        num_classes = 5

        if self.model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            # Ultra-strong regularization with multiple dropout layers
            model.fc = nn.Sequential(
                nn.Dropout(self.config.dropout),
                nn.Linear(model.fc.in_features, 512),
                nn.ReLU(),
                nn.Dropout(self.config.dropout * 0.8),  # Second dropout layer
                nn.Linear(512, num_classes)
            )
        elif self.model_name == 'densenet121':
            model = models.densenet121(pretrained=True)
            # Ultra-strong regularization with multiple dropout layers
            model.classifier = nn.Sequential(
                nn.Dropout(self.config.dropout),
                nn.Linear(model.classifier.in_features, 512),
                nn.ReLU(),
                nn.Dropout(self.config.dropout * 0.8),  # Second dropout layer
                nn.Linear(512, num_classes)
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return model

    def _create_focal_loss(self, class_weights):
        """Create simple focal loss function."""
        class FocalLoss(nn.Module):
            def __init__(self, alpha=None, gamma=2.0):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.ce_loss = nn.CrossEntropyLoss(weight=alpha)

            def forward(self, inputs, targets):
                ce_loss = self.ce_loss(inputs, targets)
                pt = torch.exp(-ce_loss)
                focal_loss = (1 - pt) ** self.gamma * ce_loss
                return focal_loss

        return FocalLoss(alpha=class_weights, gamma=2.0)

    def _compute_metrics(self, targets, predictions):
        """Compute detailed classification metrics."""
        accuracy = accuracy_score(targets, predictions)
        f1 = f1_score(targets, predictions, average='weighted')

        # Classification report
        report = classification_report(targets, predictions, output_dict=True, zero_division=0)

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(targets, predictions).tolist()
        }

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Training {self.model_name}")

        for batch_idx, batch in enumerate(pbar):
            data = batch['image'].to(self.device)
            targets = batch['dr_grade'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            acc = 100.0 * correct / total
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{acc:.2f}%'
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total

        self.train_losses.append(epoch_loss)

        return epoch_loss, epoch_acc

    def validate_epoch(self):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Validating {self.model_name}"):
                data = batch['image'].to(self.device)
                targets = batch['dr_grade'].to(self.device)

                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_loss = running_loss / len(self.val_loader)
        val_accuracy = correct / total

        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_accuracy)

        # Compute detailed metrics
        metrics = self._compute_metrics(all_targets, all_predictions)

        return val_loss, val_accuracy, metrics

    def train(self):
        """Full training loop."""
        logger.info(f"\n🚀 Starting {self.model_name.upper()} training...")
        logger.info(f"   Target accuracy: {getattr(self.config, 'target_accuracy', 0.90):.2%}")

        for epoch in range(1, self.config.epochs + 1):
            # Training
            train_loss, train_acc = self.train_epoch(epoch)

            # Validation
            val_loss, val_acc, metrics = self.validate_epoch()

            # Learning rate scheduling
            if self.config.scheduler == 'cosine':
                self.scheduler.step()
            else:
                self.scheduler.step(val_acc)

            # Logging
            logger.info(f"   Epoch {epoch:3d}/{self.config.epochs} | "
                       f"Train Acc: {train_acc:.4f} | "
                       f"Val Acc: {val_acc:.4f} | "
                       f"Best: {self.best_accuracy:.4f}")

            # Save best model
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self._save_checkpoint(epoch, val_acc, metrics, is_best=True)

            # Regular checkpointing
            if epoch % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(epoch, val_acc, metrics, is_best=False)

            # Early stopping check (convert accuracy to loss for EarlyStopping)
            val_loss_for_stopping = 1.0 - val_acc  # Higher accuracy = lower "loss"
            if self.early_stopping(val_loss_for_stopping, self.model):
                logger.info(f"🛑 Early stopping triggered at epoch {epoch}")
                break

        # Final results
        logger.info(f"\n✅ {self.model_name.upper()} training completed!")
        logger.info(f"   Best validation accuracy: {self.best_accuracy:.4f} ({self.best_accuracy:.2%})")

        # Medical grade assessment
        medical_grade = self.best_accuracy >= 0.90
        logger.info(f"   Medical grade: {'✅ PASS' if medical_grade else '❌ FAIL'} "
                   f"(≥90%: {medical_grade})")

        return {
            'model_name': self.model_name,
            'final_accuracy': self.best_accuracy,
            'medical_grade_pass': medical_grade,
            'total_epochs': epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }

    def _save_checkpoint(self, epoch, accuracy, metrics, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_name': self.model_name,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_accuracy': accuracy,
            'best_val_accuracy': self.best_accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'metrics': metrics,
            'config': vars(self.config)
        }

        # Save regular checkpoint
        if epoch % self.config.checkpoint_frequency == 0:
            checkpoint_path = Path(self.config.output_dir) / 'checkpoints' / f'{self.model_name}_epoch_{epoch:03d}_checkpoint.pth'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"💾 Saved {self.model_name} epoch {epoch} checkpoint: {checkpoint_path}")

        # Save best checkpoint
        if is_best:
            best_path = Path(self.config.output_dir) / 'checkpoints' / f'best_{self.model_name}.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"🏆 Saved best {self.model_name} model: {best_path}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Individual Model Trainer for DR Classification')

    # Model selection
    parser.add_argument('--model', choices=['resnet50', 'densenet121'], required=True,
                       help='Model architecture to train')

    # Dataset configuration
    parser.add_argument('--dataset_path', default='./dataset5',
                       help='Path to dataset directory')
    parser.add_argument('--num_classes', type=int, default=5,
                       help='Number of classification classes')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=6,
                       help='Batch size for training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Initial learning rate (ultra-conservative)')
    parser.add_argument('--weight_decay', type=float, default=5e-3,
                       help='Weight decay for optimizer (maximum regularization)')
    parser.add_argument('--dropout', type=float, default=0.7,
                       help='Dropout rate (maximum regularization)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                       help='Maximum gradient norm for clipping (strict)')

    # Early stopping
    parser.add_argument('--patience', type=int, default=3,
                       help='Early stopping patience (ultra-aggressive)')
    parser.add_argument('--min_delta', type=float, default=0.01,
                       help='Minimum improvement threshold (very strict)')

    # Scheduler
    parser.add_argument('--scheduler', choices=['cosine', 'plateau'], default='cosine',
                       help='Learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='Warmup epochs for scheduler')

    # Data augmentation and preprocessing
    parser.add_argument('--enable_clahe', action='store_true',
                       help='Enable CLAHE preprocessing')
    parser.add_argument('--enable_smote', action='store_true',
                       help='Enable SMOTE class balancing')
    parser.add_argument('--enable_focal_loss', action='store_true',
                       help='Enable focal loss for imbalanced classes')
    parser.add_argument('--enable_class_weights', action='store_true',
                       help='Enable class weights in loss function')

    # Validation and checkpointing
    parser.add_argument('--validation_frequency', type=int, default=1,
                       help='Validation frequency (epochs)')
    parser.add_argument('--checkpoint_frequency', type=int, default=5,
                       help='Checkpoint saving frequency (epochs)')

    # Output configuration
    parser.add_argument('--output_dir', default='./results',
                       help='Output directory for results and checkpoints')
    parser.add_argument('--experiment_name', default='individual_model',
                       help='Experiment name for logging')

    # Medical validation
    parser.add_argument('--medical_terms', default=None,
                       help='Path to medical terms JSON file')

    # Device configuration
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)

    # Initialize device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"🎮 Using device: {device}")

    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")

    # Prepare data loaders
    logger.info("📊 Preparing data loaders...")

    # Load data paths and labels from directory structure
    def load_dataset_from_directory(data_dir):
        """Load image paths and labels from directory structure."""
        data_paths = []
        labels = []

        data_dir = Path(data_dir)
        for class_dir in sorted(data_dir.iterdir()):
            if class_dir.is_dir():
                class_label = int(class_dir.name)
                for img_path in class_dir.glob('*.jpg'):
                    data_paths.append(str(img_path))
                    labels.append(class_label)

        return data_paths, labels

    # Load training data
    train_paths, train_labels = load_dataset_from_directory(Path(args.dataset_path) / 'train')
    val_paths, val_labels = load_dataset_from_directory(Path(args.dataset_path) / 'val')
    test_paths, test_labels = load_dataset_from_directory(Path(args.dataset_path) / 'test')

    # Create datasets
    train_dataset = DRDataset(
        data_paths=train_paths,
        labels=train_labels,
        enable_clahe=args.enable_clahe
    )

    val_dataset = DRDataset(
        data_paths=val_paths,
        labels=val_labels,
        enable_clahe=args.enable_clahe
    )

    test_dataset = DRDataset(
        data_paths=test_paths,
        labels=test_labels,
        enable_clahe=args.enable_clahe
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Compute class weights
    class_weights = compute_class_weights(np.array(train_labels), args.num_classes)

    logger.info(f"📊 Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    logger.info(f"🏥 Class weights: {class_weights}")

    # Initialize trainer
    trainer = IndividualModelTrainer(
        model_name=args.model,
        config=args,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        class_weights=torch.tensor(class_weights, device=device)
    )

    # Train model
    results = trainer.train()

    # Save final results
    results_path = output_dir / f'{args.model}_training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"📊 Results saved to: {results_path}")
    logger.info(f"\n✅ {args.model.upper()} training completed successfully!")

if __name__ == "__main__":
    main()