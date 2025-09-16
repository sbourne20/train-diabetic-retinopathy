#!/usr/bin/env python3
"""
Clean Ensemble Trainer for Diabetic Retinopathy Classification
Multi-Architecture Ensemble: EfficientNetB2, ResNet50, DenseNet121

This is a simplified, self-contained version that works with our balanced dataset6
and follows the research methodology for achieving 96.96% ensemble accuracy.
"""

import os
import json
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import ImageFolder

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# Set seeds for reproducibility
def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CLAHETransform:
    """Medical-grade CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    def __init__(self, clip_limit=3.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        l = clahe.apply(l)

        # Merge channels and convert back to RGB
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

        return Image.fromarray(enhanced)

class MedicalTransforms:
    """Medical-grade augmentation transforms"""

    @staticmethod
    def get_train_transforms(image_size=224, enable_clahe=True):
        """Training transforms with medical-grade augmentation"""
        transforms_list = []

        if enable_clahe:
            transforms_list.append(CLAHETransform(clip_limit=3.0, tile_grid_size=(8, 8)))

        transforms_list.extend([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=15, fill=0),  # Preserve retinal anatomy
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=0,
                scale=(0.95, 1.05),  # Maintain field of view
                fillcolor=0
            ),
            transforms.ColorJitter(
                brightness=0.1,  # Camera variation simulation
                contrast=0.1,
                saturation=0.05,
                hue=0.02
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet standards
                std=[0.229, 0.224, 0.225]
            )
        ])

        return transforms.Compose(transforms_list)

    @staticmethod
    def get_val_transforms(image_size=224, enable_clahe=True):
        """Validation transforms (minimal processing)"""
        transforms_list = []

        if enable_clahe:
            transforms_list.append(CLAHETransform(clip_limit=3.0, tile_grid_size=(8, 8)))

        transforms_list.extend([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        return transforms.Compose(transforms_list)

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EnsembleModel(nn.Module):
    """Multi-architecture ensemble model"""

    def __init__(self, num_classes=5, ensemble_weights=[0.4, 0.35, 0.25], pretrained=True):
        super(EnsembleModel, self).__init__()

        self.ensemble_weights = torch.tensor(ensemble_weights, dtype=torch.float32)

        # EfficientNetB2
        self.efficientnet = models.efficientnet_b2(pretrained=pretrained)
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)
        )

        # ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.resnet.fc.in_features, num_classes)
        )

        # DenseNet121
        self.densenet = models.densenet121(pretrained=pretrained)
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.densenet.classifier.in_features, num_classes)
        )

    def forward(self, x, return_individual=False):
        # Get predictions from each model
        efficientnet_out = self.efficientnet(x)
        resnet_out = self.resnet(x)
        densenet_out = self.densenet(x)

        # Apply softmax for ensemble weighting
        efficientnet_probs = F.softmax(efficientnet_out, dim=1)
        resnet_probs = F.softmax(resnet_out, dim=1)
        densenet_probs = F.softmax(densenet_out, dim=1)

        # Weighted ensemble
        device = x.device
        weights = self.ensemble_weights.to(device)

        ensemble_probs = (weights[0] * efficientnet_probs +
                         weights[1] * resnet_probs +
                         weights[2] * densenet_probs)

        # Convert back to logits for loss calculation
        ensemble_logits = torch.log(ensemble_probs + 1e-8)

        if return_individual:
            return {
                'ensemble_logits': ensemble_logits,
                'individual_logits': {
                    'efficientnetb2': efficientnet_out,
                    'resnet50': resnet_out,
                    'densenet121': densenet_out
                }
            }

        return ensemble_logits

class EnsembleTrainer:
    """Main ensemble training class"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create output directories
        self.output_dir = Path(args.output_dir)
        self.models_dir = self.output_dir / "models"
        self.logs_dir = self.output_dir / "logs"
        self.results_dir = self.output_dir / "results"

        for dir_path in [self.models_dir, self.logs_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f"🚀 CLEAN ENSEMBLE TRAINING SETUP")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎯 Device: {self.device}")
        print(f"📊 Target: 96.96% ensemble accuracy")

    def load_datasets(self):
        """Load and prepare datasets with medical-grade transforms"""

        print("\n" + "="*60)
        print("📂 LOADING BALANCED DATASET6")
        print("="*60)

        # Define transforms
        train_transforms = MedicalTransforms.get_train_transforms(
            image_size=self.args.image_size,
            enable_clahe=self.args.enable_clahe
        )

        val_transforms = MedicalTransforms.get_val_transforms(
            image_size=self.args.image_size,
            enable_clahe=self.args.enable_clahe
        )

        # Load datasets
        train_dataset = ImageFolder(
            root=os.path.join(self.args.dataset_path, 'train'),
            transform=train_transforms
        )

        val_dataset = ImageFolder(
            root=os.path.join(self.args.dataset_path, 'val'),
            transform=val_transforms
        )

        test_dataset = ImageFolder(
            root=os.path.join(self.args.dataset_path, 'test'),
            transform=val_transforms
        )

        # Print dataset info
        print(f"📊 Train samples: {len(train_dataset):,}")
        print(f"📊 Validation samples: {len(val_dataset):,}")
        print(f"📊 Test samples: {len(test_dataset):,}")
        print(f"🎯 Classes: {train_dataset.classes}")

        # Calculate class distribution
        train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
        class_counts = np.bincount(train_labels)

        print(f"\n📈 Class Distribution:")
        for i, count in enumerate(class_counts):
            percentage = count / len(train_labels) * 100
            print(f"   Class {i}: {count:,} samples ({percentage:.1f}%)")

        # Calculate class weights
        class_weights = None
        if self.args.enable_class_weights:
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(train_labels),
                y=train_labels
            )
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            print(f"⚖️ Class weights: {class_weights.cpu().numpy()}")

        # Create data loaders
        if self.args.enable_class_weights and class_weights is not None:
            # Weighted sampling
            sample_weights = [class_weights[train_dataset[i][1]] for i in range(len(train_dataset))]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                sampler=sampler,
                num_workers=4,
                pin_memory=True
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_weights = class_weights
        self.num_classes = len(train_dataset.classes)

        return train_loader, val_loader, test_loader

    def setup_model(self):
        """Setup ensemble model and training components"""

        print("\n" + "="*60)
        print("🤖 SETTING UP ENSEMBLE MODEL")
        print("="*60)

        # Create ensemble model
        self.model = EnsembleModel(
            num_classes=self.num_classes,
            ensemble_weights=self.args.ensemble_weights,
            pretrained=True
        ).to(self.device)

        # Setup optimizer
        if self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=self.args.weight_decay
            )

        # Setup scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Setup mixed precision
        self.scaler = GradScaler()

        # Initialize tracking
        self.best_accuracy = 0.0
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'val_accuracies': [],
            'individual_accuracies': {'efficientnetb2': [], 'resnet50': [], 'densenet121': []}
        }

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"📊 Total parameters: {total_params:,}")
        print(f"🎯 Trainable parameters: {trainable_params:,}")
        print(f"⚖️ Ensemble weights: {self.args.ensemble_weights}")

    def train_epoch(self, epoch):
        """Train one epoch"""

        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Setup loss function
        if self.args.enable_focal_loss:
            criterion = FocalLoss(alpha=1, gamma=2)
        elif self.class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision training
            with autocast():
                outputs = self.model(data)
                loss = criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 50 == 0:
                print(f"   Epoch {epoch+1}/{self.args.epochs} [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {running_loss/(batch_idx+1):.4f} "
                      f"Acc: {100.*correct/total:.2f}%")

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        """Validate model"""

        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        all_predictions = []
        all_targets = []
        individual_predictions = {'efficientnetb2': [], 'resnet50': [], 'densenet121': []}

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                with autocast():
                    outputs = self.model(data, return_individual=True)
                    loss = criterion(outputs['ensemble_logits'], targets)

                val_loss += loss.item()
                _, predicted = outputs['ensemble_logits'].max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                # Individual model predictions
                for model_name, logits in outputs['individual_logits'].items():
                    _, individual_pred = logits.max(1)
                    individual_predictions[model_name].extend(individual_pred.cpu().numpy())

        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total

        # Calculate individual accuracies
        individual_accuracies = {}
        for model_name, preds in individual_predictions.items():
            individual_acc = accuracy_score(all_targets, preds) * 100
            individual_accuracies[model_name] = individual_acc

        return val_loss, val_acc, all_predictions, all_targets, individual_accuracies

    def train(self):
        """Main training loop"""

        print(f"\n🚀 STARTING ENSEMBLE TRAINING")
        print("="*60)
        print(f"📝 Experiment: {self.args.experiment_name}")
        print(f"⏱️ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Load datasets
        self.load_datasets()

        # Setup model
        self.setup_model()

        # Training loop
        patience_counter = 0

        print(f"\n🎯 TRAINING LOOP")
        print("="*40)

        for epoch in range(self.args.epochs):
            start_time = time.time()

            # Training
            train_loss, train_acc = self.train_epoch(epoch)

            # Validation
            val_loss, val_acc, _, _, individual_accs = self.validate()

            # Update scheduler
            self.scheduler.step(val_acc)

            # Save training history
            self.training_history['train_losses'].append(train_loss)
            self.training_history['val_losses'].append(val_loss)
            self.training_history['val_accuracies'].append(val_acc)

            for model_name, acc in individual_accs.items():
                self.training_history['individual_accuracies'][model_name].append(acc)

            epoch_time = time.time() - start_time

            print(f"Epoch {epoch+1:3d}/{self.args.epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                  f"Time: {epoch_time:.1f}s")

            print(f"   Individual: EfficientNet: {individual_accs['efficientnetb2']:.2f}% | "
                  f"ResNet: {individual_accs['resnet50']:.2f}% | "
                  f"DenseNet: {individual_accs['densenet121']:.2f}%")

            # Save best model
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                patience_counter = 0

                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_accuracy': self.best_accuracy,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'individual_accuracies': individual_accs,
                    'ensemble_weights': self.args.ensemble_weights,
                    'training_history': self.training_history,
                    'args': vars(self.args)
                }

                torch.save(checkpoint, self.models_dir / "ensemble_best.pth")
                print(f"   ✅ New best accuracy: {self.best_accuracy:.2f}% - Model saved!")

            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.args.patience:
                print(f"   ⏰ Early stopping triggered after {epoch+1} epochs")
                break

            # Regular checkpoint saving
            if (epoch + 1) % self.args.checkpoint_frequency == 0:
                checkpoint_path = self.models_dir / f"ensemble_epoch_{epoch+1}.pth"
                torch.save(checkpoint, checkpoint_path)
                print(f"   💾 Checkpoint saved: {checkpoint_path}")

        # Final evaluation
        self.final_evaluation()

        # Save final results
        self.save_results()

        print(f"\n" + "="*60)
        print("🏁 TRAINING COMPLETED!")
        print("="*60)

    def final_evaluation(self):
        """Final evaluation on test set"""

        print(f"\n🔬 FINAL EVALUATION")
        print("="*50)

        # Load best model
        checkpoint_path = self.models_dir / "ensemble_best.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded best model (accuracy: {checkpoint['best_val_accuracy']:.2f}%)")

        # Evaluate on test set
        self.model.eval()
        all_predictions = []
        all_targets = []
        individual_predictions = {'efficientnetb2': [], 'resnet50': [], 'densenet121': []}

        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                outputs = self.model(data, return_individual=True)

                # Ensemble predictions
                _, ensemble_pred = outputs['ensemble_logits'].max(1)
                all_predictions.extend(ensemble_pred.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                # Individual predictions
                for model_name, logits in outputs['individual_logits'].items():
                    _, individual_pred = logits.max(1)
                    individual_predictions[model_name].extend(individual_pred.cpu().numpy())

        # Calculate final metrics
        ensemble_accuracy = accuracy_score(all_targets, all_predictions) * 100
        individual_accuracies = {
            name: accuracy_score(all_targets, preds) * 100
            for name, preds in individual_predictions.items()
        }

        print(f"\n📊 FINAL TEST RESULTS:")
        print(f"🎯 Ensemble Accuracy: {ensemble_accuracy:.2f}%")
        print(f"\n📋 Individual Model Accuracies:")
        for model_name, acc in individual_accuracies.items():
            print(f"   {model_name}: {acc:.2f}%")

        # Research target assessment
        target_accuracies = {
            'efficientnetb2': 96.27,
            'resnet50': 94.95,
            'densenet121': 91.21
        }
        target_ensemble = 96.96

        print(f"\n📚 RESEARCH TARGET COMPARISON:")
        for model_name, achieved in individual_accuracies.items():
            target = target_accuracies.get(model_name, 90.0)
            status = "✅" if achieved >= target else "❌"
            print(f"   {model_name}: {achieved:.2f}% / {target:.2f}% {status}")

        ensemble_status = "✅" if ensemble_accuracy >= target_ensemble else "❌"
        print(f"   Ensemble: {ensemble_accuracy:.2f}% / {target_ensemble:.2f}% {ensemble_status}")

        # Medical grade assessment
        if ensemble_accuracy >= 96.0:
            grade = "✅ MEDICAL GRADE PASS"
            status = "PRODUCTION READY"
        elif ensemble_accuracy >= 90.0:
            grade = "⚠️ NEAR MEDICAL GRADE"
            status = "CLOSE TO PRODUCTION"
        elif ensemble_accuracy >= 80.0:
            grade = "📈 PROMISING LEVEL"
            status = "RESEARCH QUALITY"
        else:
            grade = "❌ NEEDS IMPROVEMENT"
            status = "BELOW STANDARDS"

        print(f"\n🏥 {grade}")
        print(f"🔧 Status: {status}")

        # Save evaluation results
        self.final_results = {
            'ensemble_accuracy': ensemble_accuracy,
            'individual_accuracies': individual_accuracies,
            'medical_grade': grade,
            'status': status,
            'targets_achieved': {
                'ensemble': ensemble_accuracy >= target_ensemble,
                'individual': {name: acc >= target_accuracies.get(name, 90.0)
                             for name, acc in individual_accuracies.items()}
            }
        }

        return ensemble_accuracy

    def save_results(self):
        """Save comprehensive training results"""

        print(f"\n💾 SAVING RESULTS")
        print("="*30)

        # Save training history
        results = {
            'experiment_name': self.args.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'dataset_path': self.args.dataset_path,
            'args': vars(self.args),
            'training_history': self.training_history,
            'final_results': getattr(self, 'final_results', {}),
            'best_accuracy': self.best_accuracy
        }

        results_file = self.results_dir / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✅ Results saved: {results_file}")

        # Save summary
        summary_file = self.results_dir / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write("CLEAN ENSEMBLE TRAINING SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(f"Experiment: {self.args.experiment_name}\n")
            f.write(f"Dataset: {self.args.dataset_path}\n")
            f.write(f"Best Validation Accuracy: {self.best_accuracy:.2f}%\n")

            if hasattr(self, 'final_results'):
                f.write(f"Final Test Accuracy: {self.final_results['ensemble_accuracy']:.2f}%\n")
                f.write(f"Medical Grade: {self.final_results['medical_grade']}\n")

        print(f"✅ Summary saved: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Clean Ensemble Training for Diabetic Retinopathy')

    # Dataset arguments
    parser.add_argument('--dataset_path', default='./dataset6',
                       help='Path to balanced dataset6')
    parser.add_argument('--num_classes', type=int, default=5,
                       help='Number of classes')

    # Model arguments
    parser.add_argument('--ensemble_weights', nargs=3, type=float,
                       default=[0.4, 0.35, 0.25],
                       help='Ensemble weights for EfficientNetB2, ResNet50, DenseNet121')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--optimizer', choices=['adam', 'adamw', 'sgd'], default='adam',
                       help='Optimizer')

    # Medical enhancements
    parser.add_argument('--enable_clahe', action='store_true', default=True,
                       help='Enable CLAHE preprocessing')
    parser.add_argument('--enable_class_weights', action='store_true', default=True,
                       help='Enable class weights')
    parser.add_argument('--enable_focal_loss', action='store_true',
                       help='Enable focal loss')

    # Training control
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--checkpoint_frequency', type=int, default=5,
                       help='Checkpoint frequency')

    # Output
    parser.add_argument('--output_dir', default='./clean_ensemble_results',
                       help='Output directory')
    parser.add_argument('--experiment_name', default='clean_ensemble_efficientnetb2_resnet50_densenet121',
                       help='Experiment name')

    # System
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Validate dataset
    if not os.path.exists(args.dataset_path):
        print(f"❌ Error: Dataset path '{args.dataset_path}' not found!")
        return 1

    # Create trainer and start training
    trainer = EnsembleTrainer(args)
    trainer.train()

    return 0

if __name__ == "__main__":
    exit(main())