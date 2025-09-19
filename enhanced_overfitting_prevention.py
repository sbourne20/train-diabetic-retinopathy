#!/usr/bin/env python3
"""
Enhanced Overfitting Prevention Module for OVO Ensemble Training

This module provides comprehensive overfitting prevention techniques:
- Advanced early stopping with validation loss monitoring
- Dynamic dropout adjustment based on overfitting detection
- Gradient clipping for training stability
- Model complexity reduction when overfitting detected
- Advanced learning rate scheduling
- Validation loss plateau detection
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class AdvancedEarlyStopping:
    """Enhanced early stopping with validation loss monitoring and overfitting detection."""

    def __init__(self,
                 patience: int = 7,
                 min_delta: float = 0.001,
                 restore_best_weights: bool = True,
                 overfitting_threshold: float = 0.08,  # 8% train-val gap
                 validation_loss_patience: int = 5):

        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.overfitting_threshold = overfitting_threshold
        self.validation_loss_patience = validation_loss_patience

        self.best_val_accuracy = 0.0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.val_loss_patience_counter = 0
        self.best_model_state = None
        self.overfitting_detected = False

        # Track recent performance for overfitting detection
        self.recent_train_accs = []
        self.recent_val_accs = []
        self.recent_val_losses = []

    def __call__(self, val_accuracy: float, val_loss: float, train_accuracy: float, model: nn.Module) -> bool:
        """
        Check if training should stop due to early stopping or severe overfitting.

        Returns:
            bool: True if training should stop, False otherwise
        """

        # Update recent performance tracking
        self.recent_train_accs.append(train_accuracy)
        self.recent_val_accs.append(val_accuracy)
        self.recent_val_losses.append(val_loss)

        # Keep only last 5 epochs for trend analysis
        if len(self.recent_train_accs) > 5:
            self.recent_train_accs = self.recent_train_accs[-5:]
            self.recent_val_accs = self.recent_val_accs[-5:]
            self.recent_val_losses = self.recent_val_losses[-5:]

        # Check for severe overfitting
        if len(self.recent_train_accs) >= 3:
            avg_train_acc = np.mean(self.recent_train_accs[-3:])
            avg_val_acc = np.mean(self.recent_val_accs[-3:])
            overfitting_gap = avg_train_acc - avg_val_acc

            if overfitting_gap > self.overfitting_threshold * 100:  # Convert to percentage
                if not self.overfitting_detected:
                    logger.warning(f"🚨 OVERFITTING DETECTED! Train-Val gap: {overfitting_gap:.1f}%")
                    self.overfitting_detected = True

                # If overfitting is critical (≥8% gap), stop immediately for medical-grade requirements
                if overfitting_gap >= 8.0:
                    logger.error(f"❌ CRITICAL OVERFITTING: {overfitting_gap:.1f}% gap ≥8%. Stopping training for medical-grade quality.")
                    return True

        # Check validation accuracy improvement
        improved_accuracy = val_accuracy > self.best_val_accuracy + self.min_delta

        if improved_accuracy:
            self.best_val_accuracy = val_accuracy
            self.patience_counter = 0

            if self.restore_best_weights:
                self.best_model_state = model.state_dict().copy()

            logger.info(f"✅ New best validation accuracy: {val_accuracy:.2f}%")
        else:
            self.patience_counter += 1
            logger.debug(f"No improvement. Patience: {self.patience_counter}/{self.patience}")

        # Check validation loss improvement (secondary metric)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.val_loss_patience_counter = 0
        else:
            self.val_loss_patience_counter += 1

        # Early stopping decision
        if self.patience_counter >= self.patience:
            logger.info(f"🛑 Early stopping triggered after {self.patience} epochs without improvement")

            if self.restore_best_weights and self.best_model_state is not None:
                model.load_state_dict(self.best_model_state)
                logger.info("🔄 Restored best model weights")

            return True

        # Additional stopping for validation loss plateau
        if self.val_loss_patience_counter >= self.validation_loss_patience * 2:
            logger.warning(f"⚠️ Validation loss plateau detected. Consider stopping.")

        return False

    def get_overfitting_metrics(self) -> Dict[str, float]:
        """Get current overfitting metrics."""
        if len(self.recent_train_accs) >= 3:
            avg_train_acc = np.mean(self.recent_train_accs[-3:])
            avg_val_acc = np.mean(self.recent_val_accs[-3:])
            overfitting_gap = avg_train_acc - avg_val_acc
        else:
            overfitting_gap = 0.0

        return {
            'overfitting_gap': overfitting_gap,
            'overfitting_detected': self.overfitting_detected,
            'best_val_accuracy': self.best_val_accuracy,
            'patience_remaining': self.patience - self.patience_counter
        }

class DynamicDropout(nn.Module):
    """Dynamic dropout that adjusts based on overfitting detection."""

    def __init__(self, initial_dropout: float = 0.5, max_dropout: float = 0.8):
        super().__init__()
        self.initial_dropout = initial_dropout
        self.max_dropout = max_dropout
        self.current_dropout = initial_dropout
        self.dropout = nn.Dropout(self.current_dropout)

    def adjust_dropout(self, overfitting_gap: float):
        """Adjust dropout rate based on overfitting gap."""
        if overfitting_gap > 6.0:  # Approaching critical overfitting
            self.current_dropout = min(self.max_dropout, self.current_dropout + 0.1)
        elif overfitting_gap > 4.0:  # Moderate overfitting
            self.current_dropout = min(self.max_dropout, self.current_dropout + 0.05)
        elif overfitting_gap < 2.0:  # Excellent generalization
            self.current_dropout = max(self.initial_dropout, self.current_dropout - 0.02)

        self.dropout.p = self.current_dropout
        logger.debug(f"Adjusted dropout to {self.current_dropout:.3f}")

    def forward(self, x):
        return self.dropout(x)

class AdvancedLRScheduler:
    """Advanced learning rate scheduler with overfitting-aware adjustments."""

    def __init__(self, optimizer, mode='max', factor=0.5, patience=3,
                 min_lr=1e-7, overfitting_factor=0.8):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience,
            min_lr=min_lr, verbose=True
        )
        self.overfitting_factor = overfitting_factor
        self.optimizer = optimizer

    def step(self, val_metric, overfitting_gap=0.0):
        """Step with overfitting-aware learning rate adjustment."""

        # Standard scheduler step
        self.scheduler.step(val_metric)

        # Additional LR reduction if approaching critical overfitting
        if overfitting_gap > 6.0:
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] *= self.overfitting_factor
                new_lr = param_group['lr']
                logger.warning(f"🔻 Overfitting detected ({overfitting_gap:.1f}%). Reduced LR: {old_lr:.2e} → {new_lr:.2e}")

def create_enhanced_model_with_dropout(model_name: str, dropout: float = 0.6,
                                     freeze_weights: bool = True) -> nn.Module:
    """Create model with enhanced dropout layers for better overfitting prevention."""

    if model_name == 'mobilenet_v2':
        from torchvision import models
        backbone = models.mobilenet_v2(pretrained=True)

        if freeze_weights:
            for param in backbone.parameters():
                param.requires_grad = False

        # Replace classifier with enhanced version
        in_features = backbone.classifier[1].in_features

        backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),           # Input dropout
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),          # Batch normalization
            DynamicDropout(dropout * 0.8), # Dynamic dropout
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            DynamicDropout(dropout * 0.6), # Reducing dropout in deeper layers
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    elif model_name == 'inception_v3':
        from torchvision import models
        backbone = models.inception_v3(pretrained=True, aux_logits=False)

        if freeze_weights:
            for param in backbone.parameters():
                param.requires_grad = False

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            DynamicDropout(dropout * 0.8),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            DynamicDropout(dropout * 0.6),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            DynamicDropout(dropout * 0.4),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    elif model_name == 'densenet121':
        from torchvision import models
        backbone = models.densenet121(pretrained=True)

        if freeze_weights:
            for param in backbone.parameters():
                param.requires_grad = False

        in_features = backbone.classifier.in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            DynamicDropout(dropout * 0.8),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            DynamicDropout(dropout * 0.6),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    return backbone

def enhanced_training_step(model, train_loader, val_loader, config, class_pair, model_name):
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
        {'params': backbone_params, 'lr': config['training']['learning_rate'] * 0.1, 'weight_decay': config['training']['weight_decay'] * 2},
        {'params': classifier_params, 'lr': config['training']['learning_rate'], 'weight_decay': config['training']['weight_decay']}
    ])

    # Advanced learning rate scheduler
    scheduler = AdvancedLRScheduler(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-7)

    # Enhanced early stopping
    early_stopping = AdvancedEarlyStopping(
        patience=config['training'].get('early_stopping_patience', 5),
        min_delta=0.001,
        overfitting_threshold=0.15,  # 15% train-val gap threshold
        validation_loss_patience=3
    )

    # Gradient clipping for stability
    max_grad_norm = 1.0

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
    logger.info(f"   Using advanced overfitting prevention:")
    logger.info(f"   - Early stopping patience: {early_stopping.patience}")
    logger.info(f"   - Gradient clipping: {max_grad_norm}")
    logger.info(f"   - Dynamic dropout adjustment")
    logger.info(f"   - Overfitting threshold: {early_stopping.overfitting_threshold*100:.1f}%")

    for epoch in range(config['training']['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data).squeeze()

            # Convert to binary labels for current class pair
            binary_target = (target == class_pair[1]).float()

            loss = criterion(output, binary_target)
            loss.backward()

            # Gradient clipping
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

        # Update dynamic dropout if model has it
        overfitting_gap = train_acc - val_acc
        for module in model.modules():
            if isinstance(module, DynamicDropout):
                module.adjust_dropout(overfitting_gap)

        # Learning rate scheduling with overfitting awareness
        scheduler.step(val_acc, overfitting_gap)

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

        if overfitting_gap > 15.0:
            logger.warning(f"   ⚠️ Overfitting detected: {overfitting_gap:.1f}% train-val gap")

        # Early stopping check
        should_stop = early_stopping(val_acc, avg_val_loss, train_acc, model)

        if should_stop:
            logger.info(f"   🛑 Training stopped early at epoch {epoch+1}")
            break

        # Save checkpoint if best
        if val_acc == early_stopping.best_val_accuracy:
            model_path = Path(config['system']['output_dir']) / "models" / f"best_{model_name}_{class_pair[0]}_{class_pair[1]}.pth"
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.scheduler.state_dict(),
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
    logger.info(f"   Final metrics: {final_metrics}")

    return early_stopping.best_val_accuracy