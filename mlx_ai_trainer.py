#!/usr/bin/env python3
"""
MLX-based trainer for diabetic retinopathy model on Mac M4
Optimized for Apple Silicon with local training capability
"""

import os
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# MLX imports for Apple Silicon optimization
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    MLX_AVAILABLE = True
except ImportError:
    print("❌ MLX not available. Install with: pip install mlx")
    MLX_AVAILABLE = False

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_loaded = load_dotenv()
    if env_loaded:
        print("✅ Environment variables loaded from .env file")
    else:
        print("⚠️  Warning: .env file not found or empty")
except ImportError:
    print("❌ Error: python-dotenv not found. Install with: pip install python-dotenv")

# Standard ML imports
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Local imports
from dataset import DiabeticRetinopathyDataset, create_data_splits_type1, get_transforms
from models import DiabeticRetinopathyModel, calculate_medical_metrics
from utils import AverageMeter, EarlyStopping, save_results

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mlx_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MLXDRTrainer:
    """MLX-optimized trainer for diabetic retinopathy classification on Mac M4."""
    
    def __init__(self, args):
        self.args = args
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Setup paths
        self.dataset_path = Path(args.dataset_path)
        self.results_dir = Path(args.results_dir)
        self.checkpoints_dir = self.results_dir / "checkpoints"
        self.outputs_dir = self.results_dir / "outputs"
        self.models_dir = self.results_dir / "models"
        
        # Create directories
        self.results_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.outputs_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Training state
        self.start_epoch = 0
        self.best_val_accuracy = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.medical_validations = []
        
        # Initialize model and data
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()
        self._setup_scheduler()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=args.patience,
            min_delta=args.min_delta,
            restore_best_weights=True
        )
        
        logger.info("✅ MLX Trainer initialized successfully")
    
    def _setup_model(self):
        """Initialize the MedSigLIP model with local caching."""
        logger.info("🏥 Initializing MedSigLIP-448 model...")
        
        # Verify HuggingFace token
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            raise ValueError(
                "HUGGINGFACE_TOKEN not found. "
                "Add it to your .env file: HUGGINGFACE_TOKEN=hf_your_token_here"
            )
        
        # Check if model is already cached locally
        local_model_path = self.models_dir / "medsiglip-448"
        
        if local_model_path.exists():
            logger.info(f"✅ Found cached MedSigLIP-448 model at: {local_model_path}")
            logger.info("💾 Loading from local cache (no download needed)")
            
            # Set environment variable to use local cache
            os.environ["TRANSFORMERS_CACHE"] = str(self.models_dir)
            
            self.model = DiabeticRetinopathyModel(
                img_size=448,
                num_classes=5,
                dropout=self.args.dropout,
                enable_confidence=True,
                use_lora=True,  # LoRA enabled to match checkpoint
                lora_r=16,      # Same as Vertex AI checkpoint
                lora_alpha=32   # Same as Vertex AI checkpoint
            ).to(self.device)
        else:
            logger.info("📥 Downloading MedSigLIP-448 model (first time only)...")
            logger.info(f"💾 Will cache to: {local_model_path}")
            
            # Set cache directory for download
            os.environ["TRANSFORMERS_CACHE"] = str(self.models_dir)
            
            self.model = DiabeticRetinopathyModel(
                img_size=448,
                num_classes=5,
                dropout=self.args.dropout,
                enable_confidence=True,
                use_lora=True,  # LoRA enabled to match checkpoint
                lora_r=16,      # Same as Vertex AI checkpoint
                lora_alpha=32   # Same as Vertex AI checkpoint
            ).to(self.device)
            
            logger.info(f"✅ Model downloaded and cached to: {local_model_path}")
            logger.info("🚀 Next time training will start instantly!")
        
        # Load from checkpoint if specified
        if self.args.resume_from_checkpoint and self.args.resume_from_checkpoint != "none":
            self._load_checkpoint()
        
        logger.info(f"✅ Model loaded: {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def _setup_data(self):
        """Setup data loaders for training and validation."""
        logger.info("📊 Setting up data loaders...")
        
        # Validate dataset structure
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path not found: {self.dataset_path}")
        
        # Check for 5-class DR structure
        required_splits = ['train', 'val']
        for split in required_splits:
            split_dir = self.dataset_path / split
            if not split_dir.exists():
                raise ValueError(f"Missing {split} directory in {self.dataset_path}")
            
            # Check classes 0-4
            for class_id in range(5):
                class_dir = split_dir / str(class_id)
                if not class_dir.exists():
                    logger.warning(f"Missing class {class_id} in {split} split")
        
        # Load data splits using the correct function
        logger.info("📊 Loading data splits for 5-class DR dataset...")
        train_data, val_data, test_data = create_data_splits_type1(
            dataset_path=str(self.dataset_path),
            num_classes=5,
            seed=42
        )
        
        # Get transforms
        train_transform = get_transforms(img_size=448, is_training=True)
        val_transform = get_transforms(img_size=448, is_training=False)
        
        # Create datasets
        self.train_dataset = DiabeticRetinopathyDataset(
            data_info=train_data,
            transform=train_transform,
            img_size=448
        )
        
        self.val_dataset = DiabeticRetinopathyDataset(
            data_info=val_data,
            transform=val_transform,
            img_size=448
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device == "mps" else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device == "mps" else False
        )
        
        logger.info(f"✅ Data loaded: {len(self.train_dataset)} train, {len(self.val_dataset)} val samples")
        
        # Log class distribution
        train_class_counts = {}
        for i in range(len(self.train_dataset)):
            sample = self.train_dataset[i]
            label = int(sample['dr_grade'])
            train_class_counts[label] = train_class_counts.get(label, 0) + 1
        
        logger.info("📊 Training class distribution:")
        for class_id, count in sorted(train_class_counts.items()):
            logger.info(f"   Class {class_id}: {count} samples")
    
    def _setup_optimizer(self):
        """Setup optimizer with medical-grade parameters."""
        logger.info("⚙️ Setting up optimizer...")
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            eps=1e-8
        )
        
        # Setup Medical-Grade Focal Loss with Nuclear Parameters
        if self.args.enable_focal_loss:
            # Medical-Grade Focal Loss implementation for severe class imbalance
            class MedicalGradeFocalLoss(torch.nn.Module):
                def __init__(self, alpha=4.0, gamma=6.0, num_classes=5, 
                           class_weight_severe=8.0, class_weight_pdr=6.0):
                    super().__init__()
                    self.alpha = alpha  # Nuclear alpha parameter
                    self.gamma = gamma  # Nuclear gamma parameter
                    self.num_classes = num_classes
                    
                    # Setup EXTREME medical-grade class weights
                    class_weights = torch.ones(num_classes)
                    if class_weight_severe > 0:
                        class_weights[3] = class_weight_severe  # Severe NPDR (Class 3)
                    if class_weight_pdr > 0:
                        class_weights[4] = class_weight_pdr    # PDR (Class 4)
                    
                    # Register class weights buffer and ensure proper device allocation
                    self.register_buffer('class_weights', class_weights)
                    logger.info(f"✅ NUCLEAR Focal Loss: α={alpha}, γ={gamma}")
                    logger.info(f"✅ EXTREME Class Weights: {class_weights.cpu().numpy()}")
                    logger.info(f"   - Severe NPDR (Class 3): {class_weight_severe}x priority")
                    logger.info(f"   - PDR (Class 4): {class_weight_pdr}x priority")
                
                def forward(self, inputs, targets):
                    # Base cross-entropy loss with class weights
                    ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
                    
                    # Focal loss modulation with NUCLEAR parameters
                    pt = torch.exp(-ce_loss)
                    focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
                    
                    return focal_loss.mean()
            
            # Use medical-grade parameters from command line
            focal_alpha = getattr(self.args, 'focal_loss_alpha', 4.0)
            focal_gamma = getattr(self.args, 'focal_loss_gamma', 6.0)
            class_weight_severe = getattr(self.args, 'class_weight_severe', 8.0)
            class_weight_pdr = getattr(self.args, 'class_weight_pdr', 6.0)
            
            self.criterion = MedicalGradeFocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                num_classes=5,
                class_weight_severe=class_weight_severe,
                class_weight_pdr=class_weight_pdr
            ).to(self.device)
            logger.info("✅ Using Medical-Grade NUCLEAR Focal Loss")
        else:
            # Use class weights for imbalanced dataset
            if self.args.enable_class_weights:
                # Calculate class weights from training data
                class_counts = torch.zeros(5)
                for i in range(len(self.train_dataset)):
                    sample = self.train_dataset[i]
                    label = int(sample['dr_grade'])
                    class_counts[label] += 1
                
                # Apply medical-grade class weights
                class_weights = torch.ones(5)
                class_weight_severe = getattr(self.args, 'class_weight_severe', 8.0)
                class_weight_pdr = getattr(self.args, 'class_weight_pdr', 6.0)
                
                if class_weight_severe > 0:
                    class_weights[3] = class_weight_severe
                if class_weight_pdr > 0:
                    class_weights[4] = class_weight_pdr
                
                class_weights = class_weights.to(self.device)
                
                self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(self.device)
                logger.info(f"✅ Using Medical-Grade Class Weights: {class_weights.cpu().numpy()}")
            else:
                self.criterion = torch.nn.CrossEntropyLoss()
        
        logger.info(f"✅ Optimizer configured: LR={self.args.learning_rate}, WD={self.args.weight_decay}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.args.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.num_epochs,
                eta_min=1e-6
            )
        elif self.args.scheduler == "polynomial":
            self.scheduler = torch.optim.lr_scheduler.PolynomialLR(
                self.optimizer,
                total_iters=self.args.num_epochs,
                power=0.9
            )
        else:
            self.scheduler = None
        
        logger.info(f"✅ Scheduler: {self.args.scheduler}")
    
    def _load_checkpoint(self):
        """Load model from checkpoint."""
        checkpoint_path = Path(self.args.resume_from_checkpoint)
        
        # Handle local vs absolute paths
        if not checkpoint_path.is_absolute():
            # Fix doubled path issue - use direct path if it contains 'results'
            if 'results' in str(checkpoint_path):
                checkpoint_path = Path(checkpoint_path)
            else:
                checkpoint_path = self.checkpoints_dir / checkpoint_path
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return
        
        try:
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            # Load with weights_only=False for compatibility with older PyTorch checkpoints
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Option 1 implementation: Skip optimizer loading for fresh parameters
            if self.args.fresh_optimizer:
                logger.info("💰 PRESERVING INVESTMENT: Loading model weights but using fresh optimizer")
                logger.info("   ✅ Model features preserved")
                logger.info("   ✅ Fresh optimizer with new parameters")
            else:
                # Load optimizer and training state
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.scheduler and 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
                self.train_losses = checkpoint.get('train_losses', [])
                self.val_losses = checkpoint.get('val_losses', [])
                self.val_accuracies = checkpoint.get('val_accuracies', [])
                self.medical_validations = checkpoint.get('medical_validations', [])
            
            logger.info(f"✅ Checkpoint loaded successfully")
            if not self.args.fresh_optimizer:
                logger.info(f"   - Resume from epoch: {self.start_epoch}")
                logger.info(f"   - Best validation accuracy: {self.best_val_accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Starting training from scratch")
    
    def _save_checkpoint(self, epoch: int, val_accuracy: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_accuracy': self.best_val_accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'medical_validations': self.medical_validations,
            'args': vars(self.args)
        }
        
        # Save regular checkpoint
        if epoch % self.args.checkpoint_frequency == 0:
            checkpoint_path = self.checkpoints_dir / f"epoch_{epoch:03d}_checkpoint.pth"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoints_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"🏆 Best model saved: {best_path}")
        
        # Save latest checkpoint
        latest_path = self.checkpoints_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch with detailed batch progress bars."""
        self.model.train()
        losses = AverageMeter()
        dr_losses = AverageMeter()
        accuracies = AverageMeter()
        
        # Calculate total batches for the progress bar
        total_batches = len(self.train_loader)
        
        # Accumulate gradients if specified
        accumulation_steps = getattr(self.args, 'gradient_accumulation_steps', 1)
        accumulated_loss = 0.0
        
        # Create detailed progress bar like Vertex AI training
        pbar = tqdm(
            enumerate(self.train_loader), 
            total=total_batches,
            desc="Training Batches",
            ncols=120,
            postfix={'Loss': '0.0000', 'DR_Loss': '0.0000', 'Acc': '0.000', 'LR': '0.00e+00'}
        )
        
        for batch_idx, batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['dr_grade'].to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            dr_logits = outputs['dr_logits']  # Extract main classification logits
            
            # Calculate DR loss (main classification loss)
            dr_loss = self.criterion(dr_logits, labels)
            
            # Total loss (same as dr_loss for now, but can include other losses)
            total_loss = dr_loss
            
            # Scale loss for gradient accumulation
            scaled_loss = total_loss / accumulation_steps
            scaled_loss.backward()
            
            accumulated_loss += scaled_loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(dr_logits.data, 1)
            accuracy = (predicted == labels).float().mean()
            
            # Update meters
            losses.update(total_loss.item(), images.size(0))
            dr_losses.update(dr_loss.item(), images.size(0))
            accuracies.update(accuracy.item(), images.size(0))
            
            # Perform optimizer step every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == total_batches:
                # Gradient clipping for stability
                if hasattr(self.args, 'gradient_clip_norm') and self.args.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip_norm)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                accumulated_loss = 0.0
            
            # Update progress bar with detailed metrics (like Vertex AI)
            current_lr = self.optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'DR_Loss': f'{dr_losses.avg:.4f}', 
                'Acc': f'{accuracies.avg:.3f}',
                'LR': f'{current_lr:.2e}'
            })
        
        pbar.close()
        
        return {
            'train_loss': losses.avg,
            'train_dr_loss': dr_losses.avg,
            'train_accuracy': accuracies.avg,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate one epoch."""
        self.model.eval()
        losses = AverageMeter()
        accuracies = AverageMeter()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="🔍 Validation")
            
            for batch in pbar:
                images = batch['image'].to(self.device)
                labels = batch['dr_grade'].to(self.device)
                
                outputs = self.model(images)
                dr_logits = outputs['dr_logits']  # Extract main classification logits
                loss = self.criterion(dr_logits, labels)
                
                # Get predictions and probabilities
                probabilities = F.softmax(dr_logits, dim=1)
                _, predicted = torch.max(dr_logits, 1)
                
                # Calculate accuracy
                accuracy = (predicted == labels).float().mean()
                
                # Update meters
                losses.update(loss.item(), images.size(0))
                accuracies.update(accuracy.item(), images.size(0))
                
                # Store for metrics calculation
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                pbar.set_postfix({
                    'Val_Loss': f'{losses.avg:.4f}',
                    'Val_Acc': f'{accuracies.avg:.3f}'
                })
        
        # Calculate medical-grade metrics
        if self.args.enable_medical_grade:
            medical_metrics = calculate_medical_metrics(
                np.array(all_labels),
                np.array(all_predictions),
                np.array(all_probabilities),
                num_classes=5
            )
        else:
            medical_metrics = {}
        
        return {
            'val_loss': losses.avg,
            'val_accuracy': accuracies.avg,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'medical_metrics': medical_metrics
        }
    
    def train(self):
        """Main training loop."""
        logger.info("🚀 Starting NUCLEAR FOCAL LOSS MLX training on Mac M4...")
        logger.info(f"📊 MEDICAL-GRADE Training Configuration:")
        logger.info(f"   - Model: MedSigLIP-448 (880M parameters)")
        logger.info(f"   - Device: {self.device}")
        logger.info(f"   - Learning Rate: {self.args.learning_rate}")
        logger.info(f"   - Weight Decay: {self.args.weight_decay}")
        logger.info(f"   - Dropout: {self.args.dropout}")
        logger.info(f"   - Batch Size: {self.args.batch_size}")
        logger.info(f"   - Gradient Accumulation: {getattr(self.args, 'gradient_accumulation_steps', 1)}")
        logger.info(f"   - Epochs: {self.args.num_epochs}")
        logger.info(f"   - Scheduler: {self.args.scheduler}")
        
        # Nuclear focal loss configuration
        if self.args.enable_focal_loss:
            focal_alpha = getattr(self.args, 'focal_loss_alpha', 4.0)
            focal_gamma = getattr(self.args, 'focal_loss_gamma', 6.0)
            class_weight_severe = getattr(self.args, 'class_weight_severe', 8.0)
            class_weight_pdr = getattr(self.args, 'class_weight_pdr', 6.0)
            
            logger.info(f"💡 NUCLEAR FOCAL LOSS PARAMETERS:")
            logger.info(f"   - Alpha (α): {focal_alpha} (MAXIMUM penalty)")
            logger.info(f"   - Gamma (γ): {focal_gamma} (EXTREME focusing)")
            logger.info(f"   - Severe NPDR weight: {class_weight_severe}x")
            logger.info(f"   - PDR weight: {class_weight_pdr}x")
            logger.info(f"   - Target: 90%+ validation accuracy breakthrough")
        
        logger.info("")
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.args.num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self.train_epoch(epoch + 1)
            self.train_losses.append(train_metrics['train_loss'])
            
            # Validation phase
            if (epoch + 1) % self.args.validation_frequency == 0:
                val_metrics = self.validate_epoch(epoch + 1)
                self.val_losses.append(val_metrics['val_loss'])
                self.val_accuracies.append(val_metrics['val_accuracy'])
                self.medical_validations.append(val_metrics['medical_metrics'])
                
                # Check for improvement
                is_best = val_metrics['val_accuracy'] > self.best_val_accuracy
                if is_best:
                    self.best_val_accuracy = val_metrics['val_accuracy']
                
                # Medical-grade validation check
                medical_grade_pass = "❌"
                if self.args.enable_medical_grade and val_metrics['medical_metrics']:
                    if val_metrics['val_accuracy'] >= 0.90:  # 90% threshold
                        medical_grade_pass = "✅"
                
                # Log epoch results in medical-grade format
                epoch_time = time.time() - epoch_start_time
                logger.info(f"🏥 Epoch {epoch + 1}/{self.args.num_epochs} COMPLETED:")
                logger.info(f"   ✅ Train Loss: {train_metrics['train_loss']:.4f} | DR_Loss: {train_metrics.get('train_dr_loss', train_metrics['train_loss']):.4f}")
                logger.info(f"   ✅ Train Acc: {train_metrics['train_accuracy']:.3f} | Val Acc: {val_metrics['val_accuracy']:.3f}")
                logger.info(f"   ✅ Learning Rate: {train_metrics['learning_rate']:.2e}")
                logger.info(f"   ✅ Medical Grade (≥90%): {medical_grade_pass}")
                logger.info(f"   ⏱️  Epoch Time: {epoch_time:.1f}s")
                
                # Log breakthrough progress if approaching 90%
                if val_metrics['val_accuracy'] > 0.85:
                    progress_to_90 = (val_metrics['val_accuracy'] - 0.8137) / (0.90 - 0.8137) * 100
                    logger.info(f"   🎯 BREAKTHROUGH PROGRESS: {progress_to_90:.1f}% to 90% medical-grade target")
                
                logger.info("")
                
                # Save checkpoint
                self._save_checkpoint(epoch + 1, val_metrics['val_accuracy'], is_best)
                
                # Early stopping check
                if self.early_stopping(val_metrics['val_loss'], self.model):
                    logger.info(f"Early stopping triggered after epoch {epoch + 1}")
                    break
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
        
        # Training completed
        total_time = time.time() - start_time
        logger.info(f"🎉 Training completed in {total_time/3600:.2f} hours")
        logger.info(f"🏆 Best validation accuracy: {self.best_val_accuracy:.4f}")
        
        # Save final results
        self._save_training_results()
        
        return {
            'best_val_accuracy': self.best_val_accuracy,
            'total_training_time': total_time,
            'final_epoch': epoch + 1
        }
    
    def _save_training_results(self):
        """Save comprehensive training results."""
        results = {
            'training_config': vars(self.args),
            'best_validation_accuracy': self.best_val_accuracy,
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_accuracies': self.val_accuracies,
                'medical_validations': self.medical_validations
            }
        }
        
        # Save results JSON
        results_path = self.outputs_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create training plots
        self._create_training_plots()
        
        logger.info(f"📊 Results saved to: {results_path}")
    
    def _create_training_plots(self):
        """Create training visualization plots."""
        if not self.train_losses or not self.val_accuracies:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MLX Training Results - MedSigLIP-448', fontsize=16)
        
        # Training/Validation Loss
        axes[0, 0].plot(self.train_losses, label='Training Loss', color='blue')
        if self.val_losses:
            val_epochs = np.arange(self.args.validation_frequency - 1, 
                                 len(self.train_losses), 
                                 self.args.validation_frequency)[:len(self.val_losses)]
            axes[0, 0].plot(val_epochs, self.val_losses, label='Validation Loss', color='red')
        axes[0, 0].set_title('Training/Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Validation Accuracy
        if self.val_accuracies:
            val_epochs = np.arange(self.args.validation_frequency - 1, 
                                 len(self.train_losses), 
                                 self.args.validation_frequency)[:len(self.val_accuracies)]
            axes[0, 1].plot(val_epochs, self.val_accuracies, label='Validation Accuracy', color='green')
            axes[0, 1].axhline(y=0.90, color='red', linestyle='--', label='Medical Grade (90%)')
            axes[0, 1].set_title('Validation Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Learning Rate Schedule
        if hasattr(self, 'lr_history'):
            axes[1, 0].plot(self.lr_history, label='Learning Rate', color='orange')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Medical Grade Progress
        if self.args.enable_medical_grade and self.medical_validations:
            medical_scores = [m.get('overall_accuracy', 0) for m in self.medical_validations if m]
            if medical_scores:
                val_epochs = np.arange(self.args.validation_frequency - 1, 
                                     len(self.train_losses), 
                                     self.args.validation_frequency)[:len(medical_scores)]
                axes[1, 1].plot(val_epochs, medical_scores, label='Medical Grade Score', color='purple')
                axes[1, 1].axhline(y=0.90, color='red', linestyle='--', label='Pass Threshold (90%)')
                axes[1, 1].set_title('Medical Grade Progress')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Score')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = self.outputs_dir / 'training_plots.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📈 Training plots saved to: {plot_path}")


def main():
    """Main function for MLX training."""
    parser = argparse.ArgumentParser(description="MLX Diabetic Retinopathy Trainer for Mac M4")
    
    # Dataset and paths
    parser.add_argument("--dataset-path", type=str, default="dataset3_augmented_resized",
                       help="Path to dataset folder")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Directory to save results, checkpoints, and outputs")
    
    # Model parameters
    parser.add_argument("--num-epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-3, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    
    # Training configuration
    parser.add_argument("--scheduler", type=str, default="polynomial", 
                       choices=["cosine", "polynomial", "none"], help="Learning rate scheduler")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, 
                       help="Gradient accumulation steps")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Warmup epochs")
    parser.add_argument("--validation-frequency", type=int, default=1, 
                       help="Run validation every N epochs")
    parser.add_argument("--checkpoint-frequency", type=int, default=2, 
                       help="Save checkpoint every N epochs")
    
    # Early stopping
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--min-delta", type=float, default=0.01, help="Minimum improvement delta")
    
    # Loss and optimization
    parser.add_argument("--enable-focal-loss", action="store_true", help="Use focal loss")
    parser.add_argument("--focal-loss-alpha", type=float, default=4.0,
                       help="Focal loss alpha parameter (nuclear: 4.0)")
    parser.add_argument("--focal-loss-gamma", type=float, default=6.0,
                       help="Focal loss gamma parameter (nuclear: 6.0)")
    parser.add_argument("--enable-class-weights", action="store_true", help="Use class weights")
    parser.add_argument("--class-weight-severe", type=float, default=8.0,
                       help="Class weight multiplier for Severe NPDR (Class 3)")
    parser.add_argument("--class-weight-pdr", type=float, default=6.0,
                       help="Class weight multiplier for PDR (Class 4)")
    parser.add_argument("--enable-medical-grade", action="store_true", 
                       help="Enable medical-grade validation")
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0, 
                       help="Gradient clipping norm")
    
    # Resume training
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--fresh-optimizer", action="store_true",
                       help="Use fresh optimizer (preserve model weights only)")
    
    # Experiment
    parser.add_argument("--experiment-name", type=str, default="mlx_medsiglip_local",
                       help="Experiment name")
    
    args = parser.parse_args()
    
    # Validate MLX availability
    if not MLX_AVAILABLE:
        logger.error("MLX not available. Install with: pip install mlx")
        return
    
    # Create trainer
    trainer = MLXDRTrainer(args)
    
    # Start training
    try:
        results = trainer.train()
        logger.info("🎉 Training completed successfully!")
        logger.info(f"🏆 Best accuracy: {results['best_val_accuracy']:.4f}")
        
    except KeyboardInterrupt:
        logger.info("⚠️ Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()