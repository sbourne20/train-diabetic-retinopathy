import os
import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import wandb
import json

from models import DiabeticRetinopathyModel
from utils import AverageMeter, EarlyStopping, calculate_metrics, save_results
from config import Config

# Mixed precision training imports
try:
    # Try new API first (PyTorch 2.1+)
    from torch.amp import autocast, GradScaler
    AMP_AVAILABLE = True
    USE_NEW_AMP_API = True
except ImportError:
    try:
        # Fallback to old API
        from torch.cuda.amp import autocast, GradScaler
        AMP_AVAILABLE = True
        USE_NEW_AMP_API = False
    except ImportError:
        AMP_AVAILABLE = False
        USE_NEW_AMP_API = False

class DRTrainer:
    """Trainer class for diabetic retinopathy model with medical reasoning."""
    
    def __init__(self, 
                 model: DiabeticRetinopathyModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Config,
                 rg_class_weights: torch.Tensor = None,
                 me_class_weights: torch.Tensor = None,
                 dr_class_weights: torch.Tensor = None):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup device and performance optimizations
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Enable performance optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
            # Advanced memory management
            torch.cuda.empty_cache()  # Clear cache at start
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for better performance
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
            
            # Enable memory efficient attention (PyTorch 2.0+)
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
                print("✓ Flash attention enabled")
            if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                print("✓ Memory efficient attention enabled")
            if hasattr(torch.backends.cuda, 'enable_math_sdp'):
                torch.backends.cuda.enable_math_sdp(True)  # Fallback option
        
        # Mixed precision training setup
        self.use_amp = getattr(config.training, 'enable_amp', True) and AMP_AVAILABLE and torch.cuda.is_available()
        if self.use_amp:
            if USE_NEW_AMP_API:
                self.scaler = GradScaler('cuda')
            else:
                self.scaler = GradScaler()
            print("✓ Mixed precision training enabled")
        else:
            self.scaler = None
            print("Mixed precision training disabled")
        
        # Gradient checkpointing for memory efficiency
        if getattr(config.training, 'gradient_checkpointing', True):
            try:
                # Enable gradient checkpointing for backbone
                if hasattr(self.model.backbone, 'gradient_checkpointing_enable'):
                    self.model.backbone.gradient_checkpointing_enable()
                    print("✓ Gradient checkpointing enabled")
                elif hasattr(self.model.backbone, 'set_grad_checkpointing'):
                    self.model.backbone.set_grad_checkpointing(enable=True)
                    print("✓ Gradient checkpointing enabled")
                else:
                    # For custom models, try to apply to transformer blocks
                    if hasattr(self.model.backbone, 'blocks'):
                        for block in self.model.backbone.blocks:
                            if hasattr(block, 'gradient_checkpointing'):
                                block.gradient_checkpointing = True
                        print("✓ Gradient checkpointing enabled on transformer blocks")
            except Exception as e:
                print(f"Gradient checkpointing setup failed: {e}")
        
        # Model compilation for PyTorch 2.0+
        if getattr(config.training, 'compile_model', True) and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='default')
                print("✓ Model compiled with torch.compile()")
            except Exception as e:
                print(f"Model compilation failed: {e}")
                print("Continuing without compilation...")
        
        # Gradient accumulation
        self.gradient_accumulation_steps = getattr(config.training, 'gradient_accumulation_steps', 1)
        
        # Enhanced loss functions with class weights
        if rg_class_weights is not None:
            rg_class_weights = rg_class_weights.to(self.device)
        if me_class_weights is not None:
            me_class_weights = me_class_weights.to(self.device)
        if dr_class_weights is not None:
            dr_class_weights = dr_class_weights.to(self.device)
            
        self.rg_criterion = nn.CrossEntropyLoss(weight=rg_class_weights)
        self.me_criterion = nn.CrossEntropyLoss(weight=me_class_weights)
        self.dr_criterion = nn.CrossEntropyLoss(weight=dr_class_weights)
        
        # Enhanced classification losses (using CrossEntropyLoss for 2-class outputs)
        self.referable_criterion = nn.CrossEntropyLoss()
        self.sight_threatening_criterion = nn.CrossEntropyLoss()
        self.pdr_activity_criterion = nn.CrossEntropyLoss()
        
        # ETDRS rule component losses (using CrossEntropyLoss for 2-class outputs)
        self.hemorrhages_4q_criterion = nn.CrossEntropyLoss()
        self.venous_beading_2q_criterion = nn.CrossEntropyLoss()
        self.irma_1q_criterion = nn.CrossEntropyLoss()
        self.meets_421_criterion = nn.CrossEntropyLoss()
        
        # Detailed findings losses
        self.nvd_area_criterion = nn.CrossEntropyLoss()
        self.nve_area_criterion = nn.CrossEntropyLoss()
        self.nv_activity_criterion = nn.CrossEntropyLoss()
        
        # Quality and confidence losses
        self.image_quality_criterion = nn.MSELoss()
        self.confidence_criterion = nn.MSELoss()
        
        # Optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Training utilities
        self.early_stopping = EarlyStopping(
            patience=config.training.patience,
            min_delta=config.training.min_delta
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_metrics = {}
        
        # Progressive unfreezing tracking
        self.frozen_layers = set()
        self._freeze_backbone()
        
        # Initialize wandb if enabled
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.experiment_name,
                config=config.__dict__
            )
        
    def _compute_enhanced_losses(self, batch, outputs):
        """Compute enhanced losses for medical reasoning."""
        enhanced_loss = 0.0
        
        # Add enhanced classification losses if available in batch
        if 'referable_dr' in batch:
            referable_labels = batch['referable_dr'].to(self.device)
            if 'referable_dr_logits' in outputs:
                referable_loss = self.referable_criterion(
                    outputs['referable_dr_logits'], 
                    referable_labels.long()
                )
                enhanced_loss += getattr(self.config.training, 'referable_loss_weight', 0.5) * referable_loss
        
        if 'hemorrhages_4q' in batch:
            hemorrhages_labels = batch['hemorrhages_4q'].to(self.device)
            if 'hemorrhages_4q_logits' in outputs:
                hemorrhages_loss = self.hemorrhages_4q_criterion(
                    outputs['hemorrhages_4q_logits'],
                    hemorrhages_labels.long()
                )
                enhanced_loss += 0.3 * hemorrhages_loss
        
        # Add other enhanced losses if available
        etdrs_components = ['venous_beading_2q', 'irma_1q', 'meets_421']
        for component in etdrs_components:
            if component in batch and f'{component}_logits' in outputs:
                labels = batch[component].to(self.device)
                criterion = getattr(self, f'{component}_criterion')
                loss = criterion(outputs[f'{component}_logits'], labels.long())
                enhanced_loss += 0.3 * loss
        
        # Image quality loss
        if 'image_quality' in batch and 'image_quality_score' in outputs:
            quality_labels = batch['image_quality'].to(self.device)
            quality_loss = self.image_quality_criterion(
                outputs['image_quality_score'].squeeze(),
                quality_labels.float()
            )
            enhanced_loss += 0.2 * quality_loss
        
        # Confidence loss
        if 'confidence_target' in batch and 'grading_confidence' in outputs:
            confidence_labels = batch['confidence_target'].to(self.device)
            confidence_loss = self.confidence_criterion(
                outputs['grading_confidence'].squeeze(),
                confidence_labels.float()
            )
            enhanced_loss += getattr(self.config.training, 'confidence_loss_weight', 0.3) * confidence_loss
        
        return enhanced_loss
    
    def _get_memory_stats(self):
        """Get GPU memory usage statistics."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            return {
                'allocated': allocated,
                'reserved': reserved, 
                'max_allocated': max_allocated
            }
        return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}
    
    def _optimize_memory(self):
        """Optimize GPU memory usage."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Force garbage collection
            import gc
            gc.collect()
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer with different learning rates for different components."""
        
        # Different learning rates for different components
        backbone_params = list(self.model.backbone.parameters())
        classifier_params = list(self.model.classifier.parameters())
        reasoning_params = list(self.model.reasoning_module.parameters())
        
        param_groups = [
            {'params': backbone_params, 'lr': self.config.training.learning_rate * 0.1},  # Lower LR for pretrained
            {'params': classifier_params, 'lr': self.config.training.learning_rate},
            {'params': reasoning_params, 'lr': self.config.training.learning_rate * 0.5}  # Medium LR for language
        ]
        
        return optim.AdamW(
            param_groups,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
    
    def _setup_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Setup enhanced learning rate scheduler with warmup."""
        
        total_steps = len(self.train_loader) * self.config.training.num_epochs
        warmup_steps = len(self.train_loader) * self.config.training.warmup_epochs
        
        # Enhanced schedulers with proper warmup
        if self.config.training.scheduler_type == 'cosine':
            # Cosine annealing with warmup
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=total_steps - warmup_steps,
                eta_min=self.config.training.learning_rate * 0.01  # 1% of base LR
            )
        elif self.config.training.scheduler_type == 'cosine_restarts':
            # Cosine annealing with warm restarts
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=total_steps // 4,  # First restart after 25% of training
                T_mult=2,  # Double the cycle length after each restart
                eta_min=self.config.training.learning_rate * 0.01
            )
        elif self.config.training.scheduler_type == 'linear':
            scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=warmup_steps
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        
        # Add warmup if specified
        if warmup_steps > 0 and self.config.training.scheduler_type in ['cosine', 'cosine_restarts']:
            from torch.optim.lr_scheduler import SequentialLR, LinearLR
            
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,  # Start at 1% of base LR
                end_factor=1.0,     # Reach full LR
                total_iters=warmup_steps
            )
            
            # Combine warmup + main scheduler
            scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[warmup_steps]
            )
        
        return scheduler
    
    def _freeze_backbone(self):
        """Freeze backbone parameters for progressive unfreezing."""
        for name, param in self.model.backbone.named_parameters():
            param.requires_grad = False
            self.frozen_layers.add(name)
    
    def _progressive_unfreeze(self, epoch: int):
        """Progressively unfreeze backbone layers."""
        if epoch >= self.config.training.freeze_backbone_epochs:
            unfreeze_rate = self.config.training.unfreeze_rate
            layers_to_unfreeze = (epoch - self.config.training.freeze_backbone_epochs) // unfreeze_rate
            
            # Get all frozen layer names
            frozen_layer_names = sorted(list(self.frozen_layers))
            
            # Unfreeze layers from the end (top layers first)
            for i in range(min(layers_to_unfreeze, len(frozen_layer_names))):
                layer_name = frozen_layer_names[-(i+1)]
                for name, param in self.model.backbone.named_parameters():
                    if name == layer_name and not param.requires_grad:
                        param.requires_grad = True
                        self.frozen_layers.remove(name)
                        print(f"Unfrozen layer: {name}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.model.train()
        
        # Meters for tracking metrics
        total_loss_meter = AverageMeter()
        rg_loss_meter = AverageMeter()
        me_loss_meter = AverageMeter()
        dr_loss_meter = AverageMeter()
        reasoning_loss_meter = AverageMeter()
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", file=sys.stdout)
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            
            # Handle different dataset types
            if 'dr_grade' in batch:
                # Dataset type 1: single DR classification
                dr_labels = batch['dr_grade'].to(self.device)
                rg_labels = None
                me_labels = None
            else:
                # Dataset type 0: RG/ME classification
                rg_labels = batch['rg_grade'].to(self.device)
                me_labels = batch['me_grade'].to(self.device)
                dr_labels = None
            
            # Zero gradients (only every gradient_accumulation_steps)
            if batch_idx % self.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            # Mixed precision forward pass with proper loss computation
            if self.use_amp:
                if USE_NEW_AMP_API:
                    autocast_context = autocast('cuda')
                else:
                    autocast_context = autocast()
                    
                with autocast_context:
                    outputs = self.model(images)
                    
                    # Compute losses inside autocast context for proper precision handling
                    if dr_labels is not None:
                        # Dataset type 1: DR classification using RG head
                        dr_loss = self.dr_criterion(outputs['rg_logits'], dr_labels)
                        classification_loss = dr_loss
                    else:
                        # Dataset type 0: RG/ME classification
                        rg_loss = self.rg_criterion(outputs['rg_logits'], rg_labels)
                        me_loss = self.me_criterion(outputs['me_logits'], me_labels)
                        classification_loss = (
                            self.config.training.rg_loss_weight * rg_loss +
                            self.config.training.me_loss_weight * me_loss
                        )
            else:
                outputs = self.model(images)
                
                # Standard loss computation
                if dr_labels is not None:
                    # Dataset type 1: DR classification using RG head
                    dr_loss = self.dr_criterion(outputs['rg_logits'], dr_labels)
                    classification_loss = dr_loss
                else:
                    # Dataset type 0: RG/ME classification
                    rg_loss = self.rg_criterion(outputs['rg_logits'], rg_labels)
                    me_loss = self.me_criterion(outputs['me_logits'], me_labels)
                    classification_loss = (
                        self.config.training.rg_loss_weight * rg_loss +
                        self.config.training.me_loss_weight * me_loss
                    )
            
            # Enhanced losses (computed with proper precision handling)
            if self.use_amp:
                if USE_NEW_AMP_API:
                    autocast_context = autocast('cuda')
                else:
                    autocast_context = autocast()
                    
                with autocast_context:
                    enhanced_loss = self._compute_enhanced_losses(batch, outputs)
            else:
                enhanced_loss = self._compute_enhanced_losses(batch, outputs)
            
            # Reasoning loss (if available)
            reasoning_loss = outputs.get('reasoning_loss')
            if reasoning_loss is not None:
                total_loss = classification_loss + enhanced_loss + 0.1 * reasoning_loss
                reasoning_loss_meter.update(reasoning_loss.item())
            else:
                total_loss = classification_loss + enhanced_loss
                reasoning_loss_meter.update(0.0)
            
            # Scale loss for gradient accumulation
            total_loss = total_loss / self.gradient_accumulation_steps
            
            # Mixed precision backward pass
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                
                # Gradient clipping and optimizer step every gradient_accumulation_steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or batch_idx == len(self.train_loader) - 1:
                    # Gradient clipping with scaler
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                   max_norm=getattr(self.config.training, 'max_grad_norm', 1.0))
                    
                    # Optimizer step and scaler update
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                total_loss.backward()
                
                # Standard gradient clipping and optimizer step
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or batch_idx == len(self.train_loader) - 1:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                   max_norm=getattr(self.config.training, 'max_grad_norm', 1.0))
                    self.optimizer.step()
            
            # Update meters based on dataset type (scale back for display)
            actual_loss = total_loss.item() * self.gradient_accumulation_steps
            total_loss_meter.update(actual_loss)
            
            if dr_labels is not None:
                # Dataset type 1: DR classification
                dr_loss_meter.update(dr_loss.item())
                
                # Update progress bar less frequently for performance
                log_freq = getattr(self.config.training, 'log_frequency', 50)
                if batch_idx % log_freq == 0:
                    pbar.set_postfix({
                        'Total Loss': f'{total_loss_meter.avg:.4f}',
                        'DR Loss': f'{dr_loss_meter.avg:.4f}',
                        'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                    })
            else:
                # Dataset type 0: RG/ME classification
                rg_loss_meter.update(rg_loss.item())
                me_loss_meter.update(me_loss.item())
                
                # Update progress bar less frequently for performance
                log_freq = getattr(self.config.training, 'log_frequency', 50)
                if batch_idx % log_freq == 0:
                    pbar.set_postfix({
                        'Total Loss': f'{total_loss_meter.avg:.4f}',
                        'RG Loss': f'{rg_loss_meter.avg:.4f}',
                        'ME Loss': f'{me_loss_meter.avg:.4f}',
                        'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                    })
        
        # Update scheduler
        if hasattr(self.scheduler, 'step'):
            self.scheduler.step()
        
        # Return metrics based on what was actually computed
        metrics = {
            'total_loss': total_loss_meter.avg,
            'reasoning_loss': reasoning_loss_meter.avg
        }
        
        # Add dataset-specific metrics
        if dr_loss_meter.count > 0:
            metrics['dr_loss'] = dr_loss_meter.avg
        if rg_loss_meter.count > 0:
            metrics['rg_loss'] = rg_loss_meter.avg
        if me_loss_meter.count > 0:
            metrics['me_loss'] = me_loss_meter.avg
            
        return metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        
        self.model.eval()
        
        # Meters and prediction storage
        total_loss_meter = AverageMeter()
        rg_loss_meter = AverageMeter()
        me_loss_meter = AverageMeter()
        dr_loss_meter = AverageMeter()
        
        rg_predictions, rg_targets = [], []
        me_predictions, me_targets = [], []
        dr_predictions, dr_targets = [], []
        rg_probabilities, me_probabilities, dr_probabilities = [], [], []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Validation {epoch}", file=sys.stdout)
            
            for batch in pbar:
                # Move to device
                images = batch['image'].to(self.device)
                
                # Handle different dataset types
                if 'dr_grade' in batch:
                    # Dataset type 1: single DR classification
                    dr_labels = batch['dr_grade'].to(self.device)
                    rg_labels = None
                    me_labels = None
                else:
                    # Dataset type 0: RG/ME classification
                    rg_labels = batch['rg_grade'].to(self.device)
                    me_labels = batch['me_grade'].to(self.device)
                    dr_labels = None
                
                # Mixed precision forward pass for validation
                if self.use_amp:
                    if USE_NEW_AMP_API:
                        autocast_context = autocast('cuda')
                    else:
                        autocast_context = autocast()
                        
                    with autocast_context:
                        outputs = self.model(images)
                else:
                    outputs = self.model(images)
                
                # Compute losses based on dataset type
                if dr_labels is not None:
                    # Dataset type 1: DR classification using RG head
                    dr_loss = self.dr_criterion(outputs['rg_logits'], dr_labels)
                    total_loss = dr_loss
                    
                    # Update meters
                    total_loss_meter.update(total_loss.item())
                    dr_loss_meter.update(dr_loss.item())
                    
                    # Store predictions and targets
                    dr_prob = torch.softmax(outputs['rg_logits'], dim=1)
                    dr_predictions.extend(dr_prob.argmax(dim=1).cpu().numpy())
                    dr_targets.extend(dr_labels.cpu().numpy())
                    dr_probabilities.extend(dr_prob.cpu().numpy())
                    
                else:
                    # Dataset type 0: RG/ME classification
                    rg_loss = self.rg_criterion(outputs['rg_logits'], rg_labels)
                    me_loss = self.me_criterion(outputs['me_logits'], me_labels)
                    
                    total_loss = (
                        self.config.training.rg_loss_weight * rg_loss +
                        self.config.training.me_loss_weight * me_loss
                    )
                    
                    # Update meters
                    total_loss_meter.update(total_loss.item())
                    rg_loss_meter.update(rg_loss.item())
                    me_loss_meter.update(me_loss.item())
                    
                    # Store predictions and targets
                    rg_prob = torch.softmax(outputs['rg_logits'], dim=1)
                    me_prob = torch.softmax(outputs['me_logits'], dim=1)
                    
                    rg_pred = torch.argmax(rg_prob, dim=1)
                    me_pred = torch.argmax(me_prob, dim=1)
                    
                    rg_predictions.extend(rg_pred.cpu().numpy())
                    me_predictions.extend(me_pred.cpu().numpy())
                    rg_targets.extend(rg_labels.cpu().numpy())
                    me_targets.extend(me_labels.cpu().numpy())
                    rg_probabilities.extend(rg_prob.cpu().numpy())
                    me_probabilities.extend(me_prob.cpu().numpy())
                
                # Update progress bar
                if dr_labels is not None:
                    pbar.set_postfix({
                        'Val Loss': f'{total_loss_meter.avg:.4f}',
                        'DR Loss': f'{dr_loss_meter.avg:.4f}'
                    })
                else:
                    pbar.set_postfix({
                        'Val Loss': f'{total_loss_meter.avg:.4f}',
                        'RG Loss': f'{rg_loss_meter.avg:.4f}',
                        'ME Loss': f'{me_loss_meter.avg:.4f}'
                    })
        
        # Calculate metrics based on dataset type
        if dr_predictions:
            # Dataset type 1: DR metrics
            dr_metrics = calculate_metrics(
                np.array(dr_targets), 
                np.array(dr_predictions),
                np.array(dr_probabilities)
            )
            
            return {
                'total_loss': total_loss_meter.avg,
                'dr_loss': dr_loss_meter.avg,
                'dr_accuracy': dr_metrics['accuracy'],
                'dr_auc': dr_metrics.get('auc', 0.0),
                'dr_f1': dr_metrics.get('f1', 0.0),
                'dr_precision': dr_metrics.get('precision', 0.0),
                'dr_recall': dr_metrics.get('recall', 0.0),
                'val_accuracy': dr_metrics['accuracy']  # Primary metric for monitoring
            }
        else:
            # Dataset type 0: RG/ME metrics
            rg_metrics = calculate_metrics(
                np.array(rg_targets), 
                np.array(rg_predictions),
                np.array(rg_probabilities)
            )
            
            me_metrics = calculate_metrics(
                np.array(me_targets), 
                np.array(me_predictions),
                np.array(me_probabilities)
            )
            
            return {
                'total_loss': total_loss_meter.avg,
                'rg_loss': rg_loss_meter.avg,
                'me_loss': me_loss_meter.avg,
                'rg_accuracy': rg_metrics['accuracy'],
                'me_accuracy': me_metrics['accuracy'],
                'rg_auc': rg_metrics.get('auc_macro', 0.0),
                'me_auc': me_metrics.get('auc_macro', 0.0),
                'rg_predictions': rg_predictions,
                'me_predictions': me_predictions,
                'rg_targets': rg_targets,
                'me_targets': me_targets,
                'val_accuracy': (rg_metrics['accuracy'] + me_metrics['accuracy']) / 2
            }
    
    def train(self) -> Dict[str, List[float]]:
        """Main training loop."""
        
        print(f"Starting training for {self.config.training.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Batch size: {self.config.data.batch_size}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        
        # Performance monitoring
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'val_rg_accuracies': [],
            'val_me_accuracies': [],
            'val_rg_aucs': [],
            'val_me_aucs': [],
            'epoch_times': [],
            'gpu_memory_used': []
        }
        
        for epoch in range(1, self.config.training.num_epochs + 1):
            start_time = time.time()
            
            # Progressive unfreezing
            self._progressive_unfreeze(epoch)
            
            # Training
            train_start = time.time()
            train_metrics = self.train_epoch(epoch)
            train_time = time.time() - train_start
            
            # Validation (check validation frequency)
            val_freq = getattr(self.config.training, 'validation_frequency', 1)
            if epoch % val_freq == 0:
                val_start = time.time()
                val_metrics = self.validate_epoch(epoch)
                val_time = time.time() - val_start
            else:
                # Skip validation, use previous metrics or defaults
                val_metrics = getattr(self, '_last_val_metrics', {
                    'total_loss': float('inf'),
                    'val_accuracy': 0.0,
                    'rg_accuracy': 0.0, 'me_accuracy': 0.0,
                    'rg_auc': 0.0, 'me_auc': 0.0,
                    'rg_loss': 0.0, 'me_loss': 0.0
                })
                val_time = 0.0
            
            # Store last validation metrics for skipped validations
            if epoch % val_freq == 0:
                self._last_val_metrics = val_metrics
            
            # Store metrics
            training_history['train_losses'].append(train_metrics['total_loss'])
            training_history['val_losses'].append(val_metrics['total_loss'])
            training_history['val_rg_accuracies'].append(val_metrics['rg_accuracy'])
            training_history['val_me_accuracies'].append(val_metrics['me_accuracy'])
            training_history['val_rg_aucs'].append(val_metrics['rg_auc'])
            training_history['val_me_aucs'].append(val_metrics['me_auc'])
            
            # Log to wandb
            if self.config.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['total_loss'],
                    'val_loss': val_metrics['total_loss'],
                    'val_rg_accuracy': val_metrics['rg_accuracy'],
                    'val_me_accuracy': val_metrics['me_accuracy'],
                    'val_rg_auc': val_metrics['rg_auc'],
                    'val_me_auc': val_metrics['me_auc'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Performance monitoring
            epoch_time = time.time() - start_time
            training_history['epoch_times'].append(epoch_time)
            
            # Enhanced GPU memory monitoring
            memory_stats = self._get_memory_stats()
            if torch.cuda.is_available():
                gpu_memory = memory_stats['max_allocated']
                training_history['gpu_memory_used'].append(gpu_memory)
                torch.cuda.reset_peak_memory_stats()
                
                # Periodic memory optimization
                if epoch % 5 == 0:
                    self._optimize_memory()
            
            # Print epoch summary with enhanced performance metrics
            print(f"\nEpoch {epoch}/{self.config.training.num_epochs}")
            print(f"Time: {epoch_time:.1f}s (Train: {train_time:.1f}s, Val: {val_time:.1f}s)")
            print(f"Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"Val Loss: {val_metrics['total_loss']:.4f}")
            
            # Only print detailed metrics for epochs that had validation
            if epoch % val_freq == 0:
                print(f"RG Accuracy: {val_metrics['rg_accuracy']:.4f}")
                print(f"ME Accuracy: {val_metrics['me_accuracy']:.4f}")
                print(f"RG AUC: {val_metrics['rg_auc']:.4f}")
                print(f"ME AUC: {val_metrics['me_auc']:.4f}")
            
            # Enhanced memory reporting
            if torch.cuda.is_available():
                print(f"GPU Memory: {memory_stats['allocated']:.1f}GB allocated, {memory_stats['max_allocated']:.1f}GB peak")
            
            # Save best model
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.best_metrics = val_metrics.copy()
                self.save_checkpoint(epoch, is_best=True)
                print("✓ Best model saved!")
            
            # Early stopping
            if self.early_stopping(val_metrics['total_loss'], self.model):
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Regular checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        # Restore best weights
        self.early_stopping.restore_weights(self.model)
        
        return training_history
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir, 
            f"checkpoint_epoch_{epoch}.pth"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(
                self.config.checkpoint_dir,
                "best_model.pth"
            )
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint['epoch']