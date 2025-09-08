import os
import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import PolynomialLR, LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import json
import logging
import tempfile
import shutil

# GCS imports for checkpoint saving
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    print("⚠️ Warning: google-cloud-storage not available. GCS checkpoint saving disabled.")

from models import DiabeticRetinopathyModel, calculate_medical_metrics, validate_medical_grade_performance, create_medical_loss_function
from utils import AverageMeter, EarlyStopping, save_results

# Mixed precision training imports
try:
    from torch.amp import autocast, GradScaler
    AMP_AVAILABLE = True
    USE_NEW_AMP_API = True
except ImportError:
    try:
        from torch.cuda.amp import autocast, GradScaler
        AMP_AVAILABLE = True
        USE_NEW_AMP_API = False
    except ImportError:
        AMP_AVAILABLE = False
        USE_NEW_AMP_API = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalGradeDRTrainer:
    """Medical-grade trainer for Phase 1 MedSigLIP-448 DR classification."""
    
    def __init__(self, 
                 model: DiabeticRetinopathyModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 learning_rate: float = 1e-4,
                 num_epochs: int = 50,
                 patience: int = 10,
                 use_mixed_precision: bool = True,
                 class_weights: torch.Tensor = None,
                 validation_frequency: int = 10,
                 checkpoint_frequency: int = 5,
                 gcs_bucket: str = "dr-data-2",
                 resume_from_checkpoint: str = None,
                 gradient_accumulation_steps: int = 1,
                 # Medical-grade parameters
                 enable_focal_loss: bool = True,
                 focal_loss_alpha: float = 2.0,
                 focal_loss_gamma: float = 4.0,
                 weight_decay: float = 1e-5,
                 scheduler: str = None,
                 max_grad_norm: float = 1.0):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.patience = patience
        self.validation_frequency = validation_frequency
        self.checkpoint_frequency = checkpoint_frequency
        self.gcs_bucket = gcs_bucket
        self.use_mixed_precision = use_mixed_precision and AMP_AVAILABLE
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Store medical-grade parameters
        self.enable_focal_loss = enable_focal_loss
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler
        self.max_grad_norm = max_grad_norm
        
        # Checkpoint and resume functionality  
        self.start_epoch = 0
        self.best_val_accuracy = 0.0
        self.previous_checkpoint_name = None  # Track previous checkpoint for cleanup
        self.resume_from_checkpoint = resume_from_checkpoint  # Store for later use
        
        # Setup optimizer with medical-grade learning rates
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Setup learning rate scheduler - BREAKTHROUGH CONFIGURATION
        if self.scheduler_type == 'none' or self.scheduler_type is None:
            print("🔧 SCHEDULER DISABLED - Using fixed learning rate for medical-grade training")
            self.scheduler = None
        elif self.scheduler_type == 'polynomial':
            print("🚀 BREAKTHROUGH SCHEDULER: Polynomial decay for 90%+ accuracy")
            # Polynomial decay: maintains higher LR longer, gentle decline
            self.scheduler = optim.lr_scheduler.PolynomialLR(
                self.optimizer,
                total_iters=120,  # Total epochs for gentle decay
                power=0.5  # Square root decay (gentle)
            )
        elif self.scheduler_type == 'linear':
            print("🚀 BREAKTHROUGH SCHEDULER: Linear decay for sustained learning")
            # Linear warmup then linear decay
            def linear_schedule(epoch):
                warmup_epochs = 10
                total_epochs = 120
                if epoch < warmup_epochs:
                    return epoch / warmup_epochs
                else:
                    return 1.0 - (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, linear_schedule)
        elif self.scheduler_type == 'validation_plateau':
            print("🎯 VALIDATION-BASED SCHEDULER: Adaptive reduction on plateau")
            # Reduce LR when validation accuracy plateaus
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # Maximize validation accuracy
                factor=0.7,  # Reduce by 30% on plateau
                patience=8,  # Wait 8 epochs before reducing
                threshold=0.005,  # Minimum improvement threshold
                min_lr=1e-6
            )
        elif self.scheduler_type == 'cosine_with_restarts':
            print("⚠️ COSINE RESTARTS: Using original aggressive scheduler")
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,  # Restart every 10 epochs
                T_mult=2,
                eta_min=1e-6
            )
        else:
            print(f"🔧 SCHEDULER ENABLED: {self.scheduler_type}")
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-6
            )
        
        # Medical-grade loss functions with configurable parameters
        if class_weights is not None:
            class_weights = class_weights.to(device)
        
        # Use the new medical-grade focal loss with configurable parameters
        self.dr_criterion = create_medical_loss_function(
            focal_loss=self.enable_focal_loss,
            focal_alpha=self.focal_loss_alpha,
            focal_gamma=self.focal_loss_gamma,
            class_weights=class_weights
        )
        self.referable_criterion = nn.CrossEntropyLoss()
        self.sight_threatening_criterion = nn.CrossEntropyLoss()
        self.confidence_criterion = nn.MSELoss()
        
        # Mixed precision scaler
        if self.use_mixed_precision:
            self.scaler = GradScaler('cuda' if USE_NEW_AMP_API else None)
            logger.info("✅ Mixed precision training enabled")
        else:
            logger.info("⚠️ Mixed precision training disabled")
        
        # Early stopping for medical validation
        self.early_stopping = EarlyStopping(
            patience=patience,
            min_delta=0.001,
            restore_best_weights=True
        )
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.medical_validations = []
        
        logger.info("✅ Medical-grade trainer initialized")
        logger.info(f"   - Device: {device}")
        logger.info(f"   - Learning rate: {learning_rate}")
        logger.info(f"   - Epochs: {num_epochs}")
        logger.info(f"   - Checkpoint frequency: {checkpoint_frequency} epochs")
        logger.info(f"   - GCS bucket: {gcs_bucket}")
        
        # Now that all components are initialized, check for resume
        if self.resume_from_checkpoint:
            # Explicit resume path provided
            logger.info(f"🔄 Explicit resume requested: {self.resume_from_checkpoint}")
            self._resume_from_checkpoint(self.resume_from_checkpoint)
        else:
            # Auto-detect latest checkpoint in GCS
            logger.info("🔍 Checking for existing checkpoints to auto-resume...")
            auto_checkpoint = self._detect_latest_checkpoint()
            if auto_checkpoint:
                logger.info(f"✅ Found existing checkpoint: {auto_checkpoint}")
                self._resume_from_checkpoint(auto_checkpoint)
            else:
                logger.info("🆕 No existing checkpoints found - starting fresh training")
    
    def _save_checkpoint_to_gcs(self, epoch: int, val_accuracy: float, is_best: bool = False):
        """Save model checkpoint to Google Cloud Storage with proper synchronization."""
        if not GCS_AVAILABLE:
            logger.warning("GCS not available - checkpoint saved locally only")
            return
        
        try:
            # CRITICAL FIX: Add synchronization logging to track parallel operations
            logger.info(f"🔄 Starting checkpoint save for epoch {epoch} (accuracy: {val_accuracy:.4f})")
            if is_best:
                logger.info("   ⭐ This is a BEST MODEL checkpoint - critical save operation")
            # Create checkpoint data
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                'val_accuracy': val_accuracy,
                'best_val_accuracy': self.best_val_accuracy,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_accuracies': self.val_accuracies,
                'medical_validations': self.medical_validations
            }
            
            if self.use_mixed_precision:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
                torch.save(checkpoint, tmp_file.name)
                tmp_file_path = tmp_file.name
            
            # Upload to GCS
            client = storage.Client()
            bucket = client.bucket(self.gcs_bucket)
            
            # Current checkpoint (epoch_{current-1}_checkpoint.pth for next iteration)
            current_checkpoint_name = f"checkpoints/epoch_{epoch:03d}_checkpoint.pth"
            current_blob = bucket.blob(current_checkpoint_name)
            current_blob.upload_from_filename(tmp_file_path)
            
            # Latest checkpoint (always current epoch)
            latest_blob = bucket.blob("checkpoints/latest_checkpoint.pth")
            latest_blob.upload_from_filename(tmp_file_path)
            
            # Best model checkpoint
            if is_best:
                best_blob = bucket.blob("checkpoints/best_model.pth")
                best_blob.upload_from_filename(tmp_file_path)
                logger.info(f"💾 New best model saved to GCS (accuracy: {val_accuracy:.4f})")
            
            # CRITICAL FIX: Proper completion logging for synchronization
            logger.info(f"✅ Checkpoint save COMPLETED for epoch {epoch} - safe to proceed")
            
            # Update tracking for next iteration
            self.previous_checkpoint_name = current_checkpoint_name
            
            # Cleanup local temp file
            os.unlink(tmp_file_path)
            
            logger.info(f"💾 Checkpoint saved to gs://{self.gcs_bucket}/{current_checkpoint_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint to GCS: {e}")
    
    def _cleanup_old_checkpoints(self, bucket, current_epoch: int):
        """Clean up old checkpoints, keeping only the last 5 epochs + best model for medical-grade safety."""
        try:
            # CRITICAL FIX: Add synchronization logging for cleanup operations
            logger.info(f"🔄 Starting checkpoint cleanup for epoch {current_epoch}")
            logger.info("   ⚠️ This cleanup is now SAFELY separated from training operations")
            
            # List all checkpoint blobs
            checkpoint_blobs = list(bucket.list_blobs(prefix="checkpoints/epoch_"))
            
            # Extract epoch numbers and sort
            epoch_blobs = []
            for blob in checkpoint_blobs:
                try:
                    # Extract epoch number from filename like "epoch_012_checkpoint.pth"
                    epoch_part = blob.name.split('/')[-1]  # Get filename
                    epoch_num_str = epoch_part.split('_')[1]  # Get epoch number part
                    epoch_num = int(epoch_num_str)
                    epoch_blobs.append((epoch_num, blob))
                except (IndexError, ValueError):
                    # Skip malformed filenames
                    continue
            
            # Sort by epoch number
            epoch_blobs.sort(key=lambda x: x[0])
            
            # FIXED: Keep checkpoints based on actual checkpoint_frequency pattern
            # For checkpoint_frequency=5: keep 5 most recent checkpoint epochs (e.g., 105, 110, 115, 120, 125)
            keep_count = 5
            epochs_to_keep = set()
            
            # Get the most recent actual checkpoint epochs (respects checkpoint_frequency)
            available_epochs = [epoch_num for epoch_num, _ in epoch_blobs]
            available_epochs.sort(reverse=True)  # Most recent first
            
            # Keep the last 5 actual checkpoint epochs
            epochs_to_keep = set(available_epochs[:keep_count])
            
            # SAFETY: Never delete checkpoints from current or previous epoch
            epochs_to_keep.add(current_epoch)
            if current_epoch > 0:
                epochs_to_keep.add(current_epoch - 1)
            
            # Delete old epoch checkpoints (only if more than keep_count + safety exist)
            deleted_count = 0
            if len(epoch_blobs) > keep_count + 2:  # +2 for safety buffer
                for epoch_num, blob in epoch_blobs:
                    if epoch_num not in epochs_to_keep:
                        try:
                            blob.delete()
                            deleted_count += 1
                            logger.info(f"🗑️ Deleted old checkpoint: {blob.name} (keeping recent: {sorted(epochs_to_keep)})")
                        except Exception as delete_error:
                            logger.warning(f"⚠️ Failed to delete {blob.name}: {delete_error}")
            
            if deleted_count > 0:
                logger.info(f"💾 Storage optimized: Deleted {deleted_count} old checkpoint(s)")
            else:
                logger.debug(f"💾 Keeping all {len(epoch_blobs)} checkpoints (within safety limit of {keep_count})")
            
            # CRITICAL FIX: Proper completion logging for synchronization
            logger.info(f"✅ Checkpoint cleanup COMPLETED for epoch {current_epoch} - no interference with training")
                
        except Exception as e:
            logger.warning(f"⚠️ Checkpoint cleanup failed (continuing anyway): {e}")
    
    def _detect_latest_checkpoint(self) -> str:
        """Auto-detect latest checkpoint in GCS for resuming training."""
        if not GCS_AVAILABLE:
            logger.debug("GCS not available - cannot auto-detect checkpoints")
            return None
        
        try:
            client = storage.Client()
            bucket = client.bucket(self.gcs_bucket)
            
            # Check for latest_checkpoint.pth first (most reliable)
            latest_blob = bucket.blob("checkpoints/latest_checkpoint.pth")
            if latest_blob.exists():
                return f"gs://{self.gcs_bucket}/checkpoints/latest_checkpoint.pth"
            
            # Fallback: Find most recent epoch checkpoint
            checkpoint_blobs = list(bucket.list_blobs(prefix="checkpoints/epoch_"))
            if checkpoint_blobs:
                # Parse epoch numbers and find the highest
                latest_epoch = -1
                latest_checkpoint = None
                
                for blob in checkpoint_blobs:
                    try:
                        # Extract epoch number from "checkpoints/epoch_059_checkpoint.pth"
                        filename = blob.name.split('/')[-1]
                        epoch_str = filename.split('_')[1]
                        epoch_num = int(epoch_str)
                        
                        if epoch_num > latest_epoch:
                            latest_epoch = epoch_num
                            latest_checkpoint = f"gs://{self.gcs_bucket}/{blob.name}"
                    except (IndexError, ValueError):
                        continue
                
                if latest_checkpoint:
                    logger.info(f"📂 Found latest epoch checkpoint: epoch {latest_epoch}")
                    return latest_checkpoint
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to detect checkpoints (continuing with fresh start): {e}")
            return None
    
    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from a checkpoint."""
        try:
            if checkpoint_path.startswith('gs://'):
                # Download from GCS
                if not GCS_AVAILABLE:
                    raise RuntimeError("GCS not available for checkpoint download")
                
                # Parse GCS path
                bucket_name = checkpoint_path.split('/')[2]
                blob_name = '/'.join(checkpoint_path.split('/')[3:])
                
                client = storage.Client()
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                
                # Download to temporary file
                with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
                    blob.download_to_filename(tmp_file.name)
                    local_path = tmp_file.name
            else:
                local_path = checkpoint_path
            
            # Load checkpoint (PyTorch 2.6+ compatibility fix)
            try:
                checkpoint = torch.load(local_path, map_location=self.device, weights_only=False)
            except Exception as load_error:
                # Fallback for older PyTorch versions or other loading issues
                logger.warning(f"Initial checkpoint load failed: {load_error}")
                checkpoint = torch.load(local_path, map_location=self.device)
            
            # Restore model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # 🔄 OPTION 1: PRESERVE INVESTMENT - Skip optimizer restore for fresh parameters
            logger.info("💰 PRESERVING INVESTMENT: Loading model weights but using fresh optimizer")
            logger.info("   ✅ Model features preserved (your $200 investment)")
            logger.info("   ✅ Fresh optimizer with nuclear anti-overfitting parameters")
            # Skip: self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Skip: scheduler restore to allow new parameters to take effect
            logger.info("   🚫 Skipping optimizer/scheduler restore to apply new parameters")
            
            if self.use_mixed_precision and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # Restore training state
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.val_accuracies = checkpoint.get('val_accuracies', [])
            self.medical_validations = checkpoint.get('medical_validations', [])
            
            # Cleanup if downloaded from GCS
            if checkpoint_path.startswith('gs://'):
                os.unlink(local_path)
            
            logger.info(f"✅ Resumed training from epoch {self.start_epoch}")
            logger.info(f"   - Previous epoch completed: {checkpoint['epoch']}")
            logger.info(f"   - Best validation accuracy: {self.best_val_accuracy:.4f}")
            logger.info(f"   - Training history restored: {len(self.train_losses)} epochs")
            logger.info(f"   - Will continue from epoch {self.start_epoch} to {self.num_epochs}")
            
        except Exception as e:
            logger.error(f"❌ Failed to resume from checkpoint: {e}")
            logger.info("⚠️ Starting training from scratch")
            # Reset to fresh state if resume fails
            self.start_epoch = 0
            self.best_val_accuracy = 0.0
            self.train_losses = []
            self.val_losses = []
            self.val_accuracies = []
            self.medical_validations = []
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch with medical-grade monitoring."""
        self.model.train()
        
        # Metrics tracking
        dr_loss_meter = AverageMeter()
        referable_loss_meter = AverageMeter()
        sight_threatening_loss_meter = AverageMeter()
        confidence_loss_meter = AverageMeter()
        total_loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        
        dr_predictions = []
        dr_targets = []
        
        # Enable training batch progress bar to show real-time progress
        progress_bar = tqdm(self.train_loader, desc="  Training Batches", leave=False, 
                           file=sys.stdout, disable=False, position=1)
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            targets = batch['dr_grade'].to(self.device)  # DR severity targets
            
            # Zero grad only at start of accumulation cycle
            if batch_idx % self.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            # Generate referable and sight-threatening targets from DR severity
            referable_targets = (targets >= 2).long()  # Classes 2,3,4 are referable
            sight_threatening_targets = (targets >= 3).long()  # Classes 3,4 are sight-threatening
            
            # Forward pass with mixed precision
            if self.use_mixed_precision:
                with autocast('cuda' if USE_NEW_AMP_API else None):
                    outputs = self.model(images)
                    
                    # Calculate losses
                    dr_loss = self.dr_criterion(outputs['dr_logits'], targets)
                    referable_loss = self.referable_criterion(outputs['referable_dr_logits'], referable_targets)
                    sight_threatening_loss = self.sight_threatening_criterion(outputs['sight_threatening_logits'], sight_threatening_targets)
                    
                    # Confidence loss (target = max probability of DR prediction)
                    dr_probs = torch.softmax(outputs['dr_logits'], dim=1)
                    confidence_targets = torch.max(dr_probs, dim=1)[0]
                    if 'grading_confidence' in outputs:
                        try:
                            confidence_pred = outputs['grading_confidence'].squeeze()
                            # CRITICAL FIX: Ensure shapes match for MSELoss
                            if confidence_pred.dim() == 0:  # scalar tensor
                                confidence_pred = confidence_pred.unsqueeze(0).expand_as(confidence_targets)
                            confidence_loss = self.confidence_criterion(confidence_pred, confidence_targets)
                        except Exception as e:
                            # Fallback: Skip confidence loss if tensor shape issues persist
                            print(f"⚠️ Confidence loss error: {e} - skipping confidence loss")
                            confidence_loss = torch.tensor(0.0, device=self.device)
                    else:
                        confidence_loss = torch.tensor(0.0, device=self.device)
                    
                    # Total loss with medical-grade weighting
                    total_loss = (
                        2.0 * dr_loss +  # Primary task gets highest weight
                        1.0 * referable_loss +
                        1.0 * sight_threatening_loss +
                        0.5 * confidence_loss
                    )
                    
                    # Scale loss by accumulation steps for proper averaging
                    total_loss = total_loss / self.gradient_accumulation_steps
                
                # Backward pass with mixed precision - accumulate gradients
                self.scaler.scale(total_loss).backward()
                
                # Update only every gradient_accumulation_steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Add gradient clipping for stability
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                outputs = self.model(images)
                
                # Calculate losses
                dr_loss = self.dr_criterion(outputs['dr_logits'], targets)
                referable_loss = self.referable_criterion(outputs['referable_dr_logits'], referable_targets)
                sight_threatening_loss = self.sight_threatening_criterion(outputs['sight_threatening_logits'], sight_threatening_targets)
                
                # Confidence loss
                dr_probs = torch.softmax(outputs['dr_logits'], dim=1)
                confidence_targets = torch.max(dr_probs, dim=1)[0]
                if 'grading_confidence' in outputs:
                    try:
                        confidence_pred = outputs['grading_confidence'].squeeze()
                        # CRITICAL FIX: Ensure shapes match for MSELoss
                        if confidence_pred.dim() == 0:  # scalar tensor
                            confidence_pred = confidence_pred.unsqueeze(0).expand_as(confidence_targets)
                        confidence_loss = self.confidence_criterion(confidence_pred, confidence_targets)
                    except Exception as e:
                        # Fallback: Skip confidence loss if tensor shape issues persist
                        print(f"⚠️ Confidence loss error: {e} - skipping confidence loss")
                        confidence_loss = torch.tensor(0.0, device=self.device)
                else:
                    confidence_loss = torch.tensor(0.0, device=self.device)
                
                # Total loss
                total_loss = (
                    2.0 * dr_loss +
                    1.0 * referable_loss +
                    1.0 * sight_threatening_loss +
                    0.5 * confidence_loss
                )
                
                # Scale loss by accumulation steps for proper averaging
                total_loss = total_loss / self.gradient_accumulation_steps
                
                # Backward pass - accumulate gradients
                total_loss.backward()
                
                # Update only every gradient_accumulation_steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Add gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                    self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs['dr_logits'], 1)
            accuracy = (predicted == targets).float().mean()
            
            # Update meters
            batch_size = images.size(0)
            dr_loss_meter.update(dr_loss.item(), batch_size)
            referable_loss_meter.update(referable_loss.item(), batch_size)
            sight_threatening_loss_meter.update(sight_threatening_loss.item(), batch_size)
            confidence_loss_meter.update(confidence_loss.item(), batch_size)
            total_loss_meter.update(total_loss.item(), batch_size)
            accuracy_meter.update(accuracy.item(), batch_size)
            
            # Log progress every 50 batches to show training progress
            if (batch_idx + 1) % 50 == 0:
                print(f"    Batch {batch_idx + 1}/{len(self.train_loader)} - "
                      f"Loss: {total_loss.item():.3f}, Acc: {accuracy.item():.3f}")
                
            # Update progress bar description with current metrics
            progress_bar.set_postfix({
                'Loss': f'{total_loss.item():.3f}',
                'Acc': f'{accuracy.item():.3f}',
                'Batch': f'{batch_idx + 1}/{len(self.train_loader)}'
            })
            
            # Store predictions for medical metrics
            dr_predictions.extend(predicted.cpu().numpy())
            dr_targets.extend(targets.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_loss_meter.avg:.4f}',
                'DR_Loss': f'{dr_loss_meter.avg:.4f}',
                'Acc': f'{accuracy_meter.avg:.3f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # Calculate medical-grade metrics for training
        medical_metrics = calculate_medical_metrics(
            np.array(dr_predictions), 
            np.array(dr_targets)
        )
        
        return {
            'total_loss': total_loss_meter.avg,
            'dr_loss': dr_loss_meter.avg,
            'referable_loss': referable_loss_meter.avg,
            'sight_threatening_loss': sight_threatening_loss_meter.avg,
            'confidence_loss': confidence_loss_meter.avg,
            'accuracy': accuracy_meter.avg,
            'medical_metrics': medical_metrics
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate one epoch with medical-grade evaluation."""
        self.model.eval()
        
        # Metrics tracking
        dr_loss_meter = AverageMeter()
        referable_loss_meter = AverageMeter()
        sight_threatening_loss_meter = AverageMeter()
        confidence_loss_meter = AverageMeter()
        total_loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        
        dr_predictions = []
        dr_targets = []
        confidence_scores = []
        
        # Disable validation batch progress bar for cleaner epoch-level logging  
        progress_bar = tqdm(self.val_loader, desc="  Validation Batches", leave=False,
                           file=sys.stdout, disable=True, position=1)
        
        with torch.no_grad():
            for batch in progress_bar:
                images = batch['image'].to(self.device)
                targets = batch['dr_grade'].to(self.device)
                
                # Generate targets
                referable_targets = (targets >= 2).long()
                sight_threatening_targets = (targets >= 3).long()
                
                # Forward pass with mixed precision
                if self.use_mixed_precision:
                    with autocast('cuda' if USE_NEW_AMP_API else None):
                        outputs = self.model(images)
                        
                        # Calculate losses inside autocast context
                        dr_loss = self.dr_criterion(outputs['dr_logits'], targets)
                        referable_loss = self.referable_criterion(outputs['referable_dr_logits'], referable_targets)
                        sight_threatening_loss = self.sight_threatening_criterion(outputs['sight_threatening_logits'], sight_threatening_targets)
                else:
                    outputs = self.model(images)
                    
                    # Calculate losses
                    dr_loss = self.dr_criterion(outputs['dr_logits'], targets)
                    referable_loss = self.referable_criterion(outputs['referable_dr_logits'], referable_targets)
                    sight_threatening_loss = self.sight_threatening_criterion(outputs['sight_threatening_logits'], sight_threatening_targets)
                
                # Confidence loss (handle mixed precision)
                if self.use_mixed_precision:
                    with autocast('cuda' if USE_NEW_AMP_API else None):
                        dr_probs = torch.softmax(outputs['dr_logits'], dim=1)
                        confidence_targets = torch.max(dr_probs, dim=1)[0]
                        if 'grading_confidence' in outputs:
                            try:
                                confidence_pred = outputs['grading_confidence'].squeeze()
                                # CRITICAL FIX: Ensure shapes match for MSELoss
                                if confidence_pred.dim() == 0:  # scalar tensor
                                    confidence_pred = confidence_pred.unsqueeze(0).expand_as(confidence_targets)
                                confidence_loss = self.confidence_criterion(confidence_pred, confidence_targets)
                            except Exception as e:
                                # Fallback: Skip confidence loss if tensor shape issues persist
                                print(f"⚠️ Validation confidence loss error: {e} - skipping confidence loss")
                                confidence_loss = torch.tensor(0.0, device=self.device)
                        else:
                            confidence_loss = torch.tensor(0.0, device=self.device)
                        
                        total_loss = (
                            2.0 * dr_loss +
                            1.0 * referable_loss +
                            1.0 * sight_threatening_loss +
                            0.5 * confidence_loss
                        )
                else:
                    dr_probs = torch.softmax(outputs['dr_logits'], dim=1)
                    confidence_targets = torch.max(dr_probs, dim=1)[0]
                    if 'grading_confidence' in outputs:
                        try:
                            confidence_pred = outputs['grading_confidence'].squeeze()
                            # CRITICAL FIX: Ensure shapes match for MSELoss
                            if confidence_pred.dim() == 0:  # scalar tensor
                                confidence_pred = confidence_pred.unsqueeze(0).expand_as(confidence_targets)
                            confidence_loss = self.confidence_criterion(confidence_pred, confidence_targets)
                        except Exception as e:
                            # Fallback: Skip confidence loss if tensor shape issues persist
                            print(f"⚠️ Validation confidence loss error: {e} - skipping confidence loss")
                            confidence_loss = torch.tensor(0.0, device=self.device)
                    else:
                        confidence_loss = torch.tensor(0.0, device=self.device)
                    
                    total_loss = (
                        2.0 * dr_loss +
                        1.0 * referable_loss +
                        1.0 * sight_threatening_loss +
                        0.5 * confidence_loss
                    )
                
                # Calculate accuracy
                _, predicted = torch.max(outputs['dr_logits'], 1)
                accuracy = (predicted == targets).float().mean()
                
                # Update meters
                batch_size = images.size(0)
                dr_loss_meter.update(dr_loss.item(), batch_size)
                referable_loss_meter.update(referable_loss.item(), batch_size)
                sight_threatening_loss_meter.update(sight_threatening_loss.item(), batch_size)
                confidence_loss_meter.update(confidence_loss.item(), batch_size)
                total_loss_meter.update(total_loss.item(), batch_size)
                accuracy_meter.update(accuracy.item(), batch_size)
                
                # Store predictions for medical metrics
                dr_predictions.extend(predicted.cpu().numpy())
                dr_targets.extend(targets.cpu().numpy())
                
                if 'grading_confidence' in outputs:
                    confidence_scores.extend(outputs['grading_confidence'].cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Val_Loss': f'{total_loss_meter.avg:.4f}',
                    'Val_Acc': f'{accuracy_meter.avg:.3f}'
                })
        
        # Calculate comprehensive medical metrics
        medical_metrics = calculate_medical_metrics(
            np.array(dr_predictions), 
            np.array(dr_targets)
        )
        
        # Validate medical-grade performance
        medical_validation = validate_medical_grade_performance(medical_metrics)
        
        return {
            'total_loss': total_loss_meter.avg,
            'dr_loss': dr_loss_meter.avg,
            'referable_loss': referable_loss_meter.avg,
            'sight_threatening_loss': sight_threatening_loss_meter.avg,
            'confidence_loss': confidence_loss_meter.avg,
            'accuracy': accuracy_meter.avg,
            'medical_metrics': medical_metrics,
            'medical_validation': medical_validation,
            'mean_confidence': np.mean(confidence_scores) if confidence_scores else 0.0
        }
    
    def train(self, save_dir: str = "checkpoints") -> Dict[str, any]:
        """
        Execute complete medical-grade training with validation.
        
        Returns:
            Training results and medical validation status
        """
        logger.info("🏥 Starting medical-grade Phase 1 training...")
        logger.info(f"   - Model: MedSigLIP-448")
        logger.info(f"   - Dataset: 5-class DR classification")
        logger.info(f"   - Medical standards: >90% accuracy, >85% sensitivity, >90% specificity")
        if self.start_epoch > 0:
            logger.info(f"   - RESUMING from epoch {self.start_epoch} (previous training completed up to epoch {self.start_epoch - 1})")
        else:
            logger.info(f"   - STARTING FRESH from epoch 1")
        
        os.makedirs(save_dir, exist_ok=True)
        best_val_accuracy = self.best_val_accuracy
        best_medical_validation = False
        training_history = []
        
        # Create simple epoch progress bar - resume from start_epoch
        total_epochs = self.num_epochs
        epoch_range = range(self.start_epoch, total_epochs)
        epoch_pbar = tqdm(epoch_range, desc="Epochs", 
                         file=sys.stdout, position=0, leave=True, 
                         bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}')
        
        for epoch in epoch_pbar:
            epoch_start_time = time.time()
            
            # Log epoch start for progress visibility
            print(f"\n🏥 Starting Epoch {epoch + 1}/{self.num_epochs}")
            
            # CRITICAL FIX: Move checkpoint cleanup AFTER all epoch operations complete
            # This prevents parallel execution bugs that corrupt training state
            
            # Training step (FIRST - no blocking operations before this)
            train_results = self.train_epoch()
            
            # Validation step (only every N epochs to save time)
            if (epoch + 1) % self.validation_frequency == 0 or epoch == 0:
                print(f"  🔍 Running validation...")
                val_results = self.validate_epoch()
                
                # Update learning rate - BREAKTHROUGH CONFIGURATION
                if self.scheduler is not None:
                    if self.scheduler_type == 'validation_plateau':
                        # For validation-based scheduler, pass validation accuracy
                        self.scheduler.step(val_results['accuracy'])
                        current_lr = self.optimizer.param_groups[0]['lr']
                        print(f"  🎯 LR Adaptation: {current_lr:.2e} (based on val_acc: {val_results['accuracy']:.4f})")
                    else:
                        # For epoch-based schedulers (polynomial, linear, cosine)
                        self.scheduler.step()
                        current_lr = self.optimizer.param_groups[0]['lr']
                        print(f"  🚀 LR Schedule: {current_lr:.2e} (epoch-based)")
                
                # Record metrics
                self.train_losses.append(train_results['total_loss'])
                self.val_losses.append(val_results['total_loss'])
                self.val_accuracies.append(val_results['accuracy'])
                self.medical_validations.append(val_results['medical_validation'])
                
                epoch_time = time.time() - epoch_start_time
                
                # Medical-grade validation status
                medical_pass = val_results['medical_validation']['medical_grade_pass']
                
                # Simple, clean logging as requested
                print(f"Training   - Loss: {train_results['total_loss']:.3f}, Accuracy: {train_results['accuracy']:.3f}")
                # CRITICAL FIX: Protect against corrupted validation metrics in logging
                if (val_results['total_loss'] > 0 and not np.isnan(val_results['total_loss']) and 
                    val_results['accuracy'] > 0 and not np.isnan(val_results['accuracy'])):
                    print(f"Validation - Loss: {val_results['total_loss']:.3f}, Accuracy: {val_results['accuracy']:.3f}")
                else:
                    print(f"Validation - Loss: CORRUPTED ({val_results['total_loss']}), Accuracy: CORRUPTED ({val_results['accuracy']})")
                    print(f"⚠️ Warning: Validation metrics corrupted by tensor shape error - using fallback values")
                    # Set fallback values to continue training without validation-dependent stopping
                    medical_pass = False
                    val_results['accuracy'] = 0.0  # Fallback accuracy to prevent further errors
                
                if val_results['accuracy'] >= 0.93 and medical_pass:
                    print(f"🎯 Medical-grade accuracy target achieved: {val_results['accuracy']:.3f} (≥93%)")
                    print(f"🏆 Training completed successfully at epoch {epoch+1}")
                    print(f"Time={epoch_time:.1f}s")
                    break
                elif medical_pass:
                    print(f"🏥 Medical Grade: ✅ PASS")
                else:
                    print(f"🏥 Medical Grade: ❌ FAIL")
                
                print(f"Time={epoch_time:.1f}s")
                print()  # Empty line for spacing
                
            else:
                # Training-only epoch (faster)
                epoch_time = time.time() - epoch_start_time
            
            # Save checkpoints (when validation runs)
            if 'val_results' in locals() and 'medical_pass' in locals():
                is_best = False
                # Safety check: ensure validation accuracy is valid (not NaN or corrupted)
                if val_results['accuracy'] > 0 and not np.isnan(val_results['accuracy']):
                    if val_results['accuracy'] > best_val_accuracy:
                        best_val_accuracy = val_results['accuracy']
                        self.best_val_accuracy = best_val_accuracy  # Update instance variable
                        if medical_pass:
                            best_medical_validation = True
                        is_best = True
                else:
                    print(f"⚠️ Warning: Invalid validation accuracy detected ({val_results['accuracy']}) - skipping checkpoint")
                
                # Save regular checkpoints to GCS every checkpoint_frequency epochs
                if (epoch + 1) % self.checkpoint_frequency == 0 or is_best:
                    self._save_checkpoint_to_gcs(epoch, val_results['accuracy'], is_best)
                    
                # Also save locally for backup
                if is_best:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                        'val_accuracy': val_results['accuracy'],
                        'best_val_accuracy': best_val_accuracy,
                        'medical_validation': val_results['medical_validation'],
                        'medical_metrics': val_results['medical_metrics'],
                        'training_config': {
                            'learning_rate': self.learning_rate,
                            'num_epochs': self.num_epochs,
                            'mixed_precision': self.use_mixed_precision
                        }
                    }
                    
                    if self.use_mixed_precision:
                        checkpoint['scaler_state_dict'] = self.scaler.state_dict()
                    
                    torch.save(checkpoint, os.path.join(save_dir, 'best_medical_model.pth'))
                
                # Store training history
                training_history.append({
                    'epoch': epoch + 1,
                    'train_loss': train_results['total_loss'],
                    'val_loss': val_results['total_loss'],
                    'val_accuracy': val_results['accuracy'],
                    'medical_metrics': val_results['medical_metrics'],
                    'medical_validation': val_results['medical_validation'],
                    'is_best': is_best
                })
                # Remove incorrect break statement - training should continue
            
            # Standard early stopping check (only when validation was run)
            # CRITICAL FIX: Only apply early stopping if validation metrics are valid
            if ('val_results' in locals() and 
                val_results['total_loss'] > 0 and 
                not np.isnan(val_results['total_loss']) and 
                not np.isinf(val_results['total_loss']) and
                val_results['accuracy'] > 0 and 
                not np.isnan(val_results['accuracy'])):
                
                if self.early_stopping(val_results['total_loss'], self.model):
                    logger.info(f"   ⏹️  Early stopping at epoch {epoch+1}")
                    logger.info(f"   📊 Final accuracy: {val_results['accuracy']:.3f}")
                    break
            elif 'val_results' in locals():
                print(f"⚠️ Warning: Corrupted validation metrics detected - continuing training")
                print(f"   Loss: {val_results['total_loss']}, Accuracy: {val_results['accuracy']}")
            
            # CRITICAL FIX: Checkpoint cleanup AFTER epoch is completely finished
            # This prevents parallel execution that corrupts training state
            # Only cleanup every 10 epochs to minimize GCS overhead
            if epoch > 0 and GCS_AVAILABLE and (epoch % 10 == 0):  
                try:
                    logger.info(f"🧹 Post-epoch cleanup at epoch {epoch + 1} (every 10 epochs)")
                    logger.info("   ⚠️ This cleanup runs AFTER epoch completion to prevent state corruption")
                    client = storage.Client()
                    bucket = client.bucket(self.gcs_bucket)
                    self._cleanup_old_checkpoints(bucket, epoch)  # Cleanup based on current epoch
                    logger.info(f"✅ Cleanup completed safely for epoch {epoch + 1}")
                except Exception as cleanup_error:
                    logger.warning(f"⚠️ Post-epoch cleanup failed (continuing safely): {cleanup_error}")
        
        # Save final model and training history
        final_checkpoint = {
            'final_epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'training_history': training_history,
            'best_medical_validation': best_medical_validation,
            'best_val_accuracy': best_val_accuracy
        }
        
        torch.save(final_checkpoint, os.path.join(save_dir, 'final_model.pth'))
        
        # Save training history as JSON (convert numpy types to native Python types)
        def convert_numpy_types(obj):
            """Convert numpy types to JSON-serializable Python types."""
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        serializable_history = convert_numpy_types(training_history)
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        logger.info("\n🎯 Training completed!")
        logger.info(f"   - Best validation accuracy: {best_val_accuracy:.3f}")
        logger.info(f"   - Medical-grade validation: {'✅ PASS' if best_medical_validation else '❌ FAIL'}")
        
        # Ensure we have final results even if last validation failed
        final_results = {
            'best_val_accuracy': best_val_accuracy,
            'medical_grade_pass': best_medical_validation,
            'training_history': training_history
        }
        
        # Add final metrics if validation ran successfully
        if 'val_results' in locals() and 'accuracy' in val_results:
            final_results.update({
                'final_medical_metrics': val_results.get('medical_metrics', {}),
                'final_medical_validation': val_results.get('medical_validation', {})
            })
        
        return final_results

def create_medical_grade_trainer(model, train_loader, val_loader, device, config=None):
    """Factory function to create medical-grade trainer with proper configuration."""
    
    # Default medical-grade training configuration
    training_config = {
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'patience': 10,
        'use_mixed_precision': True,
        'class_weights': None,
        'enable_focal_loss': True,
        'focal_loss_alpha': 2.0,
        'focal_loss_gamma': 4.0,
        'weight_decay': 1e-5,
        'scheduler': None
    }
    
    if config:
        training_config.update(config)
    
    trainer = MedicalGradeDRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        **training_config
    )
    
    return trainer

logger.info("✅ Medical-grade trainer loaded successfully")
logger.info("🏥 Ready for Phase 1 MedSigLIP-448 training")