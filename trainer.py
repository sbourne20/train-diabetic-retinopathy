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

from models import DiabeticRetinopathyModel, calculate_medical_metrics, validate_medical_grade_performance
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
                 resume_from_checkpoint: str = None):
        
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
        
        # Checkpoint and resume functionality  
        self.start_epoch = 0
        self.best_val_accuracy = 0.0
        self.previous_checkpoint_name = None  # Track previous checkpoint for cleanup
        self.resume_from_checkpoint = resume_from_checkpoint  # Store for later use
        
        # Setup optimizer with medical-grade learning rates
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,
            eta_min=1e-6
        )
        
        # Medical-grade loss functions
        if class_weights is not None:
            class_weights = class_weights.to(device)
        
        self.dr_criterion = nn.CrossEntropyLoss(weight=class_weights)
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
        """Save model checkpoint to Google Cloud Storage with storage optimization."""
        if not GCS_AVAILABLE:
            logger.warning("GCS not available - checkpoint saved locally only")
            return
        
        try:
            # Create checkpoint data
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
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
            
            # Note: Checkpoint cleanup moved to beginning of next epoch for serial safety
            # This ensures cleanup only happens AFTER the next epoch completes successfully
            
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
            
            # Keep the last 5 epochs for medical-grade training safety
            # This provides more recovery options and handles checkpoint saving gaps
            keep_count = 5
            epochs_to_keep = set()
            for i in range(keep_count):
                epoch_to_keep = current_epoch - i
                if epoch_to_keep >= 0:  # Don't keep negative epochs
                    epochs_to_keep.add(epoch_to_keep)
            
            # Delete old epoch checkpoints (only if more than 5 epochs exist)
            deleted_count = 0
            if len(epoch_blobs) > keep_count:
                for epoch_num, blob in epoch_blobs:
                    if epoch_num not in epochs_to_keep:
                        try:
                            blob.delete()
                            deleted_count += 1
                            logger.info(f"🗑️ Deleted old checkpoint: {blob.name}")
                        except Exception as delete_error:
                            logger.warning(f"⚠️ Failed to delete {blob.name}: {delete_error}")
            
            if deleted_count > 0:
                logger.info(f"💾 Storage optimized: Deleted {deleted_count} old checkpoint(s)")
            else:
                logger.debug(f"💾 Keeping all {len(epoch_blobs)} checkpoints (within safety limit of {keep_count})")
                
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
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
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
                    confidence_loss = self.confidence_criterion(
                        outputs['grading_confidence'].squeeze(), confidence_targets
                    ) if 'grading_confidence' in outputs else torch.tensor(0.0, device=self.device)
                    
                    # Total loss with medical-grade weighting
                    total_loss = (
                        2.0 * dr_loss +  # Primary task gets highest weight
                        1.0 * referable_loss +
                        1.0 * sight_threatening_loss +
                        0.5 * confidence_loss
                    )
                
                # Backward pass with mixed precision
                self.scaler.scale(total_loss).backward()
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
                confidence_loss = self.confidence_criterion(
                    outputs['grading_confidence'].squeeze(), confidence_targets
                ) if 'grading_confidence' in outputs else torch.tensor(0.0, device=self.device)
                
                # Total loss
                total_loss = (
                    2.0 * dr_loss +
                    1.0 * referable_loss +
                    1.0 * sight_threatening_loss +
                    0.5 * confidence_loss
                )
                
                # Backward pass
                total_loss.backward()
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
                        confidence_loss = self.confidence_criterion(
                            outputs['grading_confidence'].squeeze(), confidence_targets
                        ) if 'grading_confidence' in outputs else torch.tensor(0.0, device=self.device)
                        
                        total_loss = (
                            2.0 * dr_loss +
                            1.0 * referable_loss +
                            1.0 * sight_threatening_loss +
                            0.5 * confidence_loss
                        )
                else:
                    dr_probs = torch.softmax(outputs['dr_logits'], dim=1)
                    confidence_targets = torch.max(dr_probs, dim=1)[0]
                    confidence_loss = self.confidence_criterion(
                        outputs['grading_confidence'].squeeze(), confidence_targets
                    ) if 'grading_confidence' in outputs else torch.tensor(0.0, device=self.device)
                    
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
            
            # Serial checkpoint cleanup: Only cleanup old checkpoints AFTER previous epoch completed successfully
            # This ensures we never delete checkpoints while training is in progress
            if epoch > 0 and GCS_AVAILABLE:  # Skip cleanup for first epoch
                try:
                    client = storage.Client()
                    bucket = client.bucket(self.gcs_bucket)
                    self._cleanup_old_checkpoints(bucket, epoch - 1)  # Cleanup based on previous epoch
                except Exception as cleanup_error:
                    logger.warning(f"⚠️ Checkpoint cleanup failed (continuing safely): {cleanup_error}")
            
            # Training step
            train_results = self.train_epoch()
            
            # Validation step (only every N epochs to save time)
            if (epoch + 1) % self.validation_frequency == 0 or epoch == 0:
                print(f"  🔍 Running validation...")
                val_results = self.validate_epoch()
                
                # Update learning rate
                self.scheduler.step()
                
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
                print(f"Validation - Loss: {val_results['total_loss']:.3f}, Accuracy: {val_results['accuracy']:.3f}")
                
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
                if medical_pass and val_results['accuracy'] > best_val_accuracy:
                    best_val_accuracy = val_results['accuracy']
                    self.best_val_accuracy = best_val_accuracy  # Update instance variable
                    best_medical_validation = True
                    is_best = True
                
                # Save regular checkpoints to GCS every checkpoint_frequency epochs
                if (epoch + 1) % self.checkpoint_frequency == 0 or is_best:
                    self._save_checkpoint_to_gcs(epoch, val_results['accuracy'], is_best)
                    
                # Also save locally for backup
                if is_best:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
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
            if 'val_results' in locals() and self.early_stopping(val_results['total_loss'], self.model):
                logger.info(f"   ⏹️  Early stopping at epoch {epoch+1}")
                logger.info(f"   📊 Final accuracy: {val_results['accuracy']:.3f}")
                break
        
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
        
        return {
            'best_val_accuracy': best_val_accuracy,
            'medical_grade_pass': best_medical_validation,
            'training_history': training_history,
            'final_medical_metrics': val_results['medical_metrics'],
            'final_medical_validation': val_results['medical_validation']
        }

def create_medical_grade_trainer(model, train_loader, val_loader, device, config=None):
    """Factory function to create medical-grade trainer with proper configuration."""
    
    # Default medical-grade training configuration
    training_config = {
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'patience': 10,
        'use_mixed_precision': True,
        'class_weights': None
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