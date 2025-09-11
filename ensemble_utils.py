#!/usr/bin/env python3
"""
Ensemble Utility Functions for Diabetic Retinopathy Classification

This module provides utility functions for the multi-architecture ensemble system,
including visualization, model management, checkpoint handling, and medical-grade
validation utilities.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import cv2
from PIL import Image
import pandas as pd
from datetime import datetime
import logging
from google.cloud import storage
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnsembleCheckpointManager:
    """
    Manages ensemble model checkpoints with medical-grade validation and GCS integration.
    """
    
    def __init__(self, local_dir: str = "./ensemble_checkpoints", 
                 gcs_bucket: Optional[str] = None):
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.gcs_bucket = gcs_bucket
        
        # Initialize GCS client if bucket specified
        self.gcs_client = None
        if gcs_bucket:
            try:
                self.gcs_client = storage.Client()
                self.bucket = self.gcs_client.bucket(gcs_bucket)
                logger.info(f"✅ Connected to GCS bucket: {gcs_bucket}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to connect to GCS: {e}")
                self.gcs_client = None
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch: int, 
                       train_loss: float, val_loss: float, val_accuracy: float,
                       best_val_accuracy: float, individual_accuracies: Dict[str, float],
                       medical_validation: Dict[str, Any],
                       config: 'EnsembleConfig',
                       is_best: bool = False,
                       upload_to_gcs: bool = True) -> str:
        """
        Save comprehensive ensemble checkpoint with medical validation data.
        
        Args:
            model: Ensemble model
            optimizer: Optimizer state
            scheduler: Learning rate scheduler
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss
            val_accuracy: Validation accuracy
            best_val_accuracy: Best validation accuracy so far
            individual_accuracies: Dictionary of individual model accuracies
            medical_validation: Medical-grade validation metrics
            config: Ensemble configuration
            is_best: Whether this is the best checkpoint
            upload_to_gcs: Whether to upload to GCS
            
        Returns:
            Path to saved checkpoint
        """
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create checkpoint data
        checkpoint = {
            'epoch': epoch,
            'timestamp': timestamp,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            
            # Training metrics
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'best_val_accuracy': best_val_accuracy,
            
            # Individual model performance
            'individual_accuracies': individual_accuracies,
            
            # Medical validation
            'medical_validation': medical_validation,
            
            # Configuration
            'config': {
                'model': config.model.__dict__,
                'training': config.training.__dict__,
                'data': config.data.__dict__,
                'system': config.system.__dict__
            },
            
            # Model architecture info
            'ensemble_weights': config.model.ensemble_weights,
            'model_variants': {
                'efficientnet': config.model.efficientnet_variant,
                'resnet': config.model.resnet_variant,
                'densenet': config.model.densenet_variant
            },
            
            # Medical compliance
            'medical_grade_compliant': medical_validation.get('meets_medical_standards', False),
            'clinical_readiness': medical_validation.get('clinical_readiness', 'unknown')
        }
        
        # Generate filename
        if is_best:
            filename = f"best_ensemble_model_acc{val_accuracy:.4f}_epoch{epoch}.pth"
        else:
            filename = f"ensemble_checkpoint_epoch{epoch:03d}_{timestamp}.pth"
        
        local_path = self.local_dir / filename
        
        # Save locally
        torch.save(checkpoint, local_path)
        logger.info(f"💾 Saved checkpoint to: {local_path}")
        
        # Upload to GCS if requested and available
        if upload_to_gcs and self.gcs_client:
            gcs_path = f"ensemble_checkpoints/{filename}"
            try:
                blob = self.bucket.blob(gcs_path)
                blob.upload_from_filename(str(local_path))
                logger.info(f"☁️ Uploaded checkpoint to GCS: gs://{self.gcs_bucket}/{gcs_path}")
                
                # Save checkpoint info for tracking
                self._update_checkpoint_registry(filename, local_path, gcs_path, checkpoint)
                
            except Exception as e:
                logger.error(f"❌ Failed to upload to GCS: {e}")
        
        return str(local_path)
    
    def load_checkpoint(self, checkpoint_path: str, model, optimizer=None, scheduler=None,
                       device: str = "cuda") -> Dict[str, Any]:
        """
        Load ensemble checkpoint and restore training state.
        
        Args:
            checkpoint_path: Path to checkpoint (local or GCS)
            model: Ensemble model to load weights into
            optimizer: Optional optimizer to restore state
            scheduler: Optional scheduler to restore state
            device: Device to load tensors on
            
        Returns:
            Dictionary with loaded checkpoint information
        """
        
        # Handle GCS paths
        if checkpoint_path.startswith("gs://"):
            local_path = self._download_from_gcs(checkpoint_path)
        else:
            local_path = checkpoint_path
        
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Checkpoint not found: {local_path}")
        
        logger.info(f"📁 Loading checkpoint from: {local_path}")
        
        # Load checkpoint
        checkpoint = torch.load(local_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("✅ Loaded model state")
        
        # Load optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("✅ Loaded optimizer state")
        
        # Load scheduler state if provided
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("✅ Loaded scheduler state")
        
        # Extract training information
        training_info = {
            'epoch': checkpoint.get('epoch', 0),
            'val_accuracy': checkpoint.get('val_accuracy', 0.0),
            'best_val_accuracy': checkpoint.get('best_val_accuracy', 0.0),
            'individual_accuracies': checkpoint.get('individual_accuracies', {}),
            'medical_validation': checkpoint.get('medical_validation', {}),
            'medical_grade_compliant': checkpoint.get('medical_grade_compliant', False)
        }
        
        logger.info(f"📊 Checkpoint info: Epoch {training_info['epoch']}, "
                   f"Val Acc: {training_info['val_accuracy']:.4f}, "
                   f"Medical Grade: {training_info['medical_grade_compliant']}")
        
        return training_info
    
    def _download_from_gcs(self, gcs_path: str) -> str:
        """Download checkpoint from GCS to local storage."""
        if not self.gcs_client:
            raise RuntimeError("GCS client not initialized")
        
        # Parse GCS path
        if gcs_path.startswith("gs://"):
            bucket_name, blob_name = gcs_path[5:].split("/", 1)
        else:
            raise ValueError(f"Invalid GCS path: {gcs_path}")
        
        # Local filename
        filename = Path(blob_name).name
        local_path = self.local_dir / filename
        
        # Download if not already present
        if not local_path.exists():
            logger.info(f"📥 Downloading from GCS: {gcs_path}")
            bucket = self.gcs_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(str(local_path))
            logger.info(f"✅ Downloaded to: {local_path}")
        
        return str(local_path)
    
    def _update_checkpoint_registry(self, filename: str, local_path: Path, 
                                  gcs_path: str, checkpoint: Dict[str, Any]):
        """Maintain a registry of saved checkpoints for tracking."""
        registry_file = self.local_dir / "checkpoint_registry.json"
        
        # Load existing registry
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                registry = json.load(f)
        else:
            registry = {"checkpoints": []}
        
        # Add new entry
        entry = {
            "filename": filename,
            "local_path": str(local_path),
            "gcs_path": gcs_path,
            "timestamp": checkpoint['timestamp'],
            "epoch": checkpoint['epoch'],
            "val_accuracy": checkpoint['val_accuracy'],
            "best_val_accuracy": checkpoint['best_val_accuracy'],
            "medical_grade_compliant": checkpoint['medical_grade_compliant'],
            "individual_accuracies": checkpoint['individual_accuracies']
        }
        
        registry["checkpoints"].append(entry)
        
        # Save updated registry
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
    
    def list_checkpoints(self) -> pd.DataFrame:
        """List all saved checkpoints with their metrics."""
        registry_file = self.local_dir / "checkpoint_registry.json"
        
        if not registry_file.exists():
            return pd.DataFrame()
        
        with open(registry_file, 'r') as f:
            registry = json.load(f)
        
        if not registry["checkpoints"]:
            return pd.DataFrame()
        
        df = pd.DataFrame(registry["checkpoints"])
        df = df.sort_values('val_accuracy', ascending=False)
        return df

class MedicalVisualizationUtils:
    """
    Utilities for medical-grade visualizations and attention maps.
    """
    
    @staticmethod
    def plot_training_curves(train_losses: List[float], val_losses: List[float],
                           val_accuracies: List[float], individual_accuracies: Dict[str, List[float]],
                           save_path: Optional[str] = None):
        """
        Plot comprehensive training curves for ensemble model.
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses  
            val_accuracies: List of validation accuracies
            individual_accuracies: Dictionary mapping model names to accuracy lists
            save_path: Optional path to save the plot
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Ensemble accuracy
        axes[0, 1].plot(epochs, val_accuracies, 'g-', label='Ensemble Accuracy', linewidth=3)
        axes[0, 1].axhline(y=0.90, color='r', linestyle='--', alpha=0.7, label='Medical Threshold (90%)')
        axes[0, 1].axhline(y=0.9696, color='orange', linestyle='--', alpha=0.7, label='Research Target (96.96%)')
        axes[0, 1].set_title('Ensemble Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])
        
        # Individual model accuracies
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        for i, (model_name, accuracies) in enumerate(individual_accuracies.items()):
            if len(accuracies) == len(epochs):
                axes[1, 0].plot(epochs, accuracies, color=colors[i % len(colors)], 
                               label=model_name, linewidth=2)
        
        axes[1, 0].axhline(y=0.90, color='black', linestyle='--', alpha=0.5, label='Medical Threshold')
        axes[1, 0].set_title('Individual Model Accuracies')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
        
        # Learning rate (if available)
        axes[1, 1].text(0.5, 0.5, 'Learning Rate\n(Add LR tracking to enable)', 
                        ha='center', va='center', transform=axes[1, 1].transAxes,
                        fontsize=12, style='italic')
        axes[1, 1].set_title('Learning Rate Schedule')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Training curves saved to: {save_path}")
        
        plt.show()
    
    @staticmethod
    def create_medical_report_summary(results: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """
        Create a medical-grade summary report from evaluation results.
        
        Args:
            results: Evaluation results dictionary
            save_path: Optional path to save the report
            
        Returns:
            Formatted medical report as string
        """
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract key metrics
        ensemble_perf = results.get('ensemble_performance', {})
        overall = ensemble_perf.get('overall_metrics', {})
        medical_assess = results.get('medical_assessment', {})
        individual_models = results.get('individual_models', {})
        
        # Generate report
        report = f"""
MEDICAL-GRADE DIABETIC RETINOPATHY ENSEMBLE MODEL EVALUATION REPORT
=================================================================

Report Generated: {timestamp}
Model Type: Multi-Architecture Ensemble (EfficientNetB2 + ResNet50 + DenseNet121)

EXECUTIVE SUMMARY
-----------------
Overall Accuracy: {overall.get('accuracy', 0):.4f} ({overall.get('accuracy', 0)*100:.2f}%)
Clinical Grade: {medical_assess.get('clinical_assessment', {}).get('clinical_grade', 'Unknown')}
Medical Standards Compliance: {medical_assess.get('medical_grade_compliance', {}).get('all_criteria_met', False)}
Clinical Readiness: {medical_assess.get('clinical_assessment', {}).get('readiness_status', 'Unknown')}

PERFORMANCE METRICS
-------------------
Precision: {overall.get('precision', 0):.4f}
Recall (Sensitivity): {overall.get('recall', 0):.4f}
F1 Score: {overall.get('f1_score', 0):.4f}
AUC Score: {overall.get('auc', 0):.4f}

MEDICAL COMPLIANCE ASSESSMENT
-----------------------------
"""
        
        # Add medical compliance details
        compliance = medical_assess.get('medical_grade_compliance', {})
        performance_summary = medical_assess.get('performance_summary', {})
        
        report += f"""
✓ Accuracy Threshold (≥90%): {compliance.get('meets_accuracy_threshold', False)}
✓ Sensitivity Threshold (≥85%): {compliance.get('meets_sensitivity_threshold', False)}
✓ Specificity Threshold (≥90%): {compliance.get('meets_specificity_threshold', False)}
✓ AUC Threshold (≥85%): {compliance.get('meets_auc_threshold', False)}
✓ Research Target (≥96.96%): {compliance.get('meets_target_accuracy', False)}

Minimum Sensitivity: {performance_summary.get('min_sensitivity', 0):.4f}
Minimum Specificity: {performance_summary.get('min_specificity', 0):.4f}
Minimum AUC: {performance_summary.get('min_auc', 0):.4f}

INDIVIDUAL MODEL PERFORMANCE
----------------------------
"""
        
        # Add individual model results
        for model_name, metrics in individual_models.items():
            report += f"""
{model_name}:
  Accuracy: {metrics.get('accuracy', 0):.4f} ({metrics.get('accuracy', 0)*100:.2f}%)
  Precision: {metrics.get('precision', 0):.4f}
  Recall: {metrics.get('recall', 0):.4f}
  F1 Score: {metrics.get('f1_score', 0):.4f}
  AUC: {metrics.get('auc', 0):.4f}
"""
        
        # Add recommendations
        recommendations = medical_assess.get('clinical_assessment', {}).get('recommendations', [])
        if recommendations:
            report += "\nRECOMMendations:\n"
            report += "-" * 20 + "\n"
            for i, rec in enumerate(recommendations, 1):
                report += f"{i}. {rec}\n"
        
        # Add class-specific performance
        per_class = ensemble_perf.get('per_class_metrics', {})
        if per_class:
            report += "\nPER-CLASS PERFORMANCE\n"
            report += "-" * 21 + "\n"
            
            class_names = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"]
            precision = per_class.get('precision', [])
            recall = per_class.get('recall', [])
            specificity = per_class.get('specificity', [])
            
            for i, class_name in enumerate(class_names):
                if i < len(precision):
                    report += f"""
{class_name}:
  Precision: {precision[i]:.4f}
  Sensitivity: {recall[i]:.4f}
  Specificity: {specificity[i]:.4f}
"""
        
        report += f"""

CLINICAL INTERPRETATION
-----------------------
This ensemble model combines three state-of-the-art CNN architectures (EfficientNetB2, 
ResNet50, and DenseNet121) to achieve robust diabetic retinopathy classification. The 
model has been evaluated against medical-grade standards for diagnostic accuracy.

Current Status: {medical_assess.get('clinical_assessment', {}).get('clinical_grade', 'Unknown')}
Next Steps: {medical_assess.get('clinical_assessment', {}).get('readiness_status', 'Unknown')}

DISCLAIMER
----------
This evaluation is for research and development purposes. Clinical deployment requires 
additional validation, regulatory approval, and compliance with medical device standards.

Report End
==========
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"📄 Medical report saved to: {save_path}")
        
        return report

class EnsembleModelUtils:
    """
    Utility functions for ensemble model management and analysis.
    """
    
    @staticmethod
    def compare_ensemble_configurations(config1: 'EnsembleConfig', config2: 'EnsembleConfig') -> Dict[str, Any]:
        """Compare two ensemble configurations and highlight differences."""
        
        differences = {
            'model_differences': {},
            'training_differences': {},
            'data_differences': {},
            'system_differences': {}
        }
        
        # Compare model configs
        model1_dict = config1.model.__dict__
        model2_dict = config2.model.__dict__
        
        for key in set(model1_dict.keys()) | set(model2_dict.keys()):
            val1 = model1_dict.get(key)
            val2 = model2_dict.get(key)
            if val1 != val2:
                differences['model_differences'][key] = {'config1': val1, 'config2': val2}
        
        # Compare training configs
        train1_dict = config1.training.__dict__
        train2_dict = config2.training.__dict__
        
        for key in set(train1_dict.keys()) | set(train2_dict.keys()):
            val1 = train1_dict.get(key)
            val2 = train2_dict.get(key)
            if val1 != val2:
                differences['training_differences'][key] = {'config1': val1, 'config2': val2}
        
        # Compare data configs
        data1_dict = config1.data.__dict__
        data2_dict = config2.data.__dict__
        
        for key in set(data1_dict.keys()) | set(data2_dict.keys()):
            val1 = data1_dict.get(key)
            val2 = data2_dict.get(key)
            if val1 != val2:
                differences['data_differences'][key] = {'config1': val1, 'config2': val2}
        
        return differences
    
    @staticmethod
    def calculate_ensemble_efficiency(individual_accuracies: Dict[str, float], 
                                    ensemble_accuracy: float,
                                    individual_complexities: Optional[Dict[str, int]] = None) -> Dict[str, float]:
        """
        Calculate ensemble efficiency metrics.
        
        Args:
            individual_accuracies: Dictionary of individual model accuracies
            ensemble_accuracy: Ensemble accuracy
            individual_complexities: Optional dictionary of model parameter counts
            
        Returns:
            Dictionary with efficiency metrics
        """
        
        # Basic efficiency metrics
        best_individual = max(individual_accuracies.values())
        worst_individual = min(individual_accuracies.values())
        mean_individual = np.mean(list(individual_accuracies.values()))
        
        ensemble_gain = ensemble_accuracy - best_individual
        ensemble_lift = ensemble_accuracy - mean_individual
        
        efficiency = {
            'ensemble_accuracy': ensemble_accuracy,
            'best_individual_accuracy': best_individual,
            'worst_individual_accuracy': worst_individual,
            'mean_individual_accuracy': mean_individual,
            'ensemble_gain_over_best': ensemble_gain,
            'ensemble_lift_over_mean': ensemble_lift,
            'efficiency_ratio': ensemble_accuracy / mean_individual if mean_individual > 0 else 0,
            'diversity_score': np.std(list(individual_accuracies.values()))
        }
        
        # Add complexity-based metrics if available
        if individual_complexities:
            total_params = sum(individual_complexities.values())
            efficiency['total_parameters'] = total_params
            efficiency['accuracy_per_million_params'] = ensemble_accuracy / (total_params / 1e6)
        
        return efficiency
    
    @staticmethod
    def generate_ensemble_summary(model, config: 'EnsembleConfig') -> Dict[str, Any]:
        """Generate a comprehensive summary of the ensemble model."""
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Get individual model parameter counts
        individual_params = {}
        individual_params['efficientnetb2'] = sum(p.numel() for p in model.efficientnet_b2.parameters())
        individual_params['resnet50'] = sum(p.numel() for p in model.resnet50.parameters())
        individual_params['densenet121'] = sum(p.numel() for p in model.densenet121.parameters())
        
        summary = {
            'model_architecture': {
                'type': 'multi_architecture_ensemble',
                'individual_models': ['EfficientNetB2', 'ResNet50', 'DenseNet121'],
                'ensemble_method': 'weighted_averaging',
                'ensemble_weights': config.model.ensemble_weights,
                'num_classes': config.model.num_classes,
                'input_size': config.model.img_size
            },
            'parameters': {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'individual_model_parameters': individual_params,
                'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
            },
            'configuration': {
                'dropouts': {
                    'efficientnet': config.model.efficientnet_dropout,
                    'resnet': config.model.resnet_dropout,
                    'densenet': config.model.densenet_dropout
                },
                'learning_rates': {
                    'efficientnet': config.training.efficientnet_lr,
                    'resnet': config.training.resnet_lr,
                    'densenet': config.training.densenet_lr
                },
                'preprocessing': {
                    'clahe_enabled': config.data.enable_clahe,
                    'smote_enabled': config.data.enable_smote,
                    'medical_augmentation': config.data.enable_medical_augmentation
                }
            },
            'medical_targets': {
                'accuracy_threshold': config.system.medical_accuracy_threshold,
                'sensitivity_threshold': config.system.medical_sensitivity_threshold,
                'specificity_threshold': config.system.medical_specificity_threshold,
                'target_ensemble_accuracy': config.system.target_ensemble_accuracy
            }
        }
        
        return summary

def create_ensemble_analysis_report(checkpoint_path: str, output_dir: str = "./analysis_report") -> str:
    """
    Create a comprehensive analysis report for an ensemble checkpoint.
    
    Args:
        checkpoint_path: Path to the ensemble checkpoint
        output_dir: Directory to save the analysis report
        
    Returns:
        Path to the generated report
    """
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract information
    training_info = {
        'epoch': checkpoint.get('epoch', 0),
        'val_accuracy': checkpoint.get('val_accuracy', 0),
        'individual_accuracies': checkpoint.get('individual_accuracies', {}),
        'medical_validation': checkpoint.get('medical_validation', {}),
        'ensemble_weights': checkpoint.get('ensemble_weights', []),
        'model_variants': checkpoint.get('model_variants', {})
    }
    
    # Generate report
    report_content = f"""
ENSEMBLE MODEL ANALYSIS REPORT
==============================

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Checkpoint: {checkpoint_path}

TRAINING SUMMARY
----------------
Epoch: {training_info['epoch']}
Validation Accuracy: {training_info['val_accuracy']:.4f} ({training_info['val_accuracy']*100:.2f}%)

INDIVIDUAL MODEL PERFORMANCE
----------------------------
"""
    
    for model_name, accuracy in training_info['individual_accuracies'].items():
        report_content += f"{model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)\n"
    
    report_content += f"""

ENSEMBLE CONFIGURATION
----------------------
Model Weights: {training_info['ensemble_weights']}
Model Variants: {training_info['model_variants']}

MEDICAL VALIDATION
------------------
"""
    
    medical_val = training_info['medical_validation']
    if medical_val:
        report_content += f"Medical Grade Compliant: {medical_val.get('meets_medical_standards', 'Unknown')}\n"
        report_content += f"Clinical Readiness: {medical_val.get('clinical_readiness', 'Unknown')}\n"
    else:
        report_content += "No medical validation data available\n"
    
    # Save report
    report_path = os.path.join(output_dir, f"ensemble_analysis_{timestamp}.txt")
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    logger.info(f"📄 Analysis report saved to: {report_path}")
    return report_path

# Convenience functions for common operations
def quick_ensemble_evaluation(checkpoint_path: str, dataset_path: str, output_dir: str = "./quick_eval") -> Dict[str, Any]:
    """Quick evaluation of an ensemble checkpoint with minimal setup."""
    from ensemble_evaluator import evaluate_ensemble_checkpoint
    
    return evaluate_ensemble_checkpoint(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        batch_size=8  # Smaller batch for quick evaluation
    )

def compare_ensemble_checkpoints(checkpoint_paths: List[str], output_dir: str = "./comparison") -> str:
    """Compare multiple ensemble checkpoints and generate comparison report."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    comparison_data = []
    
    for i, checkpoint_path in enumerate(checkpoint_paths):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        data = {
            'checkpoint_id': i + 1,
            'path': checkpoint_path,
            'epoch': checkpoint.get('epoch', 0),
            'val_accuracy': checkpoint.get('val_accuracy', 0),
            'individual_accuracies': checkpoint.get('individual_accuracies', {}),
            'medical_compliant': checkpoint.get('medical_grade_compliant', False)
        }
        comparison_data.append(data)
    
    # Generate comparison report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"checkpoint_comparison_{timestamp}.txt")
    
    with open(report_path, 'w') as f:
        f.write("ENSEMBLE CHECKPOINT COMPARISON REPORT\n")
        f.write("=" * 40 + "\n\n")
        
        for data in comparison_data:
            f.write(f"Checkpoint {data['checkpoint_id']}:\n")
            f.write(f"  Path: {data['path']}\n")
            f.write(f"  Epoch: {data['epoch']}\n")
            f.write(f"  Validation Accuracy: {data['val_accuracy']:.4f}\n")
            f.write(f"  Medical Compliant: {data['medical_compliant']}\n")
            f.write(f"  Individual Accuracies: {data['individual_accuracies']}\n\n")
    
    logger.info(f"📊 Checkpoint comparison saved to: {report_path}")
    return report_path

if __name__ == "__main__":
    print("🔧 Ensemble Utilities Module")
    print("Available utilities:")
    print("- EnsembleCheckpointManager: Checkpoint management with GCS support")
    print("- MedicalVisualizationUtils: Medical-grade visualizations") 
    print("- EnsembleModelUtils: Model analysis and comparison utilities")
    print("- Quick evaluation and comparison functions")
    print("\nFor usage examples, see the individual class and function docstrings.")