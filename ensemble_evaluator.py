#!/usr/bin/env python3
"""
Medical-Grade Ensemble Model Evaluator for Diabetic Retinopathy Classification

This module provides comprehensive evaluation capabilities for the multi-architecture
ensemble system, including medical-grade metrics, confusion matrices, ROC curves,
and clinical performance assessment.
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Import ensemble components
from ensemble_models import DRMultiArchitectureEnsemble
from ensemble_config import EnsembleConfig, create_medical_grade_config
from ensemble_dataset import DRDataset, create_ensemble_data_loaders

class MedicalGradeEvaluator:
    """
    Comprehensive medical-grade evaluation for ensemble DR classification models.
    
    Provides clinical-standard metrics including sensitivity, specificity,
    PPV, NPV, AUC, and medical compliance assessment.
    """
    
    def __init__(self, config: EnsembleConfig, device: str = "cuda"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.class_names = [
            "No DR", 
            "Mild NPDR", 
            "Moderate NPDR", 
            "Severe NPDR", 
            "PDR"
        ]
        
        # Medical-grade thresholds
        self.medical_thresholds = {
            'accuracy': 0.90,           # 90% minimum accuracy
            'sensitivity': 0.85,        # 85% minimum per-class sensitivity
            'specificity': 0.90,        # 90% minimum per-class specificity
            'auc': 0.85,               # 85% minimum AUC per class
            'target_accuracy': 0.9696  # Research target: 96.96%
        }
        
        # Results storage
        self.evaluation_results = {}
        self.detailed_metrics = {}
        self.clinical_assessment = {}
    
    def load_ensemble_model(self, checkpoint_path: str) -> DRMultiArchitectureEnsemble:
        """Load trained ensemble model from checkpoint."""
        print(f"📁 Loading ensemble model from: {checkpoint_path}")
        
        model = DRMultiArchitectureEnsemble(
            num_classes=self.config.model.num_classes,
            dropout=0.3,  # Will be overridden by loaded weights
            pretrained=False,  # Loading trained weights
            model_weights=self.config.model.ensemble_weights
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded model state from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Loaded model state directly")
        
        model.to(self.device)
        model.eval()
        return model
    
    def evaluate_model(self, model: DRMultiArchitectureEnsemble, 
                      test_loader, save_dir: str = "./evaluation_results") -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of the ensemble model.
        
        Args:
            model: Trained ensemble model
            test_loader: Test data loader
            save_dir: Directory to save evaluation results
            
        Returns:
            Complete evaluation results dictionary
        """
        print("🔬 Starting comprehensive ensemble evaluation...")
        
        # Create results directory
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Basic predictions and metrics
        predictions, probabilities, true_labels = self._get_predictions(model, test_loader)
        
        # 2. Individual model performance
        individual_results = self._evaluate_individual_models(model, test_loader)
        
        # 3. Ensemble performance
        ensemble_results = self._calculate_ensemble_metrics(predictions, probabilities, true_labels)
        
        # 4. Medical-grade assessment
        medical_assessment = self._medical_grade_assessment(ensemble_results)
        
        # 5. Clinical analysis
        clinical_analysis = self._clinical_performance_analysis(predictions, probabilities, true_labels)
        
        # 6. Generate visualizations
        self._generate_evaluation_plots(predictions, probabilities, true_labels, save_dir)
        
        # 7. Compile comprehensive results
        comprehensive_results = {
            'evaluation_summary': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'model_type': 'multi_architecture_ensemble',
                'test_samples': len(true_labels),
                'device': str(self.device)
            },
            'individual_models': individual_results,
            'ensemble_performance': ensemble_results,
            'medical_assessment': medical_assessment,
            'clinical_analysis': clinical_analysis,
            'file_paths': {
                'confusion_matrix': f"{save_dir}/confusion_matrix.png",
                'roc_curves': f"{save_dir}/roc_curves.png",
                'class_performance': f"{save_dir}/class_performance.png",
                'individual_comparison': f"{save_dir}/individual_model_comparison.png"
            }
        }
        
        # 8. Save results
        self._save_evaluation_results(comprehensive_results, save_dir)
        
        print("✅ Comprehensive evaluation completed!")
        return comprehensive_results
    
    def _get_predictions(self, model: DRMultiArchitectureEnsemble, 
                        test_loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get ensemble predictions and individual model outputs."""
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_individual_probs = []
        
        print("🔍 Generating predictions on test set...")
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Get ensemble outputs and individual model outputs
                ensemble_logits, individual_outputs = model(images, return_individual=True)
                ensemble_probs = F.softmax(ensemble_logits, dim=1)
                ensemble_preds = torch.argmax(ensemble_probs, dim=1)
                
                # Convert individual outputs to probabilities
                individual_probs = {}
                for model_name, logits in individual_outputs.items():
                    individual_probs[model_name] = F.softmax(logits, dim=1)
                
                # Store results
                all_predictions.extend(ensemble_preds.cpu().numpy())
                all_probabilities.extend(ensemble_probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_individual_probs.append(individual_probs)
                
                if (batch_idx + 1) % 20 == 0:
                    print(f"   Processed {batch_idx + 1}/{len(test_loader)} batches")
        
        # Combine individual probabilities across batches
        combined_individual_probs = {}
        for model_name in all_individual_probs[0].keys():
            combined_individual_probs[model_name] = torch.cat([
                batch[model_name] for batch in all_individual_probs
            ]).cpu().numpy()
        
        # Store for later use
        self.individual_probabilities = combined_individual_probs
        
        return (np.array(all_predictions), 
                np.array(all_probabilities), 
                np.array(all_labels))
    
    def _evaluate_individual_models(self, model: DRMultiArchitectureEnsemble, 
                                  test_loader) -> Dict[str, Dict[str, float]]:
        """Evaluate performance of individual models within the ensemble."""
        print("📊 Evaluating individual model performance...")
        
        individual_results = {}
        
        # Get individual model predictions
        for model_name, probs in self.individual_probabilities.items():
            preds = np.argmax(probs, axis=1)
            true_labels = np.array([label for _, labels in test_loader for label in labels.numpy()])
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, preds, average='weighted', zero_division=0
            )
            
            # Per-class metrics
            per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
                true_labels, preds, average=None, zero_division=0
            )
            
            # AUC calculation
            try:
                # Binarize labels for multiclass AUC
                y_bin = label_binarize(true_labels, classes=list(range(self.config.model.num_classes)))
                if y_bin.shape[1] == 1:  # Binary case
                    auc_score = roc_auc_score(true_labels, probs[:, 1])
                else:  # Multiclass case
                    auc_score = roc_auc_score(y_bin, probs, multi_class='ovr', average='weighted')
            except:
                auc_score = 0.0
            
            individual_results[model_name] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc': float(auc_score),
                'per_class_precision': per_class_precision.tolist(),
                'per_class_recall': per_class_recall.tolist(),
                'per_class_f1': per_class_f1.tolist()
            }
            
            print(f"   {model_name}: Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={auc_score:.4f}")
        
        return individual_results
    
    def _calculate_ensemble_metrics(self, predictions: np.ndarray, 
                                  probabilities: np.ndarray, 
                                  true_labels: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive metrics for the ensemble model."""
        print("📈 Calculating ensemble performance metrics...")
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
            true_labels, predictions, average=None, zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Calculate sensitivity (recall) and specificity for each class
        sensitivity = per_class_recall
        specificity = []
        for i in range(len(self.class_names)):
            tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
            fp = np.sum(cm[:, i]) - cm[i, i]
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificity.append(spec)
        
        # AUC calculation
        try:
            y_bin = label_binarize(true_labels, classes=list(range(self.config.model.num_classes)))
            if y_bin.shape[1] == 1:  # Binary case
                auc_score = roc_auc_score(true_labels, probabilities[:, 1])
                per_class_auc = [auc_score]
            else:  # Multiclass case
                auc_score = roc_auc_score(y_bin, probabilities, multi_class='ovr', average='weighted')
                per_class_auc = []
                for i in range(self.config.model.num_classes):
                    try:
                        class_auc = roc_auc_score(y_bin[:, i], probabilities[:, i])
                        per_class_auc.append(class_auc)
                    except:
                        per_class_auc.append(0.0)
        except:
            auc_score = 0.0
            per_class_auc = [0.0] * self.config.model.num_classes
        
        # Classification report
        report = classification_report(true_labels, predictions, 
                                     target_names=self.class_names, 
                                     output_dict=True, zero_division=0)
        
        return {
            'overall_metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc': float(auc_score)
            },
            'per_class_metrics': {
                'precision': per_class_precision.tolist(),
                'recall': per_class_recall.tolist(),
                'f1_score': per_class_f1.tolist(),
                'sensitivity': sensitivity.tolist(),
                'specificity': specificity,
                'auc': per_class_auc,
                'support': support.tolist()
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
    
    def _medical_grade_assessment(self, ensemble_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess model performance against medical-grade standards."""
        print("🏥 Performing medical-grade assessment...")
        
        overall = ensemble_results['overall_metrics']
        per_class = ensemble_results['per_class_metrics']
        
        # Check overall thresholds
        meets_accuracy = overall['accuracy'] >= self.medical_thresholds['accuracy']
        meets_target = overall['accuracy'] >= self.medical_thresholds['target_accuracy']
        
        # Check per-class thresholds
        min_sensitivity = min(per_class['sensitivity'])
        min_specificity = min(per_class['specificity'])
        min_auc = min(per_class['auc'])
        
        meets_sensitivity = min_sensitivity >= self.medical_thresholds['sensitivity']
        meets_specificity = min_specificity >= self.medical_thresholds['specificity']
        meets_auc = min_auc >= self.medical_thresholds['auc']
        
        # Overall medical grade
        all_criteria_met = all([
            meets_accuracy, meets_sensitivity, meets_specificity, meets_auc
        ])
        
        # Clinical readiness assessment
        if all_criteria_met:
            if meets_target:
                clinical_grade = "EXCELLENT - Exceeds Research Target"
                readiness = "READY_FOR_CLINICAL_TRIALS"
            else:
                clinical_grade = "GOOD - Meets Medical Standards"
                readiness = "READY_FOR_VALIDATION"
        else:
            if meets_accuracy:
                clinical_grade = "ACCEPTABLE - Needs Improvement"
                readiness = "NEEDS_REFINEMENT"
            else:
                clinical_grade = "INSUFFICIENT - Below Standards"
                readiness = "NOT_READY"
        
        # Generate recommendations
        recommendations = []
        if not meets_accuracy:
            recommendations.append("Improve overall accuracy through additional training or model refinement")
        if not meets_sensitivity:
            recommendations.append(f"Improve sensitivity (current min: {min_sensitivity:.3f}, required: {self.medical_thresholds['sensitivity']:.3f})")
        if not meets_specificity:
            recommendations.append(f"Improve specificity (current min: {min_specificity:.3f}, required: {self.medical_thresholds['specificity']:.3f})")
        if not meets_auc:
            recommendations.append(f"Improve AUC scores (current min: {min_auc:.3f}, required: {self.medical_thresholds['auc']:.3f})")
        
        if all_criteria_met:
            recommendations.append("Model meets all medical-grade criteria - ready for clinical validation")
        
        return {
            'medical_grade_compliance': {
                'meets_accuracy_threshold': meets_accuracy,
                'meets_sensitivity_threshold': meets_sensitivity,
                'meets_specificity_threshold': meets_specificity,
                'meets_auc_threshold': meets_auc,
                'meets_target_accuracy': meets_target,
                'all_criteria_met': all_criteria_met
            },
            'performance_summary': {
                'overall_accuracy': overall['accuracy'],
                'min_sensitivity': min_sensitivity,
                'min_specificity': min_specificity,
                'min_auc': min_auc,
                'target_accuracy': self.medical_thresholds['target_accuracy']
            },
            'clinical_assessment': {
                'clinical_grade': clinical_grade,
                'readiness_status': readiness,
                'recommendations': recommendations
            }
        }
    
    def _clinical_performance_analysis(self, predictions: np.ndarray, 
                                     probabilities: np.ndarray, 
                                     true_labels: np.ndarray) -> Dict[str, Any]:
        """Perform clinical-focused performance analysis."""
        print("🩺 Performing clinical analysis...")
        
        # Referral accuracy (classes 2, 3, 4 require referral)
        referable_true = (true_labels >= 2).astype(int)
        referable_pred = (predictions >= 2).astype(int)
        referral_accuracy = accuracy_score(referable_true, referable_pred)
        
        # Sight-threatening accuracy (classes 3, 4)
        sight_threatening_true = (true_labels >= 3).astype(int)
        sight_threatening_pred = (predictions >= 3).astype(int)
        sight_threatening_accuracy = accuracy_score(sight_threatening_true, sight_threatening_pred)
        
        # Confidence analysis
        max_probs = np.max(probabilities, axis=1)
        confidence_distribution = {
            'high_confidence_0.9+': np.mean(max_probs >= 0.9),
            'medium_confidence_0.7-0.9': np.mean((max_probs >= 0.7) & (max_probs < 0.9)),
            'low_confidence_<0.7': np.mean(max_probs < 0.7)
        }
        
        # Error analysis by confidence
        correct_predictions = (predictions == true_labels)
        error_by_confidence = {
            'high_confidence_errors': np.mean(~correct_predictions[max_probs >= 0.9]) if np.any(max_probs >= 0.9) else 0.0,
            'medium_confidence_errors': np.mean(~correct_predictions[(max_probs >= 0.7) & (max_probs < 0.9)]) if np.any((max_probs >= 0.7) & (max_probs < 0.9)) else 0.0,
            'low_confidence_errors': np.mean(~correct_predictions[max_probs < 0.7]) if np.any(max_probs < 0.7) else 0.0
        }
        
        return {
            'clinical_accuracy': {
                'referral_accuracy': float(referral_accuracy),
                'sight_threatening_accuracy': float(sight_threatening_accuracy)
            },
            'confidence_analysis': {
                'distribution': confidence_distribution,
                'error_rates': error_by_confidence,
                'mean_confidence': float(np.mean(max_probs)),
                'confidence_std': float(np.std(max_probs))
            }
        }
    
    def _generate_evaluation_plots(self, predictions: np.ndarray, 
                                 probabilities: np.ndarray, 
                                 true_labels: np.ndarray, 
                                 save_dir: str):
        """Generate comprehensive evaluation visualizations."""
        print("📊 Generating evaluation plots...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(true_labels, predictions)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Ensemble Model - Normalized Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC Curves
        plt.figure(figsize=(12, 8))
        
        # Binarize labels for ROC
        y_bin = label_binarize(true_labels, classes=list(range(self.config.model.num_classes)))
        
        if y_bin.shape[1] > 1:  # Multiclass
            for i in range(self.config.model.num_classes):
                try:
                    fpr, tpr, _ = roc_curve(y_bin[:, i], probabilities[:, i])
                    auc_score = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f'{self.class_names[i]} (AUC = {auc_score:.3f})')
                except:
                    continue
        else:  # Binary
            fpr, tpr, _ = roc_curve(true_labels, probabilities[:, 1])
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Ensemble Model - ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/roc_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Class Performance Comparison
        ensemble_results = self._calculate_ensemble_metrics(predictions, probabilities, true_labels)
        per_class = ensemble_results['per_class_metrics']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Precision
        axes[0, 0].bar(self.class_names, per_class['precision'])
        axes[0, 0].set_title('Precision by Class')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Recall (Sensitivity)
        axes[0, 1].bar(self.class_names, per_class['recall'])
        axes[0, 1].set_title('Recall (Sensitivity) by Class')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Specificity
        axes[1, 0].bar(self.class_names, per_class['specificity'])
        axes[1, 0].set_title('Specificity by Class')
        axes[1, 0].set_ylabel('Specificity')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # AUC
        axes[1, 1].bar(self.class_names, per_class['auc'])
        axes[1, 1].set_title('AUC by Class')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/class_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Individual Model Comparison
        if hasattr(self, 'individual_probabilities'):
            individual_results = self._evaluate_individual_models(None, None)  # Using stored results
            
            models = list(individual_results.keys())
            accuracies = [individual_results[model]['accuracy'] for model in models]
            f1_scores = [individual_results[model]['f1_score'] for model in models]
            auc_scores = [individual_results[model]['auc'] for model in models]
            
            x = np.arange(len(models))
            width = 0.25
            
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.bar(x - width, accuracies, width, label='Accuracy')
            ax.bar(x, f1_scores, width, label='F1 Score')
            ax.bar(x + width, auc_scores, width, label='AUC')
            
            # Add ensemble performance
            ensemble_acc = ensemble_results['overall_metrics']['accuracy']
            ensemble_f1 = ensemble_results['overall_metrics']['f1_score']
            ensemble_auc = ensemble_results['overall_metrics']['auc']
            
            ax.axhline(y=ensemble_acc, color='red', linestyle='--', alpha=0.7, label='Ensemble Accuracy')
            ax.axhline(y=ensemble_f1, color='green', linestyle='--', alpha=0.7, label='Ensemble F1')
            ax.axhline(y=ensemble_auc, color='blue', linestyle='--', alpha=0.7, label='Ensemble AUC')
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Performance')
            ax.set_title('Individual Models vs Ensemble Performance')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45)
            ax.legend()
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/individual_model_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📊 All plots saved to {save_dir}/")
    
    def _save_evaluation_results(self, results: Dict[str, Any], save_dir: str):
        """Save comprehensive evaluation results to files."""
        print("💾 Saving evaluation results...")
        
        # Save complete results as JSON
        results_file = f"{save_dir}/comprehensive_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary report
        summary_file = f"{save_dir}/evaluation_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("ENSEMBLE MODEL EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall performance
            overall = results['ensemble_performance']['overall_metrics']
            f.write(f"OVERALL PERFORMANCE:\n")
            f.write(f"Accuracy: {overall['accuracy']:.4f} ({overall['accuracy']*100:.2f}%)\n")
            f.write(f"Precision: {overall['precision']:.4f}\n")
            f.write(f"Recall: {overall['recall']:.4f}\n")
            f.write(f"F1 Score: {overall['f1_score']:.4f}\n")
            f.write(f"AUC: {overall['auc']:.4f}\n\n")
            
            # Medical assessment
            medical = results['medical_assessment']
            f.write(f"MEDICAL-GRADE ASSESSMENT:\n")
            f.write(f"Clinical Grade: {medical['clinical_assessment']['clinical_grade']}\n")
            f.write(f"Readiness Status: {medical['clinical_assessment']['readiness_status']}\n")
            f.write(f"Meets Medical Standards: {medical['medical_grade_compliance']['all_criteria_met']}\n\n")
            
            # Individual model comparison
            f.write(f"INDIVIDUAL MODEL PERFORMANCE:\n")
            for model_name, metrics in results['individual_models'].items():
                f.write(f"{model_name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}\n")
            
            f.write(f"\nRecommendations:\n")
            for rec in medical['clinical_assessment']['recommendations']:
                f.write(f"- {rec}\n")
        
        print(f"📁 Results saved to {save_dir}/")
        print(f"   - Complete results: {results_file}")
        print(f"   - Summary report: {summary_file}")

def evaluate_ensemble_checkpoint(checkpoint_path: str, 
                               dataset_path: str = "./dataset3_augmented_resized",
                               config_path: Optional[str] = None,
                               output_dir: str = "./evaluation_results",
                               batch_size: int = 16,
                               device: str = "cuda") -> Dict[str, Any]:
    """
    Convenience function to evaluate an ensemble checkpoint.
    
    Args:
        checkpoint_path: Path to the trained ensemble checkpoint
        dataset_path: Path to the dataset
        config_path: Optional path to config file (uses default if None)
        output_dir: Directory to save evaluation results
        batch_size: Batch size for evaluation
        device: Device to use for evaluation
        
    Returns:
        Complete evaluation results
    """
    print("🚀 Starting ensemble model evaluation...")
    
    # Load or create configuration
    if config_path and os.path.exists(config_path):
        config = EnsembleConfig.load(config_path)
        print(f"✅ Loaded configuration from {config_path}")
    else:
        config = create_medical_grade_config()
        config.data.dataset_path = dataset_path
        config.data.batch_size = batch_size
        print("✅ Using default medical-grade configuration")
    
    # Create evaluator
    evaluator = MedicalGradeEvaluator(config, device)
    
    # Load model
    model = evaluator.load_ensemble_model(checkpoint_path)
    
    # Create test data loader
    _, _, test_loader = create_ensemble_data_loaders(config)
    print(f"✅ Created test loader with {len(test_loader)} batches")
    
    # Perform evaluation
    results = evaluator.evaluate_model(model, test_loader, output_dir)
    
    # Print summary
    overall = results['ensemble_performance']['overall_metrics']
    medical = results['medical_assessment']
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"📊 Overall Accuracy: {overall['accuracy']:.4f} ({overall['accuracy']*100:.2f}%)")
    print(f"📊 F1 Score: {overall['f1_score']:.4f}")
    print(f"📊 AUC Score: {overall['auc']:.4f}")
    print(f"🏥 Clinical Grade: {medical['clinical_assessment']['clinical_grade']}")
    print(f"🏥 Readiness: {medical['clinical_assessment']['readiness_status']}")
    print(f"✅ Medical Standards Met: {medical['medical_grade_compliance']['all_criteria_met']}")
    print("="*60)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate ensemble diabetic retinopathy classification model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to the trained ensemble checkpoint")
    parser.add_argument("--dataset", type=str, default="./dataset3_augmented_resized",
                       help="Path to the dataset")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to ensemble configuration file")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for evaluation")
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_ensemble_checkpoint(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        config_path=args.config,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device
    )
    
    print(f"\n🎉 Evaluation completed! Results saved to {args.output_dir}")