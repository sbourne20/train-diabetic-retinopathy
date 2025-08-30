import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, cohen_kappa_score
)
from sklearn.calibration import calibration_curve
from typing import Dict, List, Tuple, Optional
import json

# Suppress sklearn warnings to avoid STDERR confusion
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
from tqdm import tqdm

from models import DiabeticRetinopathyModel
from utils import calculate_metrics, plot_confusion_matrix

class ModelEvaluator:
    """Comprehensive evaluation for diabetic retinopathy model."""
    
    def __init__(self, 
                 model: DiabeticRetinopathyModel,
                 test_loader,
                 config,
                 device: str = 'cuda'):
        
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Phase 1: 5-class DR classification (ICDR standard)
        self.dr_class_names = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'PDR']
        
        # Detect dataset type - Phase 1 is DR-only (5-class ICDR format)
        # Check multiple indicators for Phase 1 DR-only dataset
        self.is_dr_only = (
            # Check if we have a specific dataset path (dataset3_augmented_resized is DR-only)
            (hasattr(config.data, 'dataset_path') and 'dataset3' in str(config.data.dataset_path)) or
            # Check if dataset_type is specified as 1 (Type 1 = 5-class DR)
            (hasattr(config.data, 'dataset_type') and config.data.dataset_type == 1) or
            # Check config directly for dr_only flag
            (hasattr(config, 'dr_only') and config.dr_only) or
            # Default to Phase 1 for this project (can be overridden)
            True  # Default assumption for this DR project
        )
        
        # Debug print for troubleshooting
        print(f"🔍 ModelEvaluator Debug Info:")
        print(f"   - Dataset type: {getattr(config.data, 'dataset_type', 'NOT_SET')}")
        print(f"   - Dataset path: {getattr(config.data, 'dataset_path', 'NOT_SET')}")
        print(f"   - DR-only mode: {self.is_dr_only}")
        print(f"   - Config attributes: {[attr for attr in dir(config) if not attr.startswith('_')]}")
        print(f"   - Config.data attributes: {[attr for attr in dir(config.data) if not attr.startswith('_')]}")
        
        # Legacy class names for backward compatibility (will be phased out)
        if not self.is_dr_only:
            self.rg_class_names = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR'] 
            self.me_class_names = ['No Risk', 'Low Risk', 'High Risk']
        else:
            # For DR-only mode, define minimal ME classes for compatibility
            self.rg_class_names = self.dr_class_names  # Use DR classes as RG classes
            self.me_class_names = ['No Risk']  # Single dummy class for DR-only mode
        
    def evaluate_model(self, save_results: bool = True) -> Dict:
        """Comprehensive model evaluation."""
        
        print("Starting comprehensive model evaluation...")
        
        # Get predictions and ground truth
        results = self._get_predictions()
        
        # Phase 1: DR-only evaluation (5-class ICDR classification)
        if self.is_dr_only:
            print("📋 Phase 1 Mode: Evaluating DR classification only (5-class ICDR)")
            
            # Calculate metrics for DR classification only
            dr_metrics = self._calculate_classification_metrics(
                results['rg_targets'],  # Using rg_targets as DR targets for compatibility
                results['rg_predictions'], 
                results['rg_probabilities'],
                'DR Classification'
            )
            
            # Generate DR-only visualizations
            self._create_dr_only_visualizations(results)
            
            # Create Phase 1 compliant output
            evaluation_results = {
                'dr_metrics': dr_metrics,
                'phase': 1,
                'grading_system': 'ICDR_5class',
                'class_names': self.dr_class_names,
                'overall_summary': self._create_dr_summary(dr_metrics),
                'predictions': results
            }
            
        else:
            # Legacy: Multi-task evaluation (RG + ME)
            print("📋 Multi-task Mode: Evaluating both RG and ME classification")
            
            # Calculate metrics for both tasks
            rg_metrics = self._calculate_classification_metrics(
                results['rg_targets'], 
                results['rg_predictions'],
                results['rg_probabilities'],
                'RG'
            )
            
            me_metrics = self._calculate_classification_metrics(
                results['me_targets'], 
                results['me_predictions'],
                results['me_probabilities'],
                'ME'
            )
            
            # Generate visualizations
            self._create_visualizations(results)
            
            # Evaluate medical reasoning quality
            reasoning_quality = self._evaluate_reasoning_quality(results)
            
            # Combine all results
            evaluation_results = {
                'rg_metrics': rg_metrics,
                'me_metrics': me_metrics,
                'reasoning_quality': reasoning_quality,
                'overall_summary': self._create_summary(rg_metrics, me_metrics),
                'predictions': results
            }
        
        # Save results
        if save_results:
            self._save_evaluation_results(evaluation_results)
        
        return evaluation_results
    
    def _get_predictions(self) -> Dict:
        """Get model predictions on test set."""
        
        self.model.eval()
        
        results = {
            'rg_predictions': [],
            'me_predictions': [],
            'rg_targets': [],
            'me_targets': [],
            'rg_probabilities': [],
            'me_probabilities': [],
            'image_paths': [],
            'generated_reports': []
        }
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Generating predictions", file=sys.stdout)
            
            for batch in pbar:
                images = batch['image'].to(self.device)
                # Handle both dataset formats
                if 'dr_grade' in batch:
                    # Dataset type 1: 5-class DR format
                    dr_labels = batch['dr_grade']
                    rg_labels = dr_labels  # Use DR grade as primary label
                    me_labels = torch.zeros_like(dr_labels)  # Placeholder for compatibility
                else:
                    # Dataset type 0: RG/ME format
                    rg_labels = batch['rg_grade']
                    me_labels = batch['me_grade']
                image_paths = batch['image_path']
                
                # Forward pass
                outputs = self.model(images)
                
                # Get probabilities and predictions
                # Fix: Use 'dr_logits' instead of 'rg_logits' to match model output
                rg_probs = F.softmax(outputs['dr_logits'], dim=1)
                
                # Handle missing ME logits for 5-class DR dataset
                if 'me_logits' in outputs:
                    me_probs = F.softmax(outputs['me_logits'], dim=1)
                else:
                    # For 5-class DR dataset, create dummy ME probabilities
                    batch_size = rg_probs.shape[0]
                    me_probs = torch.zeros((batch_size, len(self.me_class_names))).to(rg_probs.device)
                    me_probs[:, 0] = 1.0  # Set to 'No Risk' class
                
                rg_preds = torch.argmax(rg_probs, dim=1)
                me_preds = torch.argmax(me_probs, dim=1)
                
                # Generate medical reports (skip if method doesn't exist)
                try:
                    reports = self.model.generate_medical_report(
                        images, rg_preds, me_preds
                    )
                except AttributeError:
                    # Model doesn't have medical report generation yet
                    reports = ['Generated report not available'] * len(rg_preds)
                
                # Store results
                results['rg_predictions'].extend(rg_preds.cpu().numpy())
                results['me_predictions'].extend(me_preds.cpu().numpy())
                results['rg_targets'].extend(rg_labels.numpy())
                results['me_targets'].extend(me_labels.numpy())
                results['rg_probabilities'].extend(rg_probs.cpu().numpy())
                results['me_probabilities'].extend(me_probs.cpu().numpy())
                results['image_paths'].extend(image_paths)
                results['generated_reports'].extend(reports)
        
        # Convert to numpy arrays
        for key in ['rg_predictions', 'me_predictions', 'rg_targets', 'me_targets']:
            results[key] = np.array(results[key])
        
        results['rg_probabilities'] = np.array(results['rg_probabilities'])
        results['me_probabilities'] = np.array(results['me_probabilities'])
        
        return results
    
    def _calculate_classification_metrics(self, 
                                        y_true: np.ndarray,
                                        y_pred: np.ndarray,
                                        y_prob: np.ndarray,
                                        task_name: str) -> Dict:
        """Calculate comprehensive classification metrics."""
        
        metrics = {}
        
        # Basic metrics
        basic_metrics = calculate_metrics(y_true, y_pred, y_prob)
        metrics.update(basic_metrics)
        
        # Additional metrics
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Per-class metrics
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics['per_class_metrics'] = class_report
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # ROC AUC for each class (one-vs-rest)
        try:
            if len(np.unique(y_true)) > 2:
                metrics['per_class_auc'] = {}
                for i in range(y_prob.shape[1]):
                    if i in np.unique(y_true):
                        y_true_binary = (y_true == i).astype(int)
                        auc_score = roc_auc_score(y_true_binary, y_prob[:, i])
                        metrics['per_class_auc'][f'class_{i}'] = auc_score
        except Exception as e:
            print(f"Could not calculate per-class AUC for {task_name}: {e}")
        
        print(f"\n{task_name} Classification Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro_avg']['f1-score']:.4f}")
        print(f"Weighted F1: {metrics['weighted_avg']['f1-score']:.4f}")
        if 'auc_macro' in metrics:
            print(f"Macro AUC: {metrics['auc_macro']:.4f}")
        print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
        
        return metrics
    
    def _create_visualizations(self, results: Dict):
        """Create evaluation visualizations with robust error handling."""
        
        os.makedirs('visualizations', exist_ok=True)
        
        # Confusion matrices
        try:
            self._plot_confusion_matrices(results)
            print("✅ Confusion matrices created")
        except Exception as e:
            print(f"⚠️ Warning: Could not create confusion matrices: {e}")
        
        # ROC curves
        try:
            self._plot_roc_curves(results)
            print("✅ ROC curves created")
        except Exception as e:
            print(f"⚠️ Warning: Could not create ROC curves: {e}")
        
        # Calibration curves
        try:
            self._plot_calibration_curves(results)
            print("✅ Calibration curves created")
        except Exception as e:
            print(f"⚠️ Warning: Could not create calibration curves: {e}")
        
        # Class distribution comparison
        try:
            self._plot_class_distributions(results)
            print("✅ Class distribution plots created")
        except Exception as e:
            print(f"⚠️ Warning: Could not create class distribution plots: {e}")
        
        # Error analysis
        try:
            self._create_error_analysis(results)
            print("✅ Error analysis created")
        except Exception as e:
            print(f"⚠️ Warning: Could not create error analysis: {e}")
    
    def _plot_confusion_matrices(self, results: Dict):
        """Plot confusion matrices for both tasks."""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # RG confusion matrix
        cm_rg = confusion_matrix(results['rg_targets'], results['rg_predictions'])
        sns.heatmap(cm_rg, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.rg_class_names,
                   yticklabels=self.rg_class_names,
                   ax=axes[0])
        axes[0].set_title('Retinopathy Grade Confusion Matrix')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        # ME confusion matrix
        cm_me = confusion_matrix(results['me_targets'], results['me_predictions'])
        sns.heatmap(cm_me, annot=True, fmt='d', cmap='Oranges',
                   xticklabels=self.me_class_names,
                   yticklabels=self.me_class_names,
                   ax=axes[1])
        axes[1].set_title('Macular Edema Risk Confusion Matrix')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('visualizations/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self, results: Dict):
        """Plot ROC curves for both tasks."""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # RG ROC curves
        n_classes_rg = len(self.rg_class_names)
        rg_curves_plotted = 0
        for i in range(n_classes_rg):
            if i in np.unique(results['rg_targets']):
                y_true_binary = (results['rg_targets'] == i).astype(int)
                y_score = results['rg_probabilities'][:, i]
                
                # Check if we have both classes (0 and 1) in y_true_binary
                unique_classes = np.unique(y_true_binary)
                if len(unique_classes) > 1:
                    try:
                        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                        auc = roc_auc_score(y_true_binary, y_score)
                        axes[0].plot(fpr, tpr, label=f'{self.rg_class_names[i]} (AUC = {auc:.3f})')
                        rg_curves_plotted += 1
                    except ValueError as e:
                        print(f"Skipping ROC curve for RG class {i}: {e}")
                else:
                    print(f"Skipping ROC curve for RG class {i}: Only one class present")
        
        # If no curves were plotted, add a placeholder
        if rg_curves_plotted == 0:
            axes[0].plot([0, 1], [0, 1], 'b--', label='No valid curves (single class predictions)')
        
        axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('RG Classification ROC Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # ME ROC curves
        n_classes_me = len(self.me_class_names)
        me_curves_plotted = 0
        for i in range(n_classes_me):
            if i in np.unique(results['me_targets']):
                y_true_binary = (results['me_targets'] == i).astype(int)
                y_score = results['me_probabilities'][:, i]
                
                # Check if we have both classes (0 and 1) in y_true_binary
                unique_classes = np.unique(y_true_binary)
                if len(unique_classes) > 1:
                    try:
                        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                        auc = roc_auc_score(y_true_binary, y_score)
                        axes[1].plot(fpr, tpr, label=f'{self.me_class_names[i]} (AUC = {auc:.3f})')
                        me_curves_plotted += 1
                    except ValueError as e:
                        print(f"Skipping ROC curve for ME class {i}: {e}")
                else:
                    print(f"Skipping ROC curve for ME class {i}: Only one class present")
        
        # If no curves were plotted, add a placeholder
        if me_curves_plotted == 0:
            axes[1].plot([0, 1], [0, 1], 'r--', label='No valid curves (single class predictions)')
        
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ME Risk Classification ROC Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_calibration_curves(self, results: Dict):
        """Plot calibration curves to assess prediction confidence."""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # RG calibration
        for i in range(len(self.rg_class_names)):
            if i in np.unique(results['rg_targets']):
                y_true_binary = (results['rg_targets'] == i).astype(int)
                y_prob = results['rg_probabilities'][:, i]
                
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true_binary, y_prob, n_bins=10
                )
                
                axes[0].plot(mean_predicted_value, fraction_of_positives, 
                           marker='o', label=self.rg_class_names[i])
        
        axes[0].plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        axes[0].set_xlabel('Mean Predicted Probability')
        axes[0].set_ylabel('Fraction of Positives')
        axes[0].set_title('RG Calibration Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # ME calibration
        for i in range(len(self.me_class_names)):
            if i in np.unique(results['me_targets']):
                y_true_binary = (results['me_targets'] == i).astype(int)
                y_prob = results['me_probabilities'][:, i]
                
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true_binary, y_prob, n_bins=10
                )
                
                axes[1].plot(mean_predicted_value, fraction_of_positives, 
                           marker='o', label=self.me_class_names[i])
        
        axes[1].plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        axes[1].set_xlabel('Mean Predicted Probability')
        axes[1].set_ylabel('Fraction of Positives')
        axes[1].set_title('ME Calibration Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/calibration_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_class_distributions(self, results: Dict):
        """Plot predicted vs true class distributions."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # RG true vs predicted
        rg_true_counts = pd.Series(results['rg_targets']).value_counts().sort_index()
        rg_pred_counts = pd.Series(results['rg_predictions']).value_counts().sort_index()
        
        x = np.arange(len(self.rg_class_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, [rg_true_counts.get(i, 0) for i in range(len(self.rg_class_names))], 
                      width, label='True', alpha=0.8)
        axes[0, 0].bar(x + width/2, [rg_pred_counts.get(i, 0) for i in range(len(self.rg_class_names))], 
                      width, label='Predicted', alpha=0.8)
        axes[0, 0].set_xlabel('RG Classes')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('RG Class Distribution: True vs Predicted')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(self.rg_class_names, rotation=45)
        axes[0, 0].legend()
        
        # ME true vs predicted
        me_true_counts = pd.Series(results['me_targets']).value_counts().sort_index()
        me_pred_counts = pd.Series(results['me_predictions']).value_counts().sort_index()
        
        x = np.arange(len(self.me_class_names))
        
        axes[0, 1].bar(x - width/2, [me_true_counts.get(i, 0) for i in range(len(self.me_class_names))], 
                      width, label='True', alpha=0.8)
        axes[0, 1].bar(x + width/2, [me_pred_counts.get(i, 0) for i in range(len(self.me_class_names))], 
                      width, label='Predicted', alpha=0.8)
        axes[0, 1].set_xlabel('ME Classes')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('ME Class Distribution: True vs Predicted')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(self.me_class_names, rotation=45)
        axes[0, 1].legend()
        
        # Prediction confidence distributions
        rg_max_probs = np.max(results['rg_probabilities'], axis=1)
        me_max_probs = np.max(results['me_probabilities'], axis=1)
        
        axes[1, 0].hist(rg_max_probs, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Max Probability')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('RG Prediction Confidence Distribution')
        axes[1, 0].axvline(np.mean(rg_max_probs), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(rg_max_probs):.3f}')
        axes[1, 0].legend()
        
        axes[1, 1].hist(me_max_probs, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Max Probability')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('ME Prediction Confidence Distribution')
        axes[1, 1].axvline(np.mean(me_max_probs), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(me_max_probs):.3f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('visualizations/class_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_error_analysis(self, results: Dict):
        """Create detailed error analysis."""
        
        # Find misclassified samples
        rg_errors = results['rg_targets'] != results['rg_predictions']
        me_errors = results['me_targets'] != results['me_predictions']
        
        error_analysis = {
            'rg_errors': {
                'count': int(np.sum(rg_errors)),
                'percentage': float(np.mean(rg_errors) * 100),
                'error_pairs': []
            },
            'me_errors': {
                'count': int(np.sum(me_errors)),
                'percentage': float(np.mean(me_errors) * 100),
                'error_pairs': []
            }
        }
        
        # Analyze error patterns
        for i in range(len(results['rg_targets'])):
            if rg_errors[i]:
                error_analysis['rg_errors']['error_pairs'].append({
                    'image_path': results['image_paths'][i],
                    'true_label': int(results['rg_targets'][i]),
                    'predicted_label': int(results['rg_predictions'][i]),
                    'confidence': float(np.max(results['rg_probabilities'][i]))
                })
        
        for i in range(len(results['me_targets'])):
            if me_errors[i]:
                error_analysis['me_errors']['error_pairs'].append({
                    'image_path': results['image_paths'][i],
                    'true_label': int(results['me_targets'][i]),
                    'predicted_label': int(results['me_predictions'][i]),
                    'confidence': float(np.max(results['me_probabilities'][i]))
                })
        
        # Save error analysis
        with open('visualizations/error_analysis.json', 'w') as f:
            json.dump(error_analysis, f, indent=2)
        
        print(f"RG Classification Errors: {error_analysis['rg_errors']['count']} ({error_analysis['rg_errors']['percentage']:.2f}%)")
        print(f"ME Classification Errors: {error_analysis['me_errors']['count']} ({error_analysis['me_errors']['percentage']:.2f}%)")
    
    def _evaluate_reasoning_quality(self, results: Dict) -> Dict:
        """Evaluate quality of generated medical reasoning."""
        
        reasoning_metrics = {
            'total_reports': len(results['generated_reports']),
            'avg_report_length': 0,
            'medical_terms_coverage': 0,
            'consistency_score': 0
        }
        
        if results['generated_reports']:
            # Calculate average report length
            report_lengths = [len(report.split()) for report in results['generated_reports']]
            reasoning_metrics['avg_report_length'] = np.mean(report_lengths)
            
            # Simple medical terms coverage (basic implementation)
            medical_terms = [
                'retinopathy', 'macular', 'edema', 'hemorrhage', 'exudate',
                'microaneurysm', 'neovascularization', 'diabetic', 'fundus'
            ]
            
            term_counts = []
            for report in results['generated_reports']:
                report_lower = report.lower()
                count = sum(1 for term in medical_terms if term in report_lower)
                term_counts.append(count)
            
            reasoning_metrics['medical_terms_coverage'] = np.mean(term_counts)
            
            print(f"Medical Reasoning Quality:")
            print(f"Average report length: {reasoning_metrics['avg_report_length']:.1f} words")
            print(f"Medical terms coverage: {reasoning_metrics['medical_terms_coverage']:.2f} terms/report")
        
        return reasoning_metrics
    
    def _create_summary(self, rg_metrics: Dict, me_metrics: Dict) -> Dict:
        """Create overall evaluation summary."""
        
        summary = {
            'overall_accuracy': {
                'rg': rg_metrics['accuracy'],
                'me': me_metrics['accuracy'],
                'combined': (rg_metrics['accuracy'] + me_metrics['accuracy']) / 2
            },
            'overall_auc': {
                'rg': rg_metrics.get('auc_macro', 0.0),
                'me': me_metrics.get('auc_macro', 0.0),
                'combined': (rg_metrics.get('auc_macro', 0.0) + me_metrics.get('auc_macro', 0.0)) / 2
            },
            'clinical_relevance': self._assess_clinical_relevance(rg_metrics, me_metrics)
        }
        
        return summary
    
    def _assess_clinical_relevance(self, rg_metrics: Dict, me_metrics: Dict) -> Dict:
        """Assess clinical relevance of the model performance."""
        
        # High-priority classes (severe conditions)
        rg_severe_sensitivity = rg_metrics['per_class_metrics'].get('3', {}).get('recall', 0.0)
        me_high_risk_sensitivity = me_metrics['per_class_metrics'].get('2', {}).get('recall', 0.0)
        
        clinical_assessment = {
            'severe_dr_detection': rg_severe_sensitivity,
            'high_me_risk_detection': me_high_risk_sensitivity,
            'overall_clinical_utility': 'Good' if (rg_severe_sensitivity > 0.8 and me_high_risk_sensitivity > 0.8) else 'Moderate'
        }
        
        return clinical_assessment
    
    def _save_evaluation_results(self, results: Dict):
        """Save comprehensive evaluation results."""
        
        os.makedirs('outputs', exist_ok=True)
        
        # Save metrics as JSON
        metrics_to_save = {
            'rg_metrics': results['rg_metrics'],
            'me_metrics': results['me_metrics'],
            'reasoning_quality': results['reasoning_quality'],
            'overall_summary': results['overall_summary']
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        metrics_to_save = convert_numpy(metrics_to_save)
        
        with open('outputs/evaluation_results.json', 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        # Save detailed predictions
        predictions_df = pd.DataFrame({
            'image_path': results['predictions']['image_paths'],
            'rg_true': results['predictions']['rg_targets'],
            'rg_pred': results['predictions']['rg_predictions'],
            'me_true': results['predictions']['me_targets'],
            'me_pred': results['predictions']['me_predictions'],
            'rg_confidence': [np.max(probs) for probs in results['predictions']['rg_probabilities']],
            'me_confidence': [np.max(probs) for probs in results['predictions']['me_probabilities']],
            'generated_report': results['predictions']['generated_reports']
        })
        
        predictions_df.to_csv('outputs/detailed_predictions.csv', index=False)
        
        print("Evaluation results saved to 'outputs/' directory")
    
    def _create_dr_only_visualizations(self, results: Dict):
        """Create DR-only visualizations for Phase 1 evaluation."""
        
        os.makedirs('visualizations', exist_ok=True)
        
        # Confusion matrix for DR classification
        try:
            self._plot_dr_confusion_matrix(results)
            print("✅ DR confusion matrix created")
        except Exception as e:
            print(f"⚠️ Warning: Could not create DR confusion matrix: {e}")
        
        # ROC curves for DR classification
        try:
            self._plot_dr_roc_curves(results)
            print("✅ DR ROC curves created")
        except Exception as e:
            print(f"⚠️ Warning: Could not create DR ROC curves: {e}")
        
        # DR class distribution comparison
        try:
            self._plot_dr_class_distributions(results)
            print("✅ DR class distribution plots created")
        except Exception as e:
            print(f"⚠️ Warning: Could not create DR class distribution plots: {e}")
    
    def _plot_dr_confusion_matrix(self, results: Dict):
        """Plot confusion matrix for DR classification."""
        
        plt.figure(figsize=(10, 8))
        
        cm = confusion_matrix(results['rg_targets'], results['rg_predictions'])
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.dr_class_names,
                   yticklabels=self.dr_class_names,
                   cbar_kws={'shrink': 0.8})
        
        plt.title('DR Classification Confusion Matrix\n(5-Class ICDR Standard)', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Severity')
        plt.ylabel('True Severity')
        plt.tight_layout()
        
        plt.savefig('visualizations/dr_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_dr_roc_curves(self, results: Dict):
        """Plot ROC curves for DR classification."""
        
        plt.figure(figsize=(12, 8))
        
        # Check if we have multiple classes
        unique_classes = np.unique(results['rg_targets'])
        if len(unique_classes) < 2:
            print("⚠️ Cannot create ROC curves: Only one class present in targets")
            # Create placeholder plot
            plt.text(0.5, 0.5, 'ROC curves unavailable:\nOnly one class present in test data', 
                    ha='center', va='center', fontsize=16, transform=plt.gca().transAxes)
            plt.title('DR Classification ROC Curves - Insufficient Data')
            plt.savefig('visualizations/dr_roc_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        # Plot ROC curve for each class (one-vs-rest)
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.dr_class_names)))
        
        for i, (class_name, color) in enumerate(zip(self.dr_class_names, colors)):
            if i in unique_classes:
                # Binary classification: class i vs all others
                y_true_binary = (results['rg_targets'] == i).astype(int)
                y_scores = results['rg_probabilities'][:, i]
                
                try:
                    fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
                    auc_score = roc_auc_score(y_true_binary, y_scores)
                    
                    plt.plot(fpr, tpr, color=color, lw=2, 
                            label=f'{class_name} (AUC = {auc_score:.3f})')
                except Exception as e:
                    print(f"⚠️ Could not plot ROC for {class_name}: {e}")
                    continue
        
        # Plot random classifier line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('DR Classification ROC Curves\n(5-Class ICDR Standard)')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        
        plt.savefig('visualizations/dr_roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_dr_class_distributions(self, results: Dict):
        """Plot DR class distributions."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # True distribution
        unique, counts = np.unique(results['rg_targets'], return_counts=True)
        class_names_true = [self.dr_class_names[i] for i in unique]
        
        bars1 = ax1.bar(range(len(unique)), counts, color='skyblue', alpha=0.7)
        ax1.set_title('True DR Severity Distribution')
        ax1.set_xlabel('DR Severity Class')
        ax1.set_ylabel('Count')
        ax1.set_xticks(range(len(unique)))
        ax1.set_xticklabels(class_names_true, rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars1, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom')
        
        # Predicted distribution
        unique_pred, counts_pred = np.unique(results['rg_predictions'], return_counts=True)
        class_names_pred = [self.dr_class_names[i] for i in unique_pred]
        
        bars2 = ax2.bar(range(len(unique_pred)), counts_pred, color='lightcoral', alpha=0.7)
        ax2.set_title('Predicted DR Severity Distribution')
        ax2.set_xlabel('DR Severity Class')
        ax2.set_ylabel('Count')
        ax2.set_xticks(range(len(unique_pred)))
        ax2.set_xticklabels(class_names_pred, rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars2, counts_pred):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom')
        
        plt.suptitle('DR Classification Distribution Comparison\n(5-Class ICDR Standard)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig('visualizations/dr_class_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_dr_summary(self, dr_metrics: Dict) -> Dict:
        """Create DR-only evaluation summary matching Phase 1 requirements."""
        
        # Extract key metrics for Phase 1 compliance
        accuracy = dr_metrics['accuracy']
        macro_f1 = dr_metrics['macro_avg']['f1-score']
        weighted_f1 = dr_metrics['weighted_avg']['f1-score']
        cohen_kappa = dr_metrics['cohen_kappa']
        
        # Get per-class metrics for medical significance
        per_class = dr_metrics['per_class_metrics']
        
        # Calculate clinical relevance metrics
        # Classes 3 and 4 are sight-threatening (Severe NPDR and PDR)
        sight_threatening_sensitivity = 0.0
        sight_threatening_count = 0
        
        if '3' in per_class:  # Severe NPDR
            sight_threatening_sensitivity += per_class['3'].get('recall', 0.0)
            sight_threatening_count += 1
            
        if '4' in per_class:  # PDR
            sight_threatening_sensitivity += per_class['4'].get('recall', 0.0) 
            sight_threatening_count += 1
        
        if sight_threatening_count > 0:
            sight_threatening_sensitivity /= sight_threatening_count
        
        # Calculate referable DR sensitivity (Classes 2, 3, 4)
        referable_sensitivity = 0.0
        referable_count = 0
        
        for class_idx in ['2', '3', '4']:  # Moderate NPDR, Severe NPDR, PDR
            if class_idx in per_class:
                referable_sensitivity += per_class[class_idx].get('recall', 0.0)
                referable_count += 1
        
        if referable_count > 0:
            referable_sensitivity /= referable_count
        
        # Phase 1 compliant summary
        summary = {
            'phase': 1,
            'grading_system': 'ICDR_5class',
            'model_performance': {
                'overall_accuracy': accuracy,
                'macro_f1_score': macro_f1,
                'weighted_f1_score': weighted_f1,
                'cohen_kappa': cohen_kappa,
                'macro_auc': dr_metrics.get('auc_macro', 0.0)
            },
            'clinical_performance': {
                'referable_dr_sensitivity': referable_sensitivity,
                'sight_threatening_dr_sensitivity': sight_threatening_sensitivity,
                'no_dr_specificity': per_class.get('0', {}).get('precision', 0.0)
            },
            'medical_compliance': {
                'meets_medical_grade_threshold': accuracy >= 0.90,
                'clinical_sensitivity_adequate': sight_threatening_sensitivity >= 0.85,
                'suitable_for_screening': accuracy >= 0.90 and sight_threatening_sensitivity >= 0.85
            },
            'class_performance': {
                self.dr_class_names[i]: per_class.get(str(i), {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0})
                for i in range(len(self.dr_class_names))
            }
        }
        
        # Display Phase 1 summary
        print(f"\n📊 Phase 1 DR Classification Summary:")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Referable DR Sensitivity: {referable_sensitivity:.4f}")
        print(f"Sight-threatening DR Sensitivity: {sight_threatening_sensitivity:.4f}")
        print(f"Medical-grade Threshold (≥90%): {'✅ PASSED' if summary['medical_compliance']['meets_medical_grade_threshold'] else '❌ FAILED'}")
        print(f"Clinical Readiness: {'✅ READY' if summary['medical_compliance']['suitable_for_screening'] else '❌ NEEDS IMPROVEMENT'}")
        
        return summary