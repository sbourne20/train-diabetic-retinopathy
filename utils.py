import os
import random
import numpy as np
import torch
import torch.nn as nn
import json
import warnings
from typing import Dict, List, Tuple, Optional
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress sklearn warnings to avoid STDERR confusion
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

def set_seed(seed: int = 42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_directories(config):
    """Create necessary directories for the project."""
    dirs = [
        config.output_dir,
        config.checkpoint_dir,
        config.logs_dir,
        "models",
        "data",
        "visualizations"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute class weights for imbalanced dataset."""
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight(
        'balanced',
        classes=unique_labels,
        y=labels
    )
    
    # Handle missing classes
    weights = np.ones(num_classes)
    for i, label in enumerate(unique_labels):
        weights[label] = class_weights[i]
    
    return torch.FloatTensor(weights)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict:
    """Calculate comprehensive metrics for classification."""
    metrics = {}
    
    # Basic metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics['accuracy'] = report['accuracy']
    metrics['macro_avg'] = report['macro avg']
    metrics['weighted_avg'] = report['weighted avg']
    
    # Per-class metrics
    for i in range(len(np.unique(y_true))):
        if str(i) in report:
            metrics[f'class_{i}'] = report[str(i)]
    
    # AUC if probabilities provided
    if y_prob is not None:
        try:
            if y_prob.shape[1] > 2:  # Multi-class
                metrics['auc_macro'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                metrics['auc_weighted'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            else:  # Binary
                metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
        except ValueError:
            pass
    
    return metrics

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str], save_path: str = None):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_results(results: Dict, filepath: str):
    """Save results to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)

def load_medical_terms(filepath: str) -> Dict:
    """Load medical terminology for text generation."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Default medical terms if file doesn't exist
        return {
            "severity_terms": {
                "0": ["no diabetic retinopathy", "normal retina", "no pathological changes"],
                "1": ["mild non-proliferative diabetic retinopathy", "early signs", "microaneurysms present"],
                "2": ["moderate non-proliferative diabetic retinopathy", "dot-blot hemorrhages", "cotton wool spots"],
                "3": ["severe non-proliferative diabetic retinopathy", "extensive hemorrhages", "venous abnormalities"]
            },
            "macular_edema_terms": {
                "0": ["no macular edema", "normal macula", "no fluid accumulation"],
                "1": ["mild macular edema risk", "early changes", "potential fluid retention"],
                "2": ["high macular edema risk", "significant fluid accumulation", "vision-threatening changes"]
            },
            "anatomical_terms": [
                "optic disc", "macula", "fovea", "retinal vessels", "microaneurysms",
                "hemorrhages", "exudates", "cotton wool spots", "venous beading"
            ]
        }

class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        return self.counter >= self.patience
    
    def restore_weights(self, model: nn.Module):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)

class AverageMeter:
    """Compute and store the average and current value."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count