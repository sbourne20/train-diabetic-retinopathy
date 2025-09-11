#!/usr/bin/env python3
"""
Multi-Architecture Ensemble Models for Diabetic Retinopathy Classification

Based on research paper achieving 96.96% accuracy using:
- EfficientNetB2 (96.27% individual)
- ResNet50 (94.95% individual)  
- DenseNet121 (91.21% individual)

This module implements the three model architectures and ensemble combination logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import densenet121, DenseNet121_Weights
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import OrderedDict

class EfficientNetB2_DR(nn.Module):
    """
    EfficientNetB2 for diabetic retinopathy classification.
    Target accuracy: 96.27% (research validated)
    """
    
    def __init__(self, num_classes: int = 5, dropout: float = 0.3, pretrained: bool = True):
        super(EfficientNetB2_DR, self).__init__()
        
        # Load pre-trained EfficientNetB2
        if pretrained:
            self.backbone = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        else:
            self.backbone = efficientnet_b2(weights=None)
        
        # Get the number of features from the classifier
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier for diabetic retinopathy
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        # Additional heads for medical-grade outputs
        self.referable_head = nn.Linear(512, 2)  # Referable DR (classes 2,3,4 vs 0,1)
        self.sight_threatening_head = nn.Linear(512, 2)  # Sight-threatening (classes 3,4 vs 0,1,2)
        self.confidence_head = nn.Linear(512, 1)  # Confidence estimation
        
        self.num_classes = num_classes
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features using backbone (without final classifier)
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        features = self.backbone.classifier[0](features)  # First dropout
        features = self.backbone.classifier[1](features)  # First linear layer
        features = self.backbone.classifier[2](features)  # ReLU
        
        # Main DR classification
        dr_logits = self.backbone.classifier[4](self.backbone.classifier[3](features))  # Final layers
        
        # Additional medical heads
        referable_logits = self.referable_head(features)
        sight_threatening_logits = self.sight_threatening_head(features)
        confidence = torch.sigmoid(self.confidence_head(features))
        
        return {
            'dr_logits': dr_logits,
            'referable_dr_logits': referable_logits,
            'sight_threatening_logits': sight_threatening_logits,
            'grading_confidence': confidence,
            'features': features
        }

class ResNet50_DR(nn.Module):
    """
    ResNet50 for diabetic retinopathy classification.
    Target accuracy: 94.95% (research validated)
    """
    
    def __init__(self, num_classes: int = 5, dropout: float = 0.3, pretrained: bool = True):
        super(ResNet50_DR, self).__init__()
        
        # Load pre-trained ResNet50
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = resnet50(weights=None)
        
        # Get number of features
        num_features = self.backbone.fc.in_features
        
        # Replace final fully connected layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        # Additional heads for medical-grade outputs
        self.referable_head = nn.Linear(512, 2)
        self.sight_threatening_head = nn.Linear(512, 2)
        self.confidence_head = nn.Linear(512, 1)
        
        self.num_classes = num_classes
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features through ResNet layers
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1)
        
        # Get intermediate features before final classification
        features = self.backbone.fc[0](features)  # First dropout
        features = self.backbone.fc[1](features)  # First linear
        features = self.backbone.fc[2](features)  # ReLU
        
        # Main DR classification
        dr_logits = self.backbone.fc[4](self.backbone.fc[3](features))  # Final layers
        
        # Additional medical heads
        referable_logits = self.referable_head(features)
        sight_threatening_logits = self.sight_threatening_head(features)
        confidence = torch.sigmoid(self.confidence_head(features))
        
        return {
            'dr_logits': dr_logits,
            'referable_dr_logits': referable_logits,
            'sight_threatening_logits': sight_threatening_logits,
            'grading_confidence': confidence,
            'features': features
        }

class DenseNet121_DR(nn.Module):
    """
    DenseNet121 for diabetic retinopathy classification.
    Target accuracy: 91.21% (research validated)
    """
    
    def __init__(self, num_classes: int = 5, dropout: float = 0.3, pretrained: bool = True):
        super(DenseNet121_DR, self).__init__()
        
        # Load pre-trained DenseNet121
        if pretrained:
            self.backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        else:
            self.backbone = densenet121(weights=None)
        
        # Get number of features
        num_features = self.backbone.classifier.in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        # Additional heads for medical-grade outputs
        self.referable_head = nn.Linear(512, 2)
        self.sight_threatening_head = nn.Linear(512, 2)
        self.confidence_head = nn.Linear(512, 1)
        
        self.num_classes = num_classes
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features through DenseNet
        features = self.backbone.features(x)
        features = F.relu(features, inplace=True)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        
        # Get intermediate features
        features = self.backbone.classifier[0](features)  # Dropout
        features = self.backbone.classifier[1](features)  # Linear
        features = self.backbone.classifier[2](features)  # ReLU
        
        # Main DR classification
        dr_logits = self.backbone.classifier[4](self.backbone.classifier[3](features))
        
        # Additional medical heads
        referable_logits = self.referable_head(features)
        sight_threatening_logits = self.sight_threatening_head(features)
        confidence = torch.sigmoid(self.confidence_head(features))
        
        return {
            'dr_logits': dr_logits,
            'referable_dr_logits': referable_logits,
            'sight_threatening_logits': sight_threatening_logits,
            'grading_confidence': confidence,
            'features': features
        }

class DRMultiArchitectureEnsemble(nn.Module):
    """
    Multi-architecture ensemble for diabetic retinopathy classification.
    
    Combines EfficientNetB2, ResNet50, and DenseNet121 using simple averaging
    to achieve 96.96% accuracy as demonstrated in research literature.
    """
    
    def __init__(self, 
                 num_classes: int = 5, 
                 dropout: float = 0.3, 
                 pretrained: bool = True,
                 model_weights: Optional[List[float]] = None):
        super(DRMultiArchitectureEnsemble, self).__init__()
        
        # Initialize individual models
        self.efficientnet_b2 = EfficientNetB2_DR(num_classes, dropout, pretrained)
        self.resnet50 = ResNet50_DR(num_classes, dropout, pretrained)
        self.densenet121 = DenseNet121_DR(num_classes, dropout, pretrained)
        
        # Model weights for ensemble averaging (default: equal weights)
        if model_weights is None:
            self.model_weights = [1.0/3, 1.0/3, 1.0/3]  # Equal weighting
        else:
            assert len(model_weights) == 3, "Must provide exactly 3 model weights"
            assert abs(sum(model_weights) - 1.0) < 1e-6, "Model weights must sum to 1.0"
            self.model_weights = model_weights
        
        self.num_classes = num_classes
        self.model_names = ['efficientnetb2', 'resnet50', 'densenet121']
        
    def forward(self, x: torch.Tensor, return_individual: bool = False, 
                return_dict: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict], Dict[str, torch.Tensor]]:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            return_individual: If True, return individual model predictions
            return_dict: If False, return only main DR logits (for compatibility)
            
        Returns:
            - If return_dict=False: Main DR classification logits [batch_size, num_classes]
            - If return_dict=True and return_individual=False: Dict with ensemble predictions
            - If return_dict=True and return_individual=True: (logits, individual_dict)
        """
        # Get predictions from individual models
        efficientnet_out = self.efficientnet_b2(x)
        resnet_out = self.resnet50(x)
        densenet_out = self.densenet121(x)
        
        individual_outputs = [efficientnet_out, resnet_out, densenet_out]
        
        # Ensemble averaging for each output type
        ensemble_dr_logits = torch.zeros_like(efficientnet_out['dr_logits'])
        ensemble_referable_logits = torch.zeros_like(efficientnet_out['referable_dr_logits'])
        ensemble_sight_threatening_logits = torch.zeros_like(efficientnet_out['sight_threatening_logits'])
        ensemble_confidence = torch.zeros_like(efficientnet_out['grading_confidence'])
        
        # Weighted averaging
        for i, (output, weight) in enumerate(zip(individual_outputs, self.model_weights)):
            ensemble_dr_logits += weight * output['dr_logits']
            ensemble_referable_logits += weight * output['referable_dr_logits']
            ensemble_sight_threatening_logits += weight * output['sight_threatening_logits']
            ensemble_confidence += weight * output['grading_confidence']
        
        # Prepare return dictionary
        result = {
            'dr_logits': ensemble_dr_logits,
            'referable_dr_logits': ensemble_referable_logits,
            'sight_threatening_logits': ensemble_sight_threatening_logits,
            'grading_confidence': ensemble_confidence
        }
        
        # Handle different return formats
        if not return_dict:
            # Simple compatibility mode - return only main DR logits
            return ensemble_dr_logits
        
        # Add individual predictions if requested
        if return_individual:
            result['individual_predictions'] = {
                'efficientnetb2': efficientnet_out,
                'resnet50': resnet_out,
                'densenet121': densenet_out
            }
            
            # Add probability distributions for analysis
            result['individual_probabilities'] = {
                'efficientnetb2': F.softmax(efficientnet_out['dr_logits'], dim=1),
                'resnet50': F.softmax(resnet_out['dr_logits'], dim=1),
                'densenet121': F.softmax(densenet_out['dr_logits'], dim=1)
            }
            
            # For evaluator compatibility, also return as tuple
            individual_logits = {
                'efficientnetb2': efficientnet_out['dr_logits'],
                'resnet50': resnet_out['dr_logits'], 
                'densenet121': densenet_out['dr_logits']
            }
            return ensemble_dr_logits, individual_logits
        
        return result
    
    def get_individual_models(self) -> Dict[str, nn.Module]:
        """Return dictionary of individual models for separate training/evaluation."""
        return {
            'efficientnetb2': self.efficientnet_b2,
            'resnet50': self.resnet50,
            'densenet121': self.densenet121
        }
    
    def load_individual_checkpoints(self, checkpoint_paths: Dict[str, str], device: torch.device):
        """
        Load pre-trained individual model checkpoints.
        
        Args:
            checkpoint_paths: Dict mapping model names to checkpoint file paths
            device: Device to load models on
        """
        models = self.get_individual_models()
        
        for model_name, checkpoint_path in checkpoint_paths.items():
            if model_name in models:
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    if 'model_state_dict' in checkpoint:
                        models[model_name].load_state_dict(checkpoint['model_state_dict'])
                    else:
                        models[model_name].load_state_dict(checkpoint)
                    print(f"✅ Loaded checkpoint for {model_name}: {checkpoint_path}")
                except Exception as e:
                    print(f"❌ Failed to load checkpoint for {model_name}: {e}")
            else:
                print(f"⚠️ Unknown model name: {model_name}")

def calculate_ensemble_metrics(individual_predictions: List[torch.Tensor], 
                             targets: torch.Tensor) -> Dict[str, float]:
    """
    Calculate ensemble performance metrics.
    
    Args:
        individual_predictions: List of prediction tensors from individual models
        targets: Ground truth labels
        
    Returns:
        Dictionary containing ensemble metrics
    """
    # Simple averaging ensemble
    ensemble_logits = torch.stack(individual_predictions).mean(dim=0)
    ensemble_preds = torch.argmax(ensemble_logits, dim=1)
    
    # Calculate metrics
    accuracy = (ensemble_preds == targets).float().mean().item()
    
    # Per-class metrics
    num_classes = individual_predictions[0].size(1)
    per_class_accuracy = []
    per_class_sensitivity = []
    per_class_specificity = []
    
    for class_id in range(num_classes):
        class_mask = (targets == class_id)
        if class_mask.sum() > 0:
            # Sensitivity (True Positive Rate)
            tp = ((ensemble_preds == class_id) & (targets == class_id)).sum().float()
            fn = ((ensemble_preds != class_id) & (targets == class_id)).sum().float()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # Specificity (True Negative Rate)
            tn = ((ensemble_preds != class_id) & (targets != class_id)).sum().float()
            fp = ((ensemble_preds == class_id) & (targets != class_id)).sum().float()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            class_accuracy = (ensemble_preds[class_mask] == targets[class_mask]).float().mean().item()
            
            per_class_accuracy.append(class_accuracy)
            per_class_sensitivity.append(sensitivity.item())
            per_class_specificity.append(specificity.item())
        else:
            per_class_accuracy.append(0.0)
            per_class_sensitivity.append(0.0)
            per_class_specificity.append(0.0)
    
    return {
        'ensemble_accuracy': accuracy,
        'mean_sensitivity': np.mean(per_class_sensitivity),
        'mean_specificity': np.mean(per_class_specificity),
        'per_class_accuracy': per_class_accuracy,
        'per_class_sensitivity': per_class_sensitivity,
        'per_class_specificity': per_class_specificity
    }

def validate_medical_grade_ensemble(metrics: Dict[str, float]) -> Dict[str, bool]:
    """
    Validate if ensemble meets medical-grade standards.
    
    Args:
        metrics: Dictionary containing performance metrics
        
    Returns:
        Dictionary containing validation results
    """
    # Medical-grade thresholds
    ACCURACY_THRESHOLD = 0.90  # 90% minimum accuracy
    SENSITIVITY_THRESHOLD = 0.85  # 85% minimum sensitivity
    SPECIFICITY_THRESHOLD = 0.90  # 90% minimum specificity
    TARGET_ACCURACY = 0.9696  # Research target: 96.96%
    
    validation = {
        'meets_accuracy_threshold': metrics['ensemble_accuracy'] >= ACCURACY_THRESHOLD,
        'meets_sensitivity_threshold': metrics['mean_sensitivity'] >= SENSITIVITY_THRESHOLD,
        'meets_specificity_threshold': metrics['mean_specificity'] >= SPECIFICITY_THRESHOLD,
        'achieves_research_target': metrics['ensemble_accuracy'] >= TARGET_ACCURACY,
        'medical_grade_pass': False
    }
    
    # Overall medical grade pass requires all thresholds
    validation['medical_grade_pass'] = (
        validation['meets_accuracy_threshold'] and
        validation['meets_sensitivity_threshold'] and
        validation['meets_specificity_threshold']
    )
    
    return validation

# Factory function for easy model creation
def create_ensemble_model(num_classes: int = 5, 
                         dropout: float = 0.3, 
                         pretrained: bool = True,
                         model_weights: Optional[List[float]] = None) -> DRMultiArchitectureEnsemble:
    """
    Factory function to create multi-architecture ensemble model.
    
    Args:
        num_classes: Number of output classes (5 for DR)
        dropout: Dropout rate for regularization
        pretrained: Use ImageNet pre-trained weights
        model_weights: Custom weights for ensemble averaging
        
    Returns:
        Initialized ensemble model
    """
    return DRMultiArchitectureEnsemble(
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained,
        model_weights=model_weights
    )

if __name__ == "__main__":
    # Test ensemble model creation and forward pass
    print("🧪 Testing Multi-Architecture Ensemble Model")
    
    # Create model
    model = create_ensemble_model(num_classes=5, dropout=0.3, pretrained=True)
    
    # Test input
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    with torch.no_grad():
        output = model(test_input, return_individual=True)
    
    print(f"✅ Ensemble DR logits shape: {output['dr_logits'].shape}")
    print(f"✅ Individual models: {list(output['individual_predictions'].keys())}")
    print(f"✅ Confidence scores shape: {output['grading_confidence'].shape}")
    
    # Test metrics calculation
    dummy_targets = torch.randint(0, 5, (batch_size,))
    individual_preds = [
        output['individual_predictions']['efficientnetb2']['dr_logits'],
        output['individual_predictions']['resnet50']['dr_logits'],
        output['individual_predictions']['densenet121']['dr_logits']
    ]
    
    metrics = calculate_ensemble_metrics(individual_preds, dummy_targets)
    validation = validate_medical_grade_ensemble(metrics)
    
    print(f"📊 Test ensemble accuracy: {metrics['ensemble_accuracy']:.4f}")
    print(f"🏥 Medical grade validation: {validation}")
    print("✅ Ensemble model test completed successfully!")