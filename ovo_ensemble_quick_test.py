#!/usr/bin/env python3
"""
Quick OVO Ensemble Test - Fast voting fix validation without slow calibration

This script tests the core voting mechanism fix immediately without waiting
for threshold optimization. We can add calibration later if needed.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from itertools import combinations
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BinaryClassifier(nn.Module):
    """Binary classifier - matches your trained model architecture exactly"""

    def __init__(self, model_name='mobilenet_v2', freeze_weights=True, dropout=0.5):
        super().__init__()
        self.model_name = model_name

        if model_name == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=True)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif model_name == 'inception_v3':
            self.backbone = models.inception_v3(pretrained=True)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.backbone.training = True
        elif model_name == 'densenet121':
            self.backbone = models.densenet121(pretrained=True)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        if freeze_weights:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # MATCH TRAINED MODEL ARCHITECTURE: Multi-layer classifier with BatchNorm
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),                    # Layer 0: Dropout
            nn.Linear(num_features, 512),          # Layer 1: Linear reduction
            nn.ReLU(),                             # Layer 2: Activation
            nn.BatchNorm1d(512),                   # Layer 3: BatchNorm
            nn.Dropout(dropout),                   # Layer 4: Dropout
            nn.Linear(512, 128),                   # Layer 5: Further reduction
            nn.ReLU(),                             # Layer 6: Activation
            nn.BatchNorm1d(128),                   # Layer 7: BatchNorm
            nn.Dropout(dropout),                   # Layer 8: Dropout
            nn.Linear(128, 1),                     # Layer 9: Final output
            nn.Sigmoid()                           # Layer 10: Sigmoid activation
        )

    def forward(self, x):
        if self.model_name == 'inception_v3' and x.size(-1) < 75:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        if self.model_name == 'inception_v3' and self.training:
            features, aux_features = self.backbone(x)
        else:
            features = self.backbone(x)

        if isinstance(features, tuple):
            features = features[0]

        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        return self.classifier(features)

class QuickFixedOVOEnsemble(nn.Module):
    """QUICK FIXED OVO Ensemble - no slow calibration, just core voting fix"""

    def __init__(self, base_models=['mobilenet_v2', 'inception_v3', 'densenet121'],
                 num_classes=5, freeze_weights=True, dropout=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.base_models = base_models
        self.class_pairs = list(combinations(range(num_classes), 2))

        # Create binary classifiers
        self.classifiers = nn.ModuleDict()
        for model_name in base_models:
            model_classifiers = nn.ModuleDict()
            for class_a, class_b in self.class_pairs:
                classifier_name = f"pair_{class_a}_{class_b}"
                model_classifiers[classifier_name] = BinaryClassifier(
                    model_name=model_name,
                    freeze_weights=freeze_weights,
                    dropout=dropout
                )
            self.classifiers[model_name] = model_classifiers

        # Binary classifier accuracies (from your analysis)
        self.binary_accuracies = {
            'mobilenet_v2': {
                'pair_0_1': 0.923, 'pair_0_2': 0.911, 'pair_0_3': 0.996, 'pair_0_4': 0.992,
                'pair_1_2': 0.846, 'pair_1_3': 0.903, 'pair_1_4': 0.911, 'pair_2_3': 0.950,
                'pair_2_4': 0.933, 'pair_3_4': 0.821
            },
            'inception_v3': {
                'pair_0_1': 0.921, 'pair_0_2': 0.891, 'pair_0_3': 0.995, 'pair_0_4': 0.987,
                'pair_1_2': 0.851, 'pair_1_3': 0.910, 'pair_1_4': 0.905, 'pair_2_3': 0.951,
                'pair_2_4': 0.922, 'pair_3_4': 0.813
            },
            'densenet121': {
                'pair_0_1': 0.920, 'pair_0_2': 0.913, 'pair_0_3': 0.998, 'pair_0_4': 0.993,
                'pair_1_2': 0.852, 'pair_1_3': 0.913, 'pair_1_4': 0.923, 'pair_2_3': 0.953,
                'pair_2_4': 0.950, 'pair_3_4': 0.844
            }
        }

        logger.info(f"🚀 Quick Fixed OVO Ensemble initialized (no slow calibration)")

    def forward(self, x, return_individual=False):
        """QUICK FIXED forward pass with proper confidence weighting (no calibration)"""
        batch_size = x.size(0)
        device = x.device

        # Initialize vote accumulation
        class_scores = torch.zeros(batch_size, self.num_classes, device=device)
        total_weights = torch.zeros(batch_size, self.num_classes, device=device)

        individual_predictions = {} if return_individual else None

        for model_name, model_classifiers in self.classifiers.items():
            if return_individual:
                individual_predictions[model_name] = torch.zeros(batch_size, self.num_classes, device=device)

            for classifier_name, classifier in model_classifiers.items():
                class_a, class_b = map(int, classifier_name.split('_')[1:])

                # Get binary prediction
                binary_output = classifier(x).squeeze()
                if binary_output.dim() == 0:
                    binary_output = binary_output.unsqueeze(0)

                # FIXED: True confidence weighting (no calibration needed)
                true_confidence = torch.abs(binary_output - 0.5) * 2  # 0 to 1 scale

                # FIXED: Accuracy-based weighting
                accuracy_weight = self.binary_accuracies[model_name][classifier_name]

                # Combined weight: confidence * accuracy^2 (emphasize high-accuracy classifiers)
                combined_weight = true_confidence * (accuracy_weight ** 2)

                # FIXED: Simple probability calculation (no threshold optimization)
                # Use 0.5 threshold for now - still much better than original voting
                prob_class_a = 1.0 - binary_output  # probability for class_a
                prob_class_b = binary_output         # probability for class_b

                # Accumulate weighted votes
                class_scores[:, class_a] += prob_class_a * combined_weight
                class_scores[:, class_b] += prob_class_b * combined_weight

                total_weights[:, class_a] += combined_weight
                total_weights[:, class_b] += combined_weight

                if return_individual:
                    individual_predictions[model_name][:, class_a] += prob_class_a * combined_weight
                    individual_predictions[model_name][:, class_b] += prob_class_b * combined_weight

        # FIXED: Proper normalization by accumulated weights
        normalized_scores = class_scores / (total_weights + 1e-8)

        # FIXED: Apply temperature scaling for better calibration
        temperature = 1.2  # Slightly cool down overconfident predictions
        final_predictions = F.softmax(normalized_scores / temperature, dim=1)

        result = {
            'logits': final_predictions,
            'raw_scores': normalized_scores,
            'votes': class_scores
        }

        if return_individual:
            # Normalize individual predictions
            for model_name in individual_predictions:
                individual_weights = total_weights / len(self.base_models)  # Approximate
                individual_predictions[model_name] = individual_predictions[model_name] / (individual_weights + 1e-8)
            result['individual_predictions'] = individual_predictions

        return result

def load_trained_ensemble_quick(results_dir, config):
    """Load the trained ensemble quickly with fixed voting mechanism"""

    # Determine device for optimal performance
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🎮 Using device: {device}")

    # Create quick fixed ensemble
    ensemble = QuickFixedOVOEnsemble(
        base_models=config['model']['base_models'],
        num_classes=config['data']['num_classes'],
        freeze_weights=True,
        dropout=0.5
    )

    # Load trained weights
    models_dir = Path(results_dir) / "models"
    loaded_count = 0

    for model_name in config['model']['base_models']:
        for class_a, class_b in combinations(range(config['data']['num_classes']), 2):
            pair_name = f"pair_{class_a}_{class_b}"
            model_path = models_dir / f"best_{model_name}_{class_a}_{class_b}.pth"

            if model_path.exists():
                try:
                    # Load directly to GPU if available for faster performance
                    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        ensemble.classifiers[model_name][pair_name].load_state_dict(checkpoint['model_state_dict'])
                    else:
                        ensemble.classifiers[model_name][pair_name].load_state_dict(checkpoint)

                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"Failed to load {model_path.name}: {e}")
            else:
                logger.warning(f"Missing: {model_path.name}")

    logger.info(f"📦 Loaded {loaded_count}/30 binary classifiers with quick fixed voting")
    return ensemble

def evaluate_quick_ensemble(ensemble, test_loader, config):
    """Evaluate the quick fixed ensemble"""

    logger.info("📋 Evaluating Quick Fixed OVO Ensemble")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ensemble = ensemble.to(device)
    ensemble.eval()

    all_predictions = []
    all_targets = []
    individual_predictions = {model: [] for model in config['model']['base_models']}

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = ensemble(images, return_individual=True)

            _, ensemble_pred = torch.max(outputs['logits'], 1)
            all_predictions.extend(ensemble_pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Individual model predictions
            for model_name, model_votes in outputs['individual_predictions'].items():
                _, individual_pred = torch.max(model_votes, 1)
                individual_predictions[model_name].extend(individual_pred.cpu().numpy())

    # Calculate metrics
    ensemble_accuracy = accuracy_score(all_targets, all_predictions)
    individual_accuracies = {
        model: accuracy_score(all_targets, preds)
        for model, preds in individual_predictions.items()
    }

    # Per-class performance
    cm = confusion_matrix(all_targets, all_predictions)
    per_class_recall = cm.diagonal() / cm.sum(axis=1)

    results = {
        'ensemble_accuracy': ensemble_accuracy,
        'individual_accuracies': individual_accuracies,
        'per_class_recall': per_class_recall.tolist(),
        'confusion_matrix': cm.tolist(),
        'medical_grade_achieved': ensemble_accuracy >= 0.90
    }

    return results

def main():
    """Main function for quick ensemble test"""

    print("⚡ QUICK FIXED OVO ENSEMBLE TEST (NO SLOW CALIBRATION)")
    print("=" * 60)

    # Configuration
    results_dir = "./ovo_ensemble_results_v2"
    dataset_path = "./dataset6"

    # Load config
    config_path = Path(results_dir) / "ovo_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'model': {'base_models': ['mobilenet_v2', 'inception_v3', 'densenet121']},
            'data': {'num_classes': 5, 'img_size': 224}
        }

    # Load quick fixed ensemble
    logger.info("🏗️ Loading trained models with quick fixed voting mechanism...")
    ensemble = load_trained_ensemble_quick(results_dir, config)

    # Prepare test dataset
    test_transform = transforms.Compose([
        transforms.Resize((config['data']['img_size'], config['data']['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = ImageFolder(f"{dataset_path}/test", transform=test_transform)

    # Optimize DataLoader for GPU performance
    use_cuda = torch.cuda.is_available()
    test_loader = DataLoader(
        test_dataset,
        batch_size=64 if use_cuda else 32,  # Larger batch for GPU
        shuffle=False,
        num_workers=8 if use_cuda else 4,   # More workers for GPU
        pin_memory=use_cuda,                # Pin memory for GPU transfer
        persistent_workers=True             # Keep workers alive
    )

    logger.info(f"📊 Test dataset: {len(test_dataset)} images")

    # Evaluate quick fixed ensemble (NO SLOW CALIBRATION)
    logger.info("⚡ Evaluating quick fixed ensemble (skipping slow calibration)...")
    results = evaluate_quick_ensemble(ensemble, test_loader, config)

    # Display results
    print("\n" + "="*60)
    print("🏆 QUICK FIXED ENSEMBLE RESULTS")
    print("="*60)
    print(f"🎯 Ensemble Accuracy: {results['ensemble_accuracy']:.4f} ({results['ensemble_accuracy']*100:.2f}%)")
    print(f"🏥 Medical Grade: {'✅ ACHIEVED' if results['medical_grade_achieved'] else '❌ NOT ACHIEVED'}")

    print(f"\n📊 Per-Class Recall:")
    for i, recall in enumerate(results['per_class_recall']):
        print(f"   Class {i}: {recall:.3f} ({recall*100:.1f}%)")

    print(f"\n📊 Individual Model Performance:")
    for model_name, accuracy in results['individual_accuracies'].items():
        print(f"   {model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Save results
    results_path = Path(results_dir) / "results" / "quick_fixed_ensemble_results.json"
    results_path.parent.mkdir(exist_ok=True)

    with open(results_path, 'w') as f:
        json.dump({k: float(v) if isinstance(v, np.float64) else v for k, v in results.items()}, f, indent=2)

    print(f"\n💾 Results saved: {results_path}")

    # Performance comparison
    print(f"\n📈 PERFORMANCE COMPARISON:")
    print(f"   Original ensemble: ~80.9% (with voting issues)")
    print(f"   Quick fixed ensemble: {results['ensemble_accuracy']*100:.1f}% (with voting fixes)")
    improvement = (results['ensemble_accuracy'] - 0.809) * 100
    print(f"   Improvement: +{improvement:.1f} percentage points")

    print("="*60)

    return results

if __name__ == "__main__":
    main()