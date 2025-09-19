#!/usr/bin/env python3
"""
Quick Final OVO Fix - No Calibration, Immediate Results

This version skips the slow calibration and provides immediate results
with the diagnostic data improvements.
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
    """Binary classifier - matches trained model architecture exactly"""

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

        # EXACT ARCHITECTURE MATCH
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
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
    """Quick Fixed OVO Ensemble - No Calibration"""

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

        # REAL accuracies from diagnostic analysis
        self.binary_accuracies = {
            'mobilenet_v2': {
                'pair_0_1': 0.8745, 'pair_0_2': 0.8228, 'pair_0_3': 0.9827, 'pair_0_4': 0.9845,
                'pair_1_2': 0.8072, 'pair_1_3': 0.8667, 'pair_1_4': 0.9251, 'pair_2_3': 0.8567,
                'pair_2_4': 0.8836, 'pair_3_4': 0.7205
            },
            'inception_v3': {
                'pair_0_1': 0.8340, 'pair_0_2': 0.7850, 'pair_0_3': 0.9286, 'pair_0_4': 0.9732,
                'pair_1_2': 0.7972, 'pair_1_3': 0.8267, 'pair_1_4': 0.8719, 'pair_2_3': 0.8206,
                'pair_2_4': 0.8501, 'pair_3_4': 0.7654
            },
            'densenet121': {
                'pair_0_1': 0.8870, 'pair_0_2': 0.8791, 'pair_0_3': 0.9827, 'pair_0_4': 0.9881,
                'pair_1_2': 0.8483, 'pair_1_3': 0.8767, 'pair_1_4': 0.8968, 'pair_2_3': 0.8927,
                'pair_2_4': 0.8819, 'pair_3_4': 0.7937
            }
        }

        # Problem classifiers (below 80%)
        self.weak_classifiers = {
            ('inception_v3', 'pair_0_2'), ('inception_v3', 'pair_1_2'),
            ('inception_v3', 'pair_3_4'), ('mobilenet_v2', 'pair_3_4')
        }

        # AGGRESSIVE Class 1 boost
        self.class_weights = torch.tensor([1.0, 10.0, 2.0, 4.0, 5.0])  # 10x boost for Class 1
        self.class1_pairs = ['pair_0_1', 'pair_1_2', 'pair_1_3', 'pair_1_4']

        logger.info(f"🚀 Quick Fixed OVO Ensemble (no calibration)")

    def forward(self, x, return_individual=False):
        batch_size = x.size(0)
        device = x.device

        class_scores = torch.zeros(batch_size, self.num_classes, device=device)
        total_weights = torch.zeros(batch_size, self.num_classes, device=device)

        individual_predictions = {} if return_individual else None
        class_weights = self.class_weights.to(device)

        for model_name, model_classifiers in self.classifiers.items():
            if return_individual:
                individual_predictions[model_name] = torch.zeros(batch_size, self.num_classes, device=device)

            for classifier_name, classifier in model_classifiers.items():
                class_a, class_b = map(int, classifier_name.split('_')[1:])

                binary_output = classifier(x).squeeze()
                if binary_output.dim() == 0:
                    binary_output = binary_output.unsqueeze(0)

                # Real accuracy weighting
                base_accuracy = self.binary_accuracies[model_name][classifier_name]

                # Weak classifier penalty
                if (model_name, classifier_name) in self.weak_classifiers:
                    accuracy_weight = (base_accuracy ** 4) * 0.4  # Heavy penalty
                else:
                    accuracy_weight = base_accuracy ** 2  # Standard weighting

                # Excellent classifier boost
                if base_accuracy > 0.95:
                    accuracy_weight *= 1.8

                # Confidence calculation
                confidence = torch.abs(binary_output - 0.5) * 2
                weighted_confidence = confidence * accuracy_weight

                # Class weights with Class 1 emergency boost
                class_a_weight = class_weights[class_a]
                class_b_weight = class_weights[class_b]

                # Class 1 emergency boost
                if classifier_name in self.class1_pairs:
                    if class_a == 1:
                        class_a_weight *= 3.0  # Additional 3x boost
                    if class_b == 1:
                        class_b_weight *= 3.0  # Additional 3x boost

                # Vote calculation
                prob_class_a = (1.0 - binary_output) * class_a_weight * weighted_confidence
                prob_class_b = binary_output * class_b_weight * weighted_confidence

                class_scores[:, class_a] += prob_class_a
                class_scores[:, class_b] += prob_class_b

                total_weights[:, class_a] += class_a_weight * weighted_confidence
                total_weights[:, class_b] += class_b_weight * weighted_confidence

                if return_individual:
                    individual_predictions[model_name][:, class_a] += prob_class_a
                    individual_predictions[model_name][:, class_b] += prob_class_b

        # Normalize and softmax
        normalized_scores = class_scores / (total_weights + 1e-8)
        final_predictions = F.softmax(normalized_scores, dim=1)

        result = {
            'logits': final_predictions,
            'raw_scores': normalized_scores
        }

        if return_individual:
            for model_name in individual_predictions:
                model_weights = total_weights / len(self.base_models)
                individual_predictions[model_name] = individual_predictions[model_name] / (model_weights + 1e-8)
            result['individual_predictions'] = individual_predictions

        return result

def load_quick_ensemble(results_dir, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🎮 Using device: {device}")

    ensemble = QuickFixedOVOEnsemble(
        base_models=config['model']['base_models'],
        num_classes=config['data']['num_classes'],
        freeze_weights=True,
        dropout=0.5
    )

    models_dir = Path(results_dir) / "models"
    loaded_count = 0

    for model_name in config['model']['base_models']:
        for class_a, class_b in combinations(range(config['data']['num_classes']), 2):
            pair_name = f"pair_{class_a}_{class_b}"
            model_path = models_dir / f"best_{model_name}_{class_a}_{class_b}.pth"

            if model_path.exists():
                try:
                    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        ensemble.classifiers[model_name][pair_name].load_state_dict(checkpoint['model_state_dict'])
                    else:
                        ensemble.classifiers[model_name][pair_name].load_state_dict(checkpoint)

                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"Failed to load {model_path.name}: {e}")

    logger.info(f"📦 Loaded {loaded_count}/30 binary classifiers")
    return ensemble

def evaluate_quick_ensemble(ensemble, test_loader, config):
    logger.info("📋 Evaluating Quick Fixed Ensemble")
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

            for model_name, model_votes in outputs['individual_predictions'].items():
                _, individual_pred = torch.max(model_votes, 1)
                individual_predictions[model_name].extend(individual_pred.cpu().numpy())

    ensemble_accuracy = accuracy_score(all_targets, all_predictions)
    individual_accuracies = {
        model: accuracy_score(all_targets, preds)
        for model, preds in individual_predictions.items()
    }

    cm = confusion_matrix(all_targets, all_predictions)
    per_class_recall = cm.diagonal() / cm.sum(axis=1)

    results = {
        'ensemble_accuracy': ensemble_accuracy,
        'individual_accuracies': individual_accuracies,
        'per_class_recall': per_class_recall.tolist(),
        'confusion_matrix': cm.tolist(),
        'medical_grade_achieved': ensemble_accuracy >= 0.90,
        'classification_report': classification_report(all_targets, all_predictions, output_dict=True)
    }

    return results

def main():
    print("⚡ QUICK FIXED OVO ENSEMBLE (NO CALIBRATION)")
    print("=" * 50)

    results_dir = "./ovo_ensemble_results_v2"
    dataset_path = "./dataset6"

    config_path = Path(results_dir) / "ovo_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'model': {'base_models': ['mobilenet_v2', 'inception_v3', 'densenet121']},
            'data': {'num_classes': 5, 'img_size': 224}
        }

    logger.info("🏗️ Loading quick fixed ensemble...")
    ensemble = load_quick_ensemble(results_dir, config)

    test_transform = transforms.Compose([
        transforms.Resize((config['data']['img_size'], config['data']['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = ImageFolder(f"{dataset_path}/test", transform=test_transform)

    use_cuda = torch.cuda.is_available()
    test_loader = DataLoader(
        test_dataset,
        batch_size=64 if use_cuda else 32,
        shuffle=False,
        num_workers=4 if use_cuda else 2,
        pin_memory=use_cuda
    )

    logger.info(f"📊 Test dataset: {len(test_dataset)} images")

    logger.info("⚡ Evaluating quick fixed ensemble...")
    results = evaluate_quick_ensemble(ensemble, test_loader, config)

    print("\n" + "="*50)
    print("🏆 QUICK RESULTS")
    print("="*50)
    print(f"🎯 Ensemble Accuracy: {results['ensemble_accuracy']:.4f} ({results['ensemble_accuracy']*100:.2f}%)")
    print(f"🏥 Medical Grade: {'✅ ACHIEVED' if results['medical_grade_achieved'] else '❌ NOT ACHIEVED'}")

    print(f"\n📊 Per-Class Recall (Focus on Class 1):")
    for i, recall in enumerate(results['per_class_recall']):
        status = "🎯 FIXED!" if i == 1 and recall > 0.7 else "✅" if recall > 0.8 else "⚠️" if recall > 0.6 else "❌"
        print(f"   Class {i}: {recall:.3f} ({recall*100:.1f}%) {status}")

    print(f"\n📊 Individual Models:")
    for model_name, accuracy in results['individual_accuracies'].items():
        print(f"   {model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Save results
    results_path = Path(results_dir) / "results" / "quick_results.json"
    results_path.parent.mkdir(exist_ok=True)

    with open(results_path, 'w') as f:
        json.dump({k: float(v) if isinstance(v, np.float64) else v for k, v in results.items()}, f, indent=2)

    print(f"\n💾 Results saved: {results_path}")

    # Progress tracking
    print(f"\n📈 PROGRESS:")
    print(f"   Original: ~80.9%")
    print(f"   Quick Fix: {results['ensemble_accuracy']*100:.1f}%")
    improvement = (results['ensemble_accuracy'] - 0.809) * 100
    print(f"   Improvement: +{improvement:.1f} points")

    class1_recall = results['per_class_recall'][1]
    class1_improvement = (class1_recall - 0.453) * 100
    print(f"   Class 1: 45.3% → {class1_recall*100:.1f}% (+{class1_improvement:.1f} points)")

    print("="*50)

if __name__ == "__main__":
    main()