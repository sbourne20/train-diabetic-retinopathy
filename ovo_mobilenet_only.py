#!/usr/bin/env python3
"""
MobileNet-Only OVO Fix - Use Only Working Model

Since Inception-v3 (24.91%) and DenseNet121 (30.08%) are both broken,
let's test with only MobileNet-v2 (74.11%) which is working correctly.
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
    """Binary classifier - MobileNet-v2 only"""

    def __init__(self, model_name='mobilenet_v2', freeze_weights=True, dropout=0.5):
        super().__init__()
        self.model_name = model_name

        if model_name == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=True)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Only MobileNet-v2 supported in this debug version")

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
        features = self.backbone(x)
        if isinstance(features, tuple):
            features = features[0]
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        return self.classifier(features)

class MobileNetOnlyOVOEnsemble(nn.Module):
    """MobileNet-Only OVO Ensemble"""

    def __init__(self, base_models=['mobilenet_v2'],  # ONLY MobileNet
                 num_classes=5, freeze_weights=True, dropout=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.base_models = base_models
        self.class_pairs = list(combinations(range(num_classes), 2))

        # Create binary classifiers (only MobileNet)
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

        # MobileNet-v2 accuracies from diagnostic analysis
        self.binary_accuracies = {
            'mobilenet_v2': {
                'pair_0_1': 0.8745, 'pair_0_2': 0.8228, 'pair_0_3': 0.9827, 'pair_0_4': 0.9845,
                'pair_1_2': 0.8072, 'pair_1_3': 0.8667, 'pair_1_4': 0.9251, 'pair_2_3': 0.8567,
                'pair_2_4': 0.8836, 'pair_3_4': 0.7205
            }
        }

        # Only one weak classifier for MobileNet
        self.weak_classifiers = {('mobilenet_v2', 'pair_3_4')}  # 72.05%

        # Strong Class 1 boost (single model needs more help)
        self.class_weights = torch.tensor([1.0, 8.0, 2.0, 3.0, 4.0])
        self.class1_pairs = ['pair_0_1', 'pair_1_2', 'pair_1_3', 'pair_1_4']

        logger.info(f"📱 MobileNet-Only OVO Ensemble")

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

                # Weak classifier penalty (only pair_3_4)
                if (model_name, classifier_name) in self.weak_classifiers:
                    accuracy_weight = (base_accuracy ** 3) * 0.7  # Penalty for 72% classifier
                    logger.debug(f"Weak classifier penalty: {classifier_name}")
                else:
                    accuracy_weight = base_accuracy ** 2

                # Excellent classifier boost (>95%)
                if base_accuracy > 0.95:
                    accuracy_weight *= 1.8  # Strong boost for excellent classifiers

                # Confidence calculation
                confidence = torch.abs(binary_output - 0.5) * 2
                weighted_confidence = confidence * accuracy_weight

                # Class weights with aggressive Class 1 boost
                class_a_weight = class_weights[class_a]
                class_b_weight = class_weights[class_b]

                # Emergency Class 1 boost (single model needs maximum help)
                if classifier_name in self.class1_pairs:
                    if class_a == 1:
                        class_a_weight *= 4.0  # Maximum boost
                    if class_b == 1:
                        class_b_weight *= 4.0  # Maximum boost

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
                model_weights = total_weights
                individual_predictions[model_name] = individual_predictions[model_name] / (model_weights + 1e-8)
            result['individual_predictions'] = individual_predictions

        return result

def load_mobilenet_ensemble(results_dir, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🎮 Using device: {device}")

    ensemble = MobileNetOnlyOVOEnsemble(
        base_models=['mobilenet_v2'],
        num_classes=config['data']['num_classes'],
        freeze_weights=True,
        dropout=0.5
    )

    models_dir = Path(results_dir) / "models"
    loaded_count = 0
    failed_models = []

    # Only load MobileNet models
    for class_a, class_b in combinations(range(config['data']['num_classes']), 2):
        pair_name = f"pair_{class_a}_{class_b}"
        model_path = models_dir / f"best_mobilenet_v2_{class_a}_{class_b}.pth"

        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)

                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    ensemble.classifiers['mobilenet_v2'][pair_name].load_state_dict(checkpoint['model_state_dict'])
                else:
                    ensemble.classifiers['mobilenet_v2'][pair_name].load_state_dict(checkpoint)

                loaded_count += 1
                logger.info(f"✅ Loaded MobileNet {pair_name}")
            except Exception as e:
                logger.error(f"❌ Failed to load {model_path.name}: {e}")
                failed_models.append(f"mobilenet_v2_{pair_name}")
        else:
            logger.warning(f"⚠️ Missing: {model_path.name}")

    logger.info(f"📦 Loaded {loaded_count}/10 MobileNet binary classifiers")
    if failed_models:
        logger.warning(f"Failed models: {failed_models}")

    return ensemble

def evaluate_mobilenet_ensemble(ensemble, test_loader, config):
    logger.info("📋 Evaluating MobileNet-Only Ensemble")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ensemble = ensemble.to(device)
    ensemble.eval()

    all_predictions = []
    all_targets = []
    individual_predictions = {'mobilenet_v2': []}

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
    print("📱 MOBILENET-ONLY OVO ENSEMBLE")
    print("=" * 50)

    results_dir = "./ovo_ensemble_results_v2"
    dataset_path = "./dataset6"

    config = {
        'model': {'base_models': ['mobilenet_v2']},
        'data': {'num_classes': 5, 'img_size': 224}
    }

    logger.info("🏗️ Loading MobileNet-only ensemble...")
    ensemble = load_mobilenet_ensemble(results_dir, config)

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

    logger.info("📱 Evaluating MobileNet-only ensemble...")
    results = evaluate_mobilenet_ensemble(ensemble, test_loader, config)

    print("\n" + "="*50)
    print("🏆 MOBILENET-ONLY RESULTS")
    print("="*50)
    print(f"🎯 Ensemble Accuracy: {results['ensemble_accuracy']:.4f} ({results['ensemble_accuracy']*100:.2f}%)")
    print(f"🏥 Medical Grade: {'✅ ACHIEVED' if results['medical_grade_achieved'] else '❌ NOT ACHIEVED'}")

    print(f"\n📊 Per-Class Recall:")
    for i, recall in enumerate(results['per_class_recall']):
        status = "🎯 GOOD!" if i == 1 and recall > 0.7 else "✅" if recall > 0.8 else "⚠️" if recall > 0.6 else "❌"
        print(f"   Class {i}: {recall:.3f} ({recall*100:.1f}%) {status}")

    print(f"\n📊 MobileNet-v2 Performance:")
    for model_name, accuracy in results['individual_accuracies'].items():
        print(f"   {model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Save results
    results_path = Path(results_dir) / "results" / "mobilenet_only_results.json"
    results_path.parent.mkdir(exist_ok=True)

    with open(results_path, 'w') as f:
        json.dump({k: float(v) if isinstance(v, np.float64) else v for k, v in results.items()}, f, indent=2)

    print(f"\n💾 Results saved: {results_path}")

    # Comprehensive comparison
    print(f"\n📈 PERFORMANCE PROGRESSION:")
    print(f"   Original (3 models, broken voting): ~80.9%")
    print(f"   Fixed voting (3 models, 2 broken): 76.6%")
    print(f"   Debug (2 models, 1 broken): 78.0%")
    print(f"   MobileNet-only (1 model, working): {results['ensemble_accuracy']*100:.1f}%")

    print(f"\n🔍 FINAL DIAGNOSIS:")
    if results['ensemble_accuracy'] > 0.82:
        print("   ✅ MobileNet works well! Other models are broken.")
        print("   💡 Strategy: Fix Inception-v3 & DenseNet121 OR retrain with dataset7b")
    elif results['ensemble_accuracy'] > 0.75:
        print("   ⚠️ MobileNet decent but not optimal")
        print("   💡 Strategy: Definitely retrain all models with dataset7b")
    else:
        print("   ❌ Even MobileNet underperforming")
        print("   💡 Strategy: Complete retrain with dataset7b essential")

    print(f"\n🎯 CLASS 1 TRACKING:")
    class1_recall = results['per_class_recall'][1]
    print(f"   Original: 45.3% → Current: {class1_recall*100:.1f}%")
    print(f"   Improvement: {(class1_recall - 0.453)*100:+.1f} percentage points")

    print("="*50)

if __name__ == "__main__":
    main()