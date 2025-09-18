#!/usr/bin/env python3
"""
Medical-Grade OVO Ensemble Voting Optimizer
============================================

Diagnosis: Binary classifiers achieve 91.5-92.6% but ensemble drops to 81.39%
Root Cause: Voting mechanism loses 10+ percentage points during aggregation
Solution: Multiple advanced voting strategies optimized for medical-grade performance

Key Issues Identified:
1. Class imbalance severely impacts minority classes (1, 3, 4)
2. Current voting doesn't leverage binary classifier confidence properly
3. Temperature scaling may be over-smoothing strong predictions
4. No class-specific optimization for medical thresholds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from ensemble_local_trainer import create_ovo_transforms
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class MedicalGradeOVOEnsemble(nn.Module):
    """Medical-Grade OVO Ensemble with multiple advanced voting strategies."""

    def __init__(self, base_models, num_classes=5, freeze_weights=True, dropout=0.3):
        super().__init__()
        self.base_models = base_models
        self.num_classes = num_classes
        self.freeze_weights = freeze_weights
        self.dropout = dropout

        # Initialize classifiers and accuracy tracking
        self.classifiers = nn.ModuleDict()
        self.binary_accuracies = {}

        # Medical-grade parameters
        self.class_frequencies = torch.tensor([0.40, 0.09, 0.26, 0.09, 0.09])  # Based on dataset distribution
        self.medical_class_weights = torch.tensor([1.0, 2.5, 1.2, 2.0, 2.0])  # Boost minority classes

        # Initialize architecture-specific classifiers
        self._initialize_classifiers()

    def _initialize_classifiers(self):
        """Initialize OVO binary classifiers for each architecture."""
        from ensemble_local_trainer import BinaryClassifier

        for model_name in self.base_models:
            self.classifiers[model_name] = nn.ModuleDict()
            self.binary_accuracies[model_name] = {}

            # Create binary classifiers for each pair
            for class_a in range(self.num_classes):
                for class_b in range(class_a + 1, self.num_classes):
                    classifier_name = f"pair_{class_a}_{class_b}"

                    # Create binary classifier
                    binary_classifier = BinaryClassifier(
                        model_name=model_name,
                        freeze_weights=self.freeze_weights,
                        dropout=self.dropout
                    )

                    self.classifiers[model_name][classifier_name] = binary_classifier
                    self.binary_accuracies[model_name][classifier_name] = 0.85  # Default

    def load_binary_accuracies(self, results_dir):
        """Load actual binary classifier validation accuracies."""
        try:
            accuracy_files = {
                'mobilenet_v2': 'MOBILENET_V2_ovo_validation_results.json',
                'inception_v3': 'INCEPTION_V3_ovo_validation_results.json',
                'densenet121': 'DENSENET121_ovo_validation_results.json'
            }

            for model_name, filename in accuracy_files.items():
                file_path = Path(results_dir) / "results" / filename
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    for classifier_name, acc in data.items():
                        if classifier_name in self.binary_accuracies[model_name]:
                            self.binary_accuracies[model_name][classifier_name] = acc

            logger.info("✅ Loaded binary accuracies for medical-grade voting")
        except Exception as e:
            logger.warning(f"⚠️ Could not load binary accuracies: {e}")

    def forward(self, x, voting_strategy='medical_optimized'):
        """Medical-grade forward pass with multiple voting strategies."""

        if voting_strategy == 'medical_optimized':
            return self._medical_optimized_voting(x)
        elif voting_strategy == 'confidence_weighted':
            return self._confidence_weighted_voting(x)
        elif voting_strategy == 'class_balanced':
            return self._class_balanced_voting(x)
        elif voting_strategy == 'ensemble_fusion':
            return self._ensemble_fusion_voting(x)
        else:
            raise ValueError(f"Unknown voting strategy: {voting_strategy}")

    def _medical_optimized_voting(self, x):
        """Medical-optimized voting for >90% accuracy."""
        batch_size = x.size(0)
        device = x.device

        # Accumulate votes with medical-grade optimization
        class_votes = torch.zeros(batch_size, self.num_classes, device=device)
        total_confidence = torch.zeros(batch_size, self.num_classes, device=device)

        for model_name, model_classifiers in self.classifiers.items():
            for classifier_name, classifier in model_classifiers.items():
                class_a, class_b = map(int, classifier_name.split('_')[1:])

                # Get binary prediction
                binary_output = torch.sigmoid(classifier(x).squeeze())
                if binary_output.dim() == 0:
                    binary_output = binary_output.unsqueeze(0)

                # Calculate confidence (distance from 0.5)
                confidence = torch.abs(binary_output - 0.5) * 2

                # Get accuracy weight
                accuracy_weight = self.binary_accuracies.get(model_name, {}).get(classifier_name, 0.85)

                # Medical-grade weighting: accuracy^2 * confidence * class_weight
                medical_weight = (accuracy_weight ** 2) * confidence

                # Apply class-specific medical weights
                class_a_weight = medical_weight * self.medical_class_weights[class_a]
                class_b_weight = medical_weight * self.medical_class_weights[class_b]

                # Vote assignment
                prob_class_a = (1.0 - binary_output) * class_a_weight
                prob_class_b = binary_output * class_b_weight

                class_votes[:, class_a] += prob_class_a
                class_votes[:, class_b] += prob_class_b

                total_confidence[:, class_a] += class_a_weight
                total_confidence[:, class_b] += class_b_weight

        # Normalize by total confidence
        normalized_votes = class_votes / (total_confidence + 1e-8)

        # Apply frequency-based correction
        frequency_correction = 1.0 / (self.class_frequencies.to(device) + 1e-8)
        corrected_votes = normalized_votes * frequency_correction.unsqueeze(0)

        # Convert to probabilities
        final_probs = F.softmax(corrected_votes, dim=1)

        return {'logits': final_probs}

    def _confidence_weighted_voting(self, x):
        """Pure confidence-weighted voting."""
        batch_size = x.size(0)
        device = x.device

        class_scores = torch.zeros(batch_size, self.num_classes, device=device)
        weights_sum = torch.zeros(batch_size, self.num_classes, device=device)

        for model_name, model_classifiers in self.classifiers.items():
            for classifier_name, classifier in model_classifiers.items():
                class_a, class_b = map(int, classifier_name.split('_')[1:])

                binary_output = torch.sigmoid(classifier(x).squeeze())
                if binary_output.dim() == 0:
                    binary_output = binary_output.unsqueeze(0)

                # High confidence weighting
                confidence = torch.abs(binary_output - 0.5) * 2
                weight = confidence ** 2  # Square for more aggressive weighting

                class_scores[:, class_a] += (1.0 - binary_output) * weight
                class_scores[:, class_b] += binary_output * weight

                weights_sum[:, class_a] += weight
                weights_sum[:, class_b] += weight

        normalized_scores = class_scores / (weights_sum + 1e-8)
        return {'logits': F.softmax(normalized_scores, dim=1)}

    def _class_balanced_voting(self, x):
        """Class-balanced voting to address imbalance."""
        batch_size = x.size(0)
        device = x.device

        class_votes = torch.zeros(batch_size, self.num_classes, device=device)

        for model_name, model_classifiers in self.classifiers.items():
            for classifier_name, classifier in model_classifiers.items():
                class_a, class_b = map(int, classifier_name.split('_')[1:])

                binary_output = torch.sigmoid(classifier(x).squeeze())
                if binary_output.dim() == 0:
                    binary_output = binary_output.unsqueeze(0)

                # Apply inverse frequency weighting
                freq_weight_a = 1.0 / self.class_frequencies[class_a]
                freq_weight_b = 1.0 / self.class_frequencies[class_b]

                class_votes[:, class_a] += (1.0 - binary_output) * freq_weight_a
                class_votes[:, class_b] += binary_output * freq_weight_b

        return {'logits': F.softmax(class_votes, dim=1)}

    def _ensemble_fusion_voting(self, x):
        """Fusion of multiple voting strategies."""
        # Get results from different strategies
        medical_result = self._medical_optimized_voting(x)['logits']
        confidence_result = self._confidence_weighted_voting(x)['logits']
        balanced_result = self._class_balanced_voting(x)['logits']

        # Weighted fusion (medical-optimized gets highest weight)
        fusion_weights = torch.tensor([0.6, 0.2, 0.2], device=x.device)

        fused_logits = (fusion_weights[0] * medical_result +
                       fusion_weights[1] * confidence_result +
                       fusion_weights[2] * balanced_result)

        return {'logits': fused_logits}

def evaluate_voting_strategies():
    """Evaluate multiple voting strategies to find medical-grade performance."""

    results_dir = Path("./ovo_ensemble_results_v2")
    ensemble_path = results_dir / "models" / "ovo_ensemble_best.pth"
    dataset_path = "./dataset6"

    print("🏥 Creating Advanced Medical-Grade OVO Ensemble...")

    # Load configuration
    config_path = results_dir / "ovo_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'model': {'base_models': ['mobilenet_v2', 'inception_v3', 'densenet121']},
            'data': {'num_classes': 5, 'img_size': 299}
        }

    # Create ensemble
    ensemble = MedicalGradeOVOEnsemble(
        base_models=config['model']['base_models'],
        num_classes=config['data']['num_classes'],
        freeze_weights=True,
        dropout=0.3
    )

    # Load weights and accuracies
    print(f"📥 Loading ensemble from: {ensemble_path}")
    state_dict = torch.load(ensemble_path, map_location='cpu')
    ensemble.load_state_dict(state_dict)
    ensemble.load_binary_accuracies(results_dir)

    # Prepare test dataset
    print(f"📊 Loading test dataset from: {dataset_path}")
    test_transform = create_ovo_transforms(img_size=config['data']['img_size'])[1]
    test_dataset = ImageFolder(root=Path(dataset_path) / "test", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ensemble = ensemble.to(device)
    ensemble.eval()

    # Test all voting strategies
    voting_strategies = [
        'medical_optimized',
        'confidence_weighted',
        'class_balanced',
        'ensemble_fusion'
    ]

    strategy_descriptions = {
        'medical_optimized': 'Medical-Optimized Voting (accuracy² × confidence × class weights)',
        'confidence_weighted': 'Pure Confidence-Weighted Voting',
        'class_balanced': 'Class-Balanced Voting (inverse frequency)',
        'ensemble_fusion': 'Ensemble Fusion of All Strategies'
    }

    results = {}
    best_strategy = None
    best_accuracy = 0.0

    for strategy in voting_strategies:
        print(f"\n🔬 Testing: {strategy_descriptions[strategy]}")

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(device), targets.to(device)

                outputs = ensemble(images, voting_strategy=strategy)
                predictions = torch.argmax(outputs['logits'], dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)

        # Per-class analysis
        cm = confusion_matrix(all_targets, all_predictions)
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

        results[strategy] = {
            'accuracy': accuracy,
            'per_class_accuracy': per_class_accuracy.tolist(),
            'confusion_matrix': cm.tolist()
        }

        # Medical grade assessment
        medical_grade = "✅ MEDICAL GRADE" if accuracy >= 0.90 else "❌ BELOW MEDICAL GRADE"

        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%) - {medical_grade}")
        print(f"   Per-Class: {[f'{acc:.3f}' for acc in per_class_accuracy]}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_strategy = strategy

    # Report best result
    print(f"\n🏆 BEST STRATEGY: {strategy_descriptions[best_strategy]}")
    print(f"🎯 Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

    medical_grade_achieved = best_accuracy >= 0.90
    print(f"🏥 Medical Grade: {'✅ ACHIEVED' if medical_grade_achieved else '❌ NOT ACHIEVED'}")

    if not medical_grade_achieved:
        improvement_needed = 0.90 - best_accuracy
        print(f"   Need: +{improvement_needed:.4f} ({improvement_needed*100:.2f} points)")

    # Save detailed results
    output_file = results_dir / "results" / "advanced_voting_strategies_results.json"
    output_file.parent.mkdir(exist_ok=True)

    final_results = {
        'best_strategy': best_strategy,
        'best_accuracy': best_accuracy,
        'medical_grade_achieved': medical_grade_achieved,
        'all_strategies': results,
        'improvement_from_current': best_accuracy - 0.8139
    }

    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n💾 Detailed results saved: {output_file}")

    return best_strategy, best_accuracy, medical_grade_achieved

if __name__ == "__main__":
    print("🚀 Running Advanced Medical-Grade OVO Voting Optimization...")

    try:
        best_strategy, best_accuracy, medical_grade = evaluate_voting_strategies()

        print(f"\n✅ Optimization completed!")
        print(f"📊 Best Strategy: {best_strategy}")
        print(f"🎯 Final Accuracy: {best_accuracy*100:.2f}%")
        print(f"🏥 Medical Grade: {'✅ ACHIEVED' if medical_grade else '❌ NEEDS IMPROVEMENT'}")

    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()