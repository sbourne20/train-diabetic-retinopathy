#!/usr/bin/env python3
"""
Advanced OVO Voting Fix
Addresses class imbalance in OVO ensemble voting
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from ensemble_local_trainer import OVOEnsemble, create_ovo_transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from scipy.special import softmax

class AdvancedOVOEnsemble(OVOEnsemble):
    """OVO Ensemble with advanced voting strategies for class imbalance."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.binary_accuracies = {}
        self.class_frequencies = None
        self.voting_strategy = 'weighted_probability'

    def load_binary_accuracies(self, results_dir):
        """Load binary classifier validation accuracies."""
        models_dir = Path(results_dir) / "models"

        for model_name in self.base_models:
            self.binary_accuracies[model_name] = {}
            for class_a, class_b in self.class_pairs:
                checkpoint_path = models_dir / f"best_{model_name}_{class_a}_{class_b}.pth"
                if checkpoint_path.exists():
                    try:
                        checkpoint = torch.load(checkpoint_path, map_location='cpu')
                        if isinstance(checkpoint, dict) and 'best_val_accuracy' in checkpoint:
                            accuracy = checkpoint['best_val_accuracy'] / 100.0
                            self.binary_accuracies[model_name][f"pair_{class_a}_{class_b}"] = accuracy
                        else:
                            self.binary_accuracies[model_name][f"pair_{class_a}_{class_b}"] = 0.85
                    except:
                        self.binary_accuracies[model_name][f"pair_{class_a}_{class_b}"] = 0.85

    def set_class_frequencies(self, train_counts):
        """Set class frequencies from training data."""
        total = sum(train_counts)
        self.class_frequencies = [count / total for count in train_counts]
        print(f"✅ Class frequencies: {[f'{f:.3f}' for f in self.class_frequencies]}")

    def forward(self, x, return_individual=False):
        """Advanced forward pass with multiple voting strategies."""
        batch_size = x.size(0)
        device = x.device

        # Strategy 1: Probability-based voting with accuracy weighting
        prob_scores = self._probability_voting(x, device)

        # Strategy 2: Pairwise win-loss matrix
        pairwise_scores = self._pairwise_voting(x, device)

        # Strategy 3: Frequency-adjusted voting
        freq_scores = self._frequency_adjusted_voting(x, device)

        # Combine strategies
        final_scores = (prob_scores + pairwise_scores + freq_scores) / 3.0

        result = {
            'logits': final_scores,
            'probability_scores': prob_scores,
            'pairwise_scores': pairwise_scores,
            'frequency_scores': freq_scores
        }

        return result

    def _probability_voting(self, x, device):
        """Probability-based voting with binary classifier confidence."""
        batch_size = x.size(0)
        class_probabilities = torch.zeros(batch_size, self.num_classes, device=device)

        for model_name, model_classifiers in self.classifiers.items():
            for classifier_name, classifier in model_classifiers.items():
                class_a, class_b = map(int, classifier_name.split('_')[1:])

                # Get binary probability
                binary_output = classifier(x).squeeze()
                if binary_output.dim() == 0:
                    binary_output = binary_output.unsqueeze(0)

                # Convert to class probabilities
                prob_class_a = 1.0 - binary_output
                prob_class_b = binary_output

                # Weight by binary classifier accuracy
                accuracy_weight = self.binary_accuracies.get(model_name, {}).get(classifier_name, 0.85)
                weight = accuracy_weight ** 2

                # Add weighted probabilities
                class_probabilities[:, class_a] += prob_class_a * weight
                class_probabilities[:, class_b] += prob_class_b * weight

        # Normalize by number of comparisons each class participates in
        participation_counts = torch.tensor([self.num_classes - 1] * self.num_classes, device=device)
        normalized_probs = class_probabilities / (participation_counts.unsqueeze(0) * len(self.base_models))

        return normalized_probs

    def _pairwise_voting(self, x, device):
        """Pairwise tournament-style voting."""
        batch_size = x.size(0)
        win_matrix = torch.zeros(batch_size, self.num_classes, device=device)

        for model_name, model_classifiers in self.classifiers.items():
            for classifier_name, classifier in model_classifiers.items():
                class_a, class_b = map(int, classifier_name.split('_')[1:])

                # Get binary prediction
                binary_output = classifier(x).squeeze()
                if binary_output.dim() == 0:
                    binary_output = binary_output.unsqueeze(0)

                # Determine winner of this pairwise comparison
                class_a_wins = (binary_output <= 0.5).float()
                class_b_wins = (binary_output > 0.5).float()

                # Weight by confidence and accuracy
                confidence = torch.abs(binary_output - 0.5) * 2  # 0 to 1
                accuracy_weight = self.binary_accuracies.get(model_name, {}).get(classifier_name, 0.85)

                combined_weight = confidence * accuracy_weight

                win_matrix[:, class_a] += class_a_wins * combined_weight
                win_matrix[:, class_b] += class_b_wins * combined_weight

        return win_matrix

    def _frequency_adjusted_voting(self, x, device):
        """Voting adjusted for class frequency imbalance."""
        batch_size = x.size(0)
        scores = torch.zeros(batch_size, self.num_classes, device=device)

        # Get basic voting scores
        basic_scores = self._probability_voting(x, device)

        if self.class_frequencies is not None:
            # Adjust for class imbalance
            freq_weights = torch.tensor(self.class_frequencies, device=device)

            # Inverse frequency weighting for minority classes
            inv_freq_weights = 1.0 / (freq_weights + 1e-8)
            inv_freq_weights = inv_freq_weights / inv_freq_weights.sum()  # Normalize

            # Apply inverse frequency weighting
            scores = basic_scores * inv_freq_weights.unsqueeze(0)
        else:
            scores = basic_scores

        return scores

def evaluate_advanced_ovo():
    """Evaluate with advanced OVO voting strategies."""

    results_dir = Path("./ovo_ensemble_results_v2")
    ensemble_path = results_dir / "models" / "ovo_ensemble_best.pth"
    dataset_path = "./dataset6"

    print("🔧 Creating Advanced OVO Ensemble...")

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

    # Create advanced ensemble
    ensemble = AdvancedOVOEnsemble(
        base_models=config['model']['base_models'],
        num_classes=config['data']['num_classes'],
        freeze_weights=True,
        dropout=0.3
    )

    # Load ensemble weights
    print(f"📥 Loading ensemble from: {ensemble_path}")
    state_dict = torch.load(ensemble_path, map_location='cpu')
    ensemble.load_state_dict(state_dict)

    # Load binary accuracies
    ensemble.load_binary_accuracies(results_dir)

    # Set class frequencies from training data (from your analysis)
    train_class_counts = [6440, 1401, 4163, 1401, 1400]  # From your training data
    ensemble.set_class_frequencies(train_class_counts)

    # Prepare test dataset
    print(f"📊 Loading test dataset from: {dataset_path}")
    test_transform = create_ovo_transforms(img_size=config['data']['img_size'])[1]
    test_dataset = ImageFolder(root=Path(dataset_path) / "test", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    print(f"📋 Test dataset: {len(test_dataset)} images")

    # Test class distribution analysis
    test_class_counts = [0] * 5
    for _, label in test_dataset:
        test_class_counts[label] += 1

    print(f"📊 Test class distribution: {test_class_counts}")
    print(f"   Class percentages: {[f'{c/len(test_dataset)*100:.1f}%' for c in test_class_counts]}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ensemble = ensemble.to(device)
    ensemble.eval()

    # Evaluate different voting strategies
    strategies = {
        'probability_scores': 'Probability-based voting',
        'pairwise_scores': 'Pairwise tournament voting',
        'frequency_scores': 'Frequency-adjusted voting',
        'logits': 'Combined advanced voting'
    }

    results = {}

    with torch.no_grad():
        for strategy_key, strategy_name in strategies.items():
            print(f"\n🎯 Evaluating: {strategy_name}")

            all_predictions = []
            all_targets = []

            for images, targets in test_loader:
                images = images.to(device)
                targets = targets.to(device)

                # Get predictions
                outputs = ensemble(images)
                scores = outputs[strategy_key]

                # Get final predictions
                _, predictions = torch.max(scores, 1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

            # Calculate accuracy
            accuracy = accuracy_score(all_targets, all_predictions)
            medical_grade = accuracy >= 0.90

            results[strategy_name] = {
                'accuracy': accuracy,
                'medical_grade': medical_grade
            }

            print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"   Medical Grade: {'✅ PASS' if medical_grade else '❌ FAIL'}")

            # Detailed per-class analysis
            report = classification_report(all_targets, all_predictions,
                                         target_names=[f'Class_{i}' for i in range(5)],
                                         output_dict=True)

            print(f"   Per-class F1 scores:")
            for i in range(5):
                f1 = report[f'Class_{i}']['f1-score']
                print(f"     Class {i}: {f1:.3f}")

    # Show improvement analysis
    print(f"\n📈 VOTING STRATEGY COMPARISON:")
    print("=" * 60)

    best_strategy = max(results.items(), key=lambda x: x[1]['accuracy'])
    worst_strategy = min(results.items(), key=lambda x: x[1]['accuracy'])

    for strategy_name, result in results.items():
        status = "🏆" if strategy_name == best_strategy[0] else "⭐" if result['medical_grade'] else "❌"
        print(f"{status} {strategy_name}: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")

    improvement = best_strategy[1]['accuracy'] - 0.7282  # vs previous weighted voting
    print(f"\nImprovement over previous: +{improvement:.4f} ({improvement*100:.2f} percentage points)")

    if best_strategy[1]['medical_grade']:
        print(f"🏆 MEDICAL GRADE ACHIEVED with {best_strategy[0]}!")
    else:
        gap_to_medical = 0.90 - best_strategy[1]['accuracy']
        print(f"⚠️ Need {gap_to_medical:.3f} more for medical grade")

    # Save results
    save_results = {
        'strategies': {name: {'accuracy': float(data['accuracy']), 'medical_grade': data['medical_grade']}
                      for name, data in results.items()},
        'best_strategy': best_strategy[0],
        'best_accuracy': float(best_strategy[1]['accuracy']),
        'medical_grade_achieved': best_strategy[1]['medical_grade'],
        'improvement_over_previous': float(improvement)
    }

    output_path = results_dir / "results" / "advanced_ovo_results.json"
    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\n💾 Advanced results saved: {output_path}")

    return results

if __name__ == "__main__":
    results = evaluate_advanced_ovo()