#!/usr/bin/env python3
"""
Fix OVO Ensemble Performance Issues
Implements advanced voting strategies to bridge binary validation → ensemble test gap
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

class ImprovedOVOEnsemble(OVOEnsemble):
    """Enhanced OVO Ensemble with weighted voting and calibration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store binary classifier validation accuracies for weighting
        self.binary_accuracies = {}
        self.class_weights = None

    def load_binary_accuracies(self, results_dir):
        """Load binary classifier validation accuracies for weighted voting."""
        models_dir = Path(results_dir) / "models"

        for model_name in self.base_models:
            self.binary_accuracies[model_name] = {}
            for class_a, class_b in self.class_pairs:
                checkpoint_path = models_dir / f"best_{model_name}_{class_a}_{class_b}.pth"
                if checkpoint_path.exists():
                    try:
                        checkpoint = torch.load(checkpoint_path, map_location='cpu')
                        if isinstance(checkpoint, dict) and 'best_val_accuracy' in checkpoint:
                            accuracy = checkpoint['best_val_accuracy'] / 100.0  # Convert to decimal
                            self.binary_accuracies[model_name][f"pair_{class_a}_{class_b}"] = accuracy
                        else:
                            # Default weight for legacy checkpoints
                            self.binary_accuracies[model_name][f"pair_{class_a}_{class_b}"] = 0.85
                    except:
                        self.binary_accuracies[model_name][f"pair_{class_a}_{class_b}"] = 0.85

        print(f"✅ Loaded binary accuracies for weighted voting")

    def set_class_weights(self, class_counts):
        """Set class weights based on training data distribution."""
        total_samples = sum(class_counts)
        self.class_weights = torch.tensor([
            total_samples / (len(class_counts) * count) for count in class_counts
        ])
        print(f"✅ Class weights set: {self.class_weights}")

    def forward(self, x, return_individual=False, use_weighted_voting=True):
        """Enhanced forward pass with weighted voting."""
        batch_size = x.size(0)
        device = x.device

        # Collect weighted votes from all binary classifiers
        votes = torch.zeros(batch_size, self.num_classes, device=device)
        confidence_scores = torch.zeros(batch_size, self.num_classes, device=device)
        individual_predictions = {} if return_individual else None

        for model_name, model_classifiers in self.classifiers.items():
            model_votes = torch.zeros(batch_size, self.num_classes, device=device)
            model_confidences = torch.zeros(batch_size, self.num_classes, device=device)

            for classifier_name, classifier in model_classifiers.items():
                # Extract class indices
                class_a, class_b = map(int, classifier_name.split('_')[1:])

                # Get binary prediction and confidence
                binary_output = classifier(x).squeeze()
                if binary_output.dim() == 0:
                    binary_output = binary_output.unsqueeze(0)

                # Convert to class votes with confidence weighting
                class_a_confidence = 1.0 - binary_output  # Confidence for class A
                class_b_confidence = binary_output        # Confidence for class B

                # Apply binary classifier accuracy weighting
                if use_weighted_voting and model_name in self.binary_accuracies:
                    binary_acc = self.binary_accuracies[model_name].get(classifier_name, 0.85)
                    weight = binary_acc ** 2  # Square emphasizes better models more
                else:
                    weight = 1.0

                # Weighted voting
                model_votes[:, class_a] += class_a_confidence * weight
                model_votes[:, class_b] += class_b_confidence * weight

                # Track confidence
                model_confidences[:, class_a] += class_a_confidence
                model_confidences[:, class_b] += class_b_confidence

            # Add to ensemble votes
            votes += model_votes
            confidence_scores += model_confidences

            if return_individual:
                individual_predictions[model_name] = model_votes

        # Apply class frequency weighting if available
        if self.class_weights is not None:
            class_weights = self.class_weights.to(device)
            votes = votes * class_weights.unsqueeze(0)

        # Normalize by participation and confidence
        participation_count = self.num_classes - 1
        normalized_votes = votes / (participation_count * len(self.base_models))

        # Apply confidence scaling
        confidence_weights = torch.softmax(confidence_scores / len(self.base_models), dim=1)
        final_predictions = normalized_votes * confidence_weights

        result = {
            'logits': final_predictions,
            'votes': votes,
            'confidences': confidence_scores
        }

        if return_individual:
            result['individual_predictions'] = individual_predictions

        return result

def evaluate_improved_ensemble(ensemble_path, dataset_path, results_dir):
    """Evaluate the improved OVO ensemble."""

    print("🔧 Creating Improved OVO Ensemble...")

    # Load configuration
    config_path = Path(results_dir) / "ovo_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default config - ALL 3 architectures
        config = {
            'model': {'base_models': ['mobilenet_v2', 'inception_v3', 'densenet121']},
            'data': {'num_classes': 5, 'img_size': 299}
        }

    # Create improved ensemble
    ensemble = ImprovedOVOEnsemble(
        base_models=config['model']['base_models'],
        num_classes=config['data']['num_classes'],
        freeze_weights=True,  # Not relevant for inference
        dropout=0.3  # Not relevant for inference
    )

    # Load trained weights
    print(f"📥 Loading ensemble from: {ensemble_path}")
    state_dict = torch.load(ensemble_path, map_location='cpu')
    ensemble.load_state_dict(state_dict)

    # Load binary accuracies for weighted voting
    ensemble.load_binary_accuracies(results_dir)

    # Set class weights based on training distribution
    # Approximate class distribution (adjust based on your dataset)
    class_counts = [6440, 1401, 4163, 1401, 1400]  # From your training data
    ensemble.set_class_weights(class_counts)

    # Prepare test dataset
    print(f"📊 Loading test dataset from: {dataset_path}")
    test_transform = create_ovo_transforms(
        img_size=config['data']['img_size'],
        enable_clahe=False
    )[1]  # Use validation transforms

    test_dataset = ImageFolder(
        root=Path(dataset_path) / "test",
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"📋 Test dataset: {len(test_dataset)} images")

    # Evaluate with different voting strategies
    strategies = [
        ("Original Voting", False),
        ("Weighted Voting", True)
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ensemble = ensemble.to(device)
    ensemble.eval()

    results = {}

    for strategy_name, use_weighted in strategies:
        print(f"\n🎯 Evaluating: {strategy_name}")

        all_predictions = []
        all_targets = []
        all_confidences = []

        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device)
                targets = targets.to(device)

                # Get predictions with current strategy
                outputs = ensemble(images, return_individual=False, use_weighted_voting=use_weighted)

                # Get final predictions
                _, predictions = torch.max(outputs['logits'], 1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                # Track confidence
                max_confidences, _ = torch.max(torch.softmax(outputs['logits'], dim=1), 1)
                all_confidences.extend(max_confidences.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        avg_confidence = np.mean(all_confidences)

        results[strategy_name] = {
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'medical_grade_pass': accuracy >= 0.90,
            'predictions': all_predictions,
            'targets': all_targets
        }

        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Avg Confidence: {avg_confidence:.3f}")
        print(f"   Medical Grade: {'✅ PASS' if accuracy >= 0.90 else '❌ FAIL'}")

    # Show improvement
    original_acc = results["Original Voting"]["accuracy"]
    weighted_acc = results["Weighted Voting"]["accuracy"]
    improvement = weighted_acc - original_acc

    print(f"\n📈 IMPROVEMENT ANALYSIS:")
    print(f"   Original Ensemble: {original_acc:.4f} ({original_acc*100:.2f}%)")
    print(f"   Weighted Ensemble: {weighted_acc:.4f} ({weighted_acc*100:.2f}%)")
    print(f"   Improvement: +{improvement:.4f} ({improvement*100:.2f} percentage points)")

    if weighted_acc >= 0.90:
        print(f"   🏆 MEDICAL GRADE ACHIEVED!")
    else:
        print(f"   ⚠️ Still below medical grade (need {0.90-weighted_acc:.3f} more)")

    # Save improved results
    improved_results_path = Path(results_dir) / "results" / "improved_ovo_results.json"
    improved_results_path.parent.mkdir(exist_ok=True)

    save_results = {
        'original_accuracy': float(original_acc),
        'weighted_accuracy': float(weighted_acc),
        'improvement': float(improvement),
        'medical_grade_achieved': weighted_acc >= 0.90,
        'strategies': {
            name: {
                'accuracy': float(data['accuracy']),
                'avg_confidence': float(data['avg_confidence']),
                'medical_grade_pass': data['medical_grade_pass']
            }
            for name, data in results.items()
        }
    }

    with open(improved_results_path, 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\n💾 Improved results saved: {improved_results_path}")

    return results

if __name__ == "__main__":
    # Configuration
    ensemble_path = "./ovo_ensemble_results_v2/models/ovo_ensemble_best.pth"
    dataset_path = "./dataset6"  # or "./dataset7" for balanced dataset
    results_dir = "./ovo_ensemble_results_v2"

    # Run improved evaluation
    results = evaluate_improved_ensemble(ensemble_path, dataset_path, results_dir)