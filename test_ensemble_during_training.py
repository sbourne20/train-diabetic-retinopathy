#!/usr/bin/env python3
"""
Real-Time Ensemble Validation During Training

This script tests ensemble performance while training is running
to catch voting mechanism issues early.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from itertools import combinations
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from the training script
sys.path.append('.')
from ensemble_local_trainer import BinaryClassifier, OVOEnsemble

def load_available_models(models_dir, base_models=['mobilenet_v2'], num_classes=5):
    """Load currently available trained models"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_pairs = list(combinations(range(num_classes), 2))

    # Create ensemble structure
    available_classifiers = {}
    loaded_models = {}

    for model_name in base_models:
        available_classifiers[model_name] = {}
        loaded_models[model_name] = {}

        for class_a, class_b in class_pairs:
            pair_name = f"pair_{class_a}_{class_b}"
            model_path = models_dir / f"best_{model_name}_{class_a}_{class_b}.pth"

            if model_path.exists():
                try:
                    # Create model with same architecture as training
                    binary_model = BinaryClassifier(
                        model_name=model_name,
                        freeze_weights=False,  # Match training config
                        dropout=0.3
                    )

                    # Load checkpoint
                    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        binary_model.load_state_dict(checkpoint['model_state_dict'])
                        val_acc = checkpoint.get('best_val_accuracy', 0.0)
                        logger.info(f"✅ Loaded {pair_name}: {val_acc:.1f}%")
                    else:
                        binary_model.load_state_dict(checkpoint)
                        logger.info(f"✅ Loaded {pair_name} (legacy format)")

                    binary_model = binary_model.to(device)
                    binary_model.eval()

                    available_classifiers[model_name][pair_name] = binary_model
                    loaded_models[model_name][pair_name] = True

                except Exception as e:
                    logger.error(f"❌ Failed to load {pair_name}: {e}")
                    loaded_models[model_name][pair_name] = False
            else:
                loaded_models[model_name][pair_name] = False

    return available_classifiers, loaded_models

def test_individual_vs_ensemble(available_classifiers, test_loader, num_classes=5):
    """Test individual model performance vs ensemble performance"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_pairs = list(combinations(range(num_classes), 2))

    # Test individual binary classifiers
    individual_results = {}

    for model_name, model_classifiers in available_classifiers.items():
        individual_results[model_name] = {}

        for pair_name, classifier in model_classifiers.items():
            class_a, class_b = map(int, pair_name.split('_')[1:])

            # Create binary test dataset
            binary_predictions = []
            binary_targets = []

            with torch.no_grad():
                for images, targets in test_loader:
                    images = images.to(device)
                    targets = targets.to(device)

                    # Filter for this binary pair
                    mask = (targets == class_a) | (targets == class_b)
                    if not mask.any():
                        continue

                    filtered_images = images[mask]
                    filtered_targets = targets[mask]

                    # Convert to binary labels
                    binary_labels = (filtered_targets == class_b).float()

                    # Get predictions
                    outputs = classifier(filtered_images).squeeze()
                    if outputs.dim() == 0:
                        outputs = outputs.unsqueeze(0)

                    predictions = (outputs > 0.5).float()

                    binary_predictions.extend(predictions.cpu().numpy())
                    binary_targets.extend(binary_labels.cpu().numpy())

            if len(binary_predictions) > 0:
                binary_acc = accuracy_score(binary_targets, binary_predictions)
                individual_results[model_name][pair_name] = binary_acc
                logger.info(f"   {pair_name}: {binary_acc:.3f} ({len(binary_predictions)} samples)")

    # Test ensemble performance using our enhanced voting
    logger.info("\n🔗 Testing Enhanced Ensemble Performance:")

    all_predictions = []
    all_targets = []

    # Medical-grade class weights
    class_weights = torch.tensor([1.0, 8.0, 2.0, 4.0, 5.0], device=device)
    class1_pairs = ['pair_0_1', 'pair_1_2', 'pair_1_3', 'pair_1_4']

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)

            # Enhanced voting mechanism
            class_scores = torch.zeros(batch_size, num_classes, device=device)
            total_weights = torch.zeros(batch_size, num_classes, device=device)

            for model_name, model_classifiers in available_classifiers.items():
                for pair_name, classifier in model_classifiers.items():
                    class_a, class_b = map(int, pair_name.split('_')[1:])

                    # Get binary prediction
                    binary_output = classifier(images).squeeze()
                    if binary_output.dim() == 0:
                        binary_output = binary_output.unsqueeze(0)
                    if binary_output.size(0) != batch_size:
                        # Handle batch size mismatch
                        if binary_output.size(0) == 1 and batch_size > 1:
                            binary_output = binary_output.expand(batch_size)

                    # Enhanced weighting
                    base_accuracy = individual_results[model_name].get(pair_name, 0.8)
                    accuracy_weight = base_accuracy ** 2

                    if base_accuracy > 0.95:
                        accuracy_weight *= 1.5
                    elif base_accuracy < 0.8:
                        accuracy_weight *= 0.5

                    # Confidence weighting
                    confidence = torch.abs(binary_output - 0.5) * 2
                    weighted_confidence = confidence * accuracy_weight

                    # Class weights
                    class_a_weight = class_weights[class_a]
                    class_b_weight = class_weights[class_b]

                    # Class 1 emergency boost
                    if pair_name in class1_pairs:
                        if class_a == 1:
                            class_a_weight *= 3.0
                        if class_b == 1:
                            class_b_weight *= 3.0

                    # Probability voting
                    prob_class_a = (1.0 - binary_output) * class_a_weight * weighted_confidence
                    prob_class_b = binary_output * class_b_weight * weighted_confidence

                    class_scores[:, class_a] += prob_class_a
                    class_scores[:, class_b] += prob_class_b

                    total_weights[:, class_a] += class_a_weight * weighted_confidence
                    total_weights[:, class_b] += class_b_weight * weighted_confidence

            # Final ensemble prediction
            normalized_scores = class_scores / (total_weights + 1e-8)
            final_predictions = F.softmax(normalized_scores, dim=1)

            _, ensemble_pred = torch.max(final_predictions, 1)
            all_predictions.extend(ensemble_pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    if len(all_predictions) > 0:
        ensemble_accuracy = accuracy_score(all_targets, all_predictions)

        # Per-class analysis
        cm = classification_report(all_targets, all_predictions, output_dict=True)

        return individual_results, ensemble_accuracy, cm
    else:
        return individual_results, 0.0, {}

def main():
    """Main function to test ensemble during training"""

    print("🔄 REAL-TIME ENSEMBLE VALIDATION")
    print("=" * 50)

    # Check for available models (try multiple directories)
    models_dir = Path("./ovo_ensemble_results_v3/models")

    if not models_dir.exists():
        models_dir = Path("./ovo_ensemble_results_v2/models")

    if not models_dir.exists():
        models_dir = Path("./ovo_ensemble_results/models")

    if not models_dir.exists():
        print("❌ No models directory found. Training hasn't started yet.")
        return

    print(f"🔍 Checking models in: {models_dir}")

    # Count available models like analyze_ovo_with_metrics.py
    binary_models = []
    for model_file in models_dir.glob('best_*.pth'):
        if model_file.name != 'ovo_ensemble_best.pth':
            binary_models.append(model_file)

    print(f"📋 Found {len(binary_models)} binary classifiers")

    if len(binary_models) == 0:
        print("⏳ No trained models found yet. Download models and try again.")
        return

    # Load test dataset
    test_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = ImageFolder("./dataset7b/test", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    print(f"📊 Test dataset: {len(test_dataset)} images")

    # Test each model type separately
    for model_name in ['mobilenet_v2', 'inception_v3', 'densenet121']:
        print(f"\n📱 Testing {model_name} models:")

        # Load available models for this architecture
        available_classifiers, loaded_models = load_available_models(
            models_dir, base_models=[model_name], num_classes=5
        )

        if not any(loaded_models[model_name].values()):
            print(f"   ⏳ No {model_name} models trained yet")
            continue

        # Count loaded models
        loaded_count = sum(loaded_models[model_name].values())
        print(f"   📦 Loaded {loaded_count}/10 binary classifiers")

        if loaded_count >= 5:  # Need minimum for ensemble test
            print(f"   🧪 Testing individual vs ensemble performance:")

            try:
                individual_results, ensemble_acc, classification_report_dict = test_individual_vs_ensemble(
                    available_classifiers, test_loader
                )

                # Calculate average individual performance
                all_individual_accs = []
                for pair_results in individual_results[model_name].values():
                    all_individual_accs.append(pair_results)

                if all_individual_accs:
                    avg_individual = np.mean(all_individual_accs)

                    print(f"   📊 Results:")
                    print(f"      Individual Binary Avg: {avg_individual:.3f} ({avg_individual*100:.1f}%)")
                    print(f"      Ensemble Performance:  {ensemble_acc:.3f} ({ensemble_acc*100:.1f}%)")

                    # Performance gap analysis
                    gap = avg_individual - ensemble_acc
                    if gap > 0.1:
                        print(f"      ⚠️ Large gap: {gap:.3f} ({gap*100:.1f} points) - Voting issue!")
                    elif gap > 0.05:
                        print(f"      ⚠️ Moderate gap: {gap:.3f} ({gap*100:.1f} points)")
                    else:
                        print(f"      ✅ Good alignment: {gap:.3f} ({gap*100:.1f} points)")

                    # Class-specific performance
                    if '1' in classification_report_dict:
                        class1_recall = classification_report_dict['1']['recall']
                        print(f"      🎯 Class 1 Recall: {class1_recall:.3f} ({class1_recall*100:.1f}%)")

            except Exception as e:
                print(f"   ❌ Ensemble test failed: {e}")
        else:
            print(f"   ⏳ Need more models for ensemble test (have {loaded_count}, need ≥5)")

if __name__ == "__main__":
    main()