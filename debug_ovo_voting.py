#!/usr/bin/env python3
"""
Debug OVO voting mechanism to understand why 99% binary accuracy → 73% ensemble accuracy
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import sys

sys.path.insert(0, str(Path(__file__).parent))
from ensemble_5class_trainer import BinaryClassifier, CLAHETransform, OVOEnsemble

def analyze_sample_predictions(ovo_ensemble, test_loader, num_samples=10):
    """Analyze individual sample predictions to debug voting."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ovo_ensemble = ovo_ensemble.to(device)
    ovo_ensemble.eval()

    print("\n" + "="*100)
    print("🔍 DETAILED SAMPLE-BY-SAMPLE ANALYSIS")
    print("="*100)

    samples_analyzed = 0
    misclassifications = []

    with torch.no_grad():
        for images, targets in test_loader:
            if samples_analyzed >= num_samples:
                break

            images = images.to(device)
            targets = targets.to(device)

            # Get ensemble predictions with individual pair outputs
            outputs = ovo_ensemble(images, return_individual=True)

            # Get final prediction
            _, ensemble_pred = torch.max(outputs['logits'], 1)

            # Analyze each sample in batch
            for i in range(images.size(0)):
                if samples_analyzed >= num_samples:
                    break

                true_label = targets[i].item()
                pred_label = ensemble_pred[i].item()
                vote_counts = outputs['logits'][i].cpu().numpy()

                print(f"\n{'='*100}")
                print(f"Sample {samples_analyzed + 1}: True Label = {true_label}, Predicted = {pred_label}")
                print(f"Vote Counts: {vote_counts}")
                print(f"Status: {'✅ CORRECT' if true_label == pred_label else '❌ WRONG'}")

                # Show individual binary classifier outputs
                print("\nBinary Classifier Predictions:")
                print(f"{'Pair':<10} {'Pred':<8} {'Expected':<12} {'Status':<10}")
                print("-" * 50)

                # Get binary predictions for this sample
                for pair_idx, (class_a, class_b) in enumerate(ovo_ensemble.class_pairs):
                    pair_name = f"pair_{class_a}_{class_b}"

                    # Get the classifier
                    model_name = list(ovo_ensemble.classifiers.keys())[0]
                    classifier = ovo_ensemble.classifiers[model_name][pair_name]

                    # Get prediction
                    binary_logits = classifier(images[i:i+1]).squeeze()
                    binary_output = torch.sigmoid(binary_logits).item()

                    # Determine winner
                    predicted_class = class_b if binary_output > 0.5 else class_a

                    # Expected prediction
                    if true_label == class_a:
                        expected = class_a
                    elif true_label == class_b:
                        expected = class_b
                    else:
                        expected = "N/A"

                    # Status
                    if expected == "N/A":
                        status = "⚪ N/A"
                    elif predicted_class == expected:
                        status = "✅ Correct"
                    else:
                        status = "❌ Wrong"

                    print(f"{class_a}_vs_{class_b:<5} {predicted_class:<8} {expected!s:<12} {status:<10} (prob={binary_output:.3f})")

                # Calculate how many binary classifiers got it right
                correct_pairs = 0
                total_relevant_pairs = 0

                for class_a, class_b in ovo_ensemble.class_pairs:
                    if true_label in [class_a, class_b]:
                        total_relevant_pairs += 1
                        # Check if pair predicted correctly
                        # (This would require re-running, simplified here)

                print(f"\nVote Distribution:")
                for class_idx in range(5):
                    votes = vote_counts[class_idx]
                    marker = "👑" if class_idx == pred_label else "  "
                    correct = "✅" if class_idx == true_label else "  "
                    print(f"  {marker} Class {class_idx}: {votes:.1f} votes {correct}")

                if true_label != pred_label:
                    misclassifications.append({
                        'true': true_label,
                        'pred': pred_label,
                        'votes': vote_counts.copy()
                    })

                samples_analyzed += 1

    # Summary
    print("\n" + "="*100)
    print("📊 MISCLASSIFICATION SUMMARY")
    print("="*100)

    if misclassifications:
        for idx, mis in enumerate(misclassifications):
            print(f"\nMisclassification {idx+1}:")
            print(f"  True: Class {mis['true']}, Predicted: Class {mis['pred']}")
            print(f"  Votes: {mis['votes']}")
            print(f"  Vote difference: {mis['votes'][mis['pred']] - mis['votes'][mis['true']]:.1f}")
    else:
        print("No misclassifications in analyzed samples!")

def main():
    # Setup test dataset
    transform = transforms.Compose([
        CLAHETransform(clip_limit=3.0),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(
        './dataset_eyepacs_5class_balanced_enhanced_v2/test',
        transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    print("🔍 Loading OVO Ensemble for Debugging")

    # Load ensemble
    ensemble_path = Path('./coatnet_5class_results/models/ovo_ensemble_best.pth')

    if not ensemble_path.exists():
        print(f"❌ Ensemble not found: {ensemble_path}")
        return

    # Create ensemble
    ovo_ensemble = OVOEnsemble(
        base_models=['coatnet_0_rw_224'],
        num_classes=5,
        freeze_weights=True,
        dropout=0.28
    )

    # Load state
    state_dict = torch.load(ensemble_path, map_location='cpu', weights_only=False)
    ovo_ensemble.load_state_dict(state_dict)

    print("✅ Ensemble loaded successfully")

    # Analyze predictions
    analyze_sample_predictions(ovo_ensemble, test_loader, num_samples=5)

if __name__ == '__main__':
    main()
