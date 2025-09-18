#!/usr/bin/env python3
"""
Diagnose OVO Performance Gap
Analyzes why 91-93% binary validation → 72% ensemble test performance
"""

import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from ensemble_local_trainer import create_ovo_transforms, BinaryClassifier
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import json

def analyze_individual_binary_performance():
    """Test individual binary classifiers on full test set to understand the gap."""

    results_dir = Path("./ovo_ensemble_results_v2")
    models_dir = results_dir / "models"

    # Load test dataset
    test_transform = create_ovo_transforms(img_size=299, enable_clahe=False)[1]
    test_dataset = ImageFolder(root="./dataset6/test", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    print(f"📊 Test Dataset Analysis:")
    print(f"   Total samples: {len(test_dataset)}")

    # Analyze class distribution
    class_counts = [0] * 5
    for _, label in test_dataset:
        class_counts[label] += 1

    print(f"   Class distribution: {class_counts}")
    print(f"   Class 0: {class_counts[0]/len(test_dataset)*100:.1f}%")
    print(f"   Class 1: {class_counts[1]/len(test_dataset)*100:.1f}%")
    print(f"   Class 2: {class_counts[2]/len(test_dataset)*100:.1f}%")
    print(f"   Class 3: {class_counts[3]/len(test_dataset)*100:.1f}%")
    print(f"   Class 4: {class_counts[4]/len(test_dataset)*100:.1f}%")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test individual binary classifiers on test data
    architectures = ['mobilenet_v2', 'inception_v3', 'densenet121']
    class_pairs = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]

    print(f"\n🔍 Individual Binary Classifier Test Performance:")
    print("=" * 80)

    all_test_results = {}

    for arch in architectures:
        print(f"\n🏗️ {arch.upper()}")
        print("-" * 60)

        arch_results = {}

        for class_a, class_b in class_pairs:
            model_path = models_dir / f"best_{arch}_{class_a}_{class_b}.pth"

            if not model_path.exists():
                print(f"   ❌ Missing: {class_a}-{class_b}")
                continue

            # Load binary classifier
            try:
                model = BinaryClassifier(model_name=arch, freeze_weights=True, dropout=0.3)
                checkpoint = torch.load(model_path, map_location='cpu')

                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    val_acc = checkpoint.get('best_val_accuracy', 0.0)
                else:
                    model.load_state_dict(checkpoint)
                    val_acc = 0.0

                model = model.to(device)
                model.eval()

                # Create binary test dataset for this pair
                binary_predictions = []
                binary_targets = []

                with torch.no_grad():
                    for images, targets in test_loader:
                        images = images.to(device)

                        # Filter for this binary pair
                        pair_mask = (targets == class_a) | (targets == class_b)
                        if pair_mask.sum() == 0:
                            continue

                        pair_images = images[pair_mask]
                        pair_targets = targets[pair_mask]

                        # Convert to binary labels (class_a=0, class_b=1)
                        binary_labels = (pair_targets == class_b).float()

                        # Get predictions
                        outputs = model(pair_images).squeeze()
                        if outputs.dim() == 0:
                            outputs = outputs.unsqueeze(0)

                        binary_preds = (outputs > 0.5).float()

                        binary_predictions.extend(binary_preds.cpu().numpy())
                        binary_targets.extend(binary_labels.cpu().numpy())

                if len(binary_predictions) > 0:
                    test_acc = np.mean(np.array(binary_predictions) == np.array(binary_targets)) * 100
                    gap = val_acc - test_acc

                    status = "✅" if gap < 10 else "⚠️" if gap < 20 else "❌"
                    print(f"   {status} Classes {class_a}-{class_b}: Val={val_acc:.1f}% → Test={test_acc:.1f}% (Gap: {gap:.1f}%)")

                    arch_results[f"{class_a}_{class_b}"] = {
                        'val_accuracy': val_acc,
                        'test_accuracy': test_acc,
                        'gap': gap,
                        'test_samples': len(binary_predictions)
                    }
                else:
                    print(f"   ⚪ Classes {class_a}-{class_b}: No test samples")

            except Exception as e:
                print(f"   ❌ Error {class_a}-{class_b}: {e}")

        all_test_results[arch] = arch_results

        # Calculate architecture summary
        if arch_results:
            gaps = [r['gap'] for r in arch_results.values()]
            avg_gap = np.mean(gaps)
            max_gap = np.max(gaps)

            print(f"\n   📈 {arch} SUMMARY:")
            print(f"      Average Val→Test Gap: {avg_gap:.1f}%")
            print(f"      Maximum Gap: {max_gap:.1f}%")
            print(f"      Models with >15% gap: {sum(1 for g in gaps if g > 15)}/{len(gaps)}")

    # Overall analysis
    print(f"\n🎯 OVERALL GAP ANALYSIS:")
    print("=" * 80)

    all_gaps = []
    severe_gaps = 0

    for arch_results in all_test_results.values():
        for result in arch_results.values():
            gap = result['gap']
            all_gaps.append(gap)
            if gap > 15:
                severe_gaps += 1

    if all_gaps:
        print(f"   📊 Statistics:")
        print(f"      Mean gap: {np.mean(all_gaps):.1f}%")
        print(f"      Median gap: {np.median(all_gaps):.1f}%")
        print(f"      Max gap: {np.max(all_gaps):.1f}%")
        print(f"      Models with severe gaps (>15%): {severe_gaps}/{len(all_gaps)}")

        if np.mean(all_gaps) > 10:
            print(f"\n   ⚠️ DIAGNOSIS: SEVERE OVERFITTING")
            print(f"      Models are overfitted to validation data")
            print(f"      Poor generalization to test set")
        else:
            print(f"\n   ✅ DIAGNOSIS: Gaps are reasonable")

    # Save detailed results
    output_path = results_dir / "results" / "binary_test_analysis.json"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_test_results, f, indent=2)

    print(f"\n💾 Detailed analysis saved: {output_path}")

    return all_test_results

def analyze_ensemble_voting_breakdown():
    """Analyze what happens during ensemble voting."""

    print(f"\n🗳️ ENSEMBLE VOTING ANALYSIS:")
    print("=" * 80)

    # Load test dataset
    test_transform = create_ovo_transforms(img_size=299, enable_clahe=False)[1]
    test_dataset = ImageFolder(root="./dataset6/test", transform=test_transform)

    # Sample a few examples for detailed analysis
    sample_indices = [0, 100, 500, 1000, 1500]  # Different classes hopefully

    for idx in sample_indices:
        if idx >= len(test_dataset):
            continue

        image, true_label = test_dataset[idx]
        print(f"\n📝 Sample {idx}: True Class = {true_label}")

        # Here you would run the image through each binary classifier
        # and show the voting breakdown - this requires more implementation
        # but gives the idea of what needs to be analyzed

        print(f"   Binary votes breakdown would go here...")
        print(f"   Final ensemble prediction would go here...")

if __name__ == "__main__":
    print("🚨 DIAGNOSING OVO ENSEMBLE PERFORMANCE GAP")
    print("=" * 60)

    # Step 1: Analyze individual binary classifier test performance
    binary_results = analyze_individual_binary_performance()

    # Step 2: Analyze ensemble voting
    analyze_ensemble_voting_breakdown()

    print(f"\n✅ Diagnosis complete!")
    print(f"📁 Check binary_test_analysis.json for detailed results")