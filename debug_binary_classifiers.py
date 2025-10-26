#!/usr/bin/env python3
"""
Debug script to test individual binary classifiers on test set
and identify which pairs are failing.
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score
import sys

# Import from ensemble_5class_trainer
sys.path.insert(0, str(Path(__file__).parent))
from ensemble_5class_trainer import BinaryClassifier, CLAHETransform

def test_binary_classifier(model_path, class_a, class_b, test_dataset):
    """Test a single binary classifier on test set."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = BinaryClassifier('coatnet_0_rw_224', freeze_weights=True, dropout=0.28)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Filter test dataset for only class_a and class_b
    indices = [i for i, (_, label) in enumerate(test_dataset) if label in [class_a, class_b]]
    filtered_dataset = Subset(test_dataset, indices)
    test_loader = DataLoader(filtered_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Evaluate
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc=f"Testing pair {class_a}_{class_b}", leave=False):
            images = images.to(device)
            targets = targets.to(device)

            # Convert multi-class labels to binary (0 = class_a, 1 = class_b)
            binary_targets = (targets == class_b).float()

            # Get predictions
            logits = model(images).squeeze()
            binary_preds = (torch.sigmoid(logits) > 0.5).float()

            all_predictions.extend(binary_preds.cpu().numpy())
            all_targets.extend(binary_targets.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(all_targets, all_predictions)

    # Get validation accuracy from checkpoint
    val_accuracy = checkpoint.get('best_val_accuracy', 0.0)

    return {
        'pair': f'{class_a}_{class_b}',
        'test_accuracy': accuracy * 100,
        'val_accuracy': val_accuracy,
        'difference': val_accuracy - (accuracy * 100),
        'num_samples': len(all_targets)
    }

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

    print("🔍 Testing Individual Binary Classifiers on Test Set")
    print("=" * 80)

    # Test all 10 pairs
    class_pairs = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    results = []

    for class_a, class_b in class_pairs:
        model_path = Path(f'./coatnet_5class_results/models/best_coatnet_0_rw_224_{class_a}_{class_b}.pth')

        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            continue

        result = test_binary_classifier(model_path, class_a, class_b, test_dataset)
        results.append(result)

        # Print result
        status = "✅" if result['difference'] < 5.0 else "⚠️" if result['difference'] < 10.0 else "❌"
        print(f"{status} Pair {result['pair']}: Val={result['val_accuracy']:.2f}% | Test={result['test_accuracy']:.2f}% | Diff={result['difference']:.2f}% | Samples={result['num_samples']}")

    print("\n" + "=" * 80)
    print("📊 Summary:")
    print(f"   Average Val Accuracy: {np.mean([r['val_accuracy'] for r in results]):.2f}%")
    print(f"   Average Test Accuracy: {np.mean([r['test_accuracy'] for r in results]):.2f}%")
    print(f"   Average Difference: {np.mean([r['difference'] for r in results]):.2f}%")
    print("\n🔍 Pairs with largest val-test gap:")
    sorted_results = sorted(results, key=lambda x: x['difference'], reverse=True)
    for r in sorted_results[:3]:
        print(f"   Pair {r['pair']}: {r['difference']:.2f}% gap (Val={r['val_accuracy']:.2f}%, Test={r['test_accuracy']:.2f}%)")

if __name__ == '__main__':
    main()
