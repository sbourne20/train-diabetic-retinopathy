#!/usr/bin/env python3
"""
Quick Test Set Evaluation for Trained Models

This script evaluates your trained OVO ensemble on the test set and displays
both validation and test accuracies for comparison.

Usage:
    python3 evaluate_test_set.py --results_dir ./densenet_5class_results
"""

import os
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import sys

# Import from ensemble trainer
sys.path.insert(0, os.path.dirname(__file__))
from ensemble_5class_trainer import OVOEnsemble, BinaryClassifier

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained OVO ensemble on test set')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Path to results directory (e.g., ./densenet_5class_results)')
    parser.add_argument('--dataset_path', type=str, default='./dataset_eyepacs_5class_balanced',
                       help='Path to dataset root')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    return parser.parse_args()

def load_checkpoint_accuracies(results_dir):
    """Load validation accuracies from all binary classifier checkpoints"""
    models_dir = Path(results_dir) / "models"

    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return None

    # Find all checkpoint files
    checkpoint_files = list(models_dir.glob("*.pth"))

    if not checkpoint_files:
        print(f"❌ No checkpoint files found in {models_dir}")
        return None

    print(f"\n📊 Found {len(checkpoint_files)} checkpoint files")

    # Extract validation accuracies
    val_accuracies = {}
    model_name = None

    for ckpt_path in checkpoint_files:
        filename = ckpt_path.name

        # Skip ensemble checkpoint
        if 'ensemble' in filename.lower():
            continue

        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

            # Extract model name and class pair
            if 'best_' in filename:
                # Format: best_densenet121_0_1.pth
                parts = filename.replace('best_', '').replace('.pth', '').split('_')
                if len(parts) >= 3:
                    model_name = parts[0]
                    class_a = parts[-2]
                    class_b = parts[-1]
                    pair_key = f"{class_a}-{class_b}"

                    # Extract validation accuracy
                    val_acc = None
                    for key in ['best_val_accuracy', 'val_accuracy', 'best_accuracy']:
                        if key in checkpoint:
                            val_acc = checkpoint[key]
                            break

                    if val_acc is not None:
                        val_accuracies[pair_key] = val_acc
        except Exception as e:
            print(f"   ⚠️ Could not load {filename}: {e}")
            continue

    return val_accuracies, model_name

def evaluate_on_test_set(results_dir, dataset_path, batch_size):
    """Evaluate the trained ensemble on test set"""

    print("\n" + "="*80)
    print("🧪 TEST SET EVALUATION")
    print("="*80)

    # Load validation accuracies from checkpoints
    val_accuracies, model_name = load_checkpoint_accuracies(results_dir)

    if val_accuracies is None or model_name is None:
        print("❌ Could not extract validation accuracies from checkpoints")
        return

    print(f"\n✅ Detected model: {model_name}")
    print(f"✅ Found {len(val_accuracies)} binary classifier pairs")

    # Load test dataset
    test_path = Path(dataset_path) / "test"

    if not test_path.exists():
        print(f"❌ Test dataset not found: {test_path}")
        return

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = ImageFolder(str(test_path), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"✅ Test dataset loaded: {len(test_dataset)} images")

    # Load OVO ensemble
    models_dir = Path(results_dir) / "models"
    ensemble_path = models_dir / "ovo_ensemble_best.pth"

    if not ensemble_path.exists():
        print(f"❌ Ensemble checkpoint not found: {ensemble_path}")
        return

    print(f"✅ Loading ensemble from {ensemble_path}")

    try:
        # Detect number of classes from test dataset
        num_classes = len(test_dataset.classes)

        # Create OVO ensemble
        ovo_ensemble = OVOEnsemble(
            base_models=[model_name],
            num_classes=num_classes,
            freeze_weights=True,
            dropout=0.5
        )

        # Load ensemble weights
        checkpoint = torch.load(ensemble_path, map_location='cpu', weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            ovo_ensemble.load_state_dict(checkpoint['model_state_dict'])
        else:
            ovo_ensemble.load_state_dict(checkpoint)

        print("✅ Ensemble loaded successfully")

    except Exception as e:
        print(f"❌ Error loading ensemble: {e}")
        return

    # Evaluate on test set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ovo_ensemble = ovo_ensemble.to(device)
    ovo_ensemble.eval()

    print(f"\n🔄 Evaluating on test set using {device}...")

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            targets = targets.to(device)

            # Get ensemble predictions
            outputs = ovo_ensemble(images)
            _, predictions = torch.max(outputs['logits'], 1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate test accuracy
    test_accuracy = accuracy_score(all_targets, all_predictions)

    # Display results
    print("\n" + "="*80)
    print("📊 RESULTS COMPARISON")
    print("="*80)

    print(f"\n{'Binary Pair':<15} | {'Validation Acc':<15} | {'Notes'}")
    print("-" * 60)

    for pair, val_acc in sorted(val_accuracies.items()):
        print(f"{pair:<15} | {val_acc*100:>13.2f}% | Individual pair")

    # Calculate average validation accuracy
    avg_val_acc = sum(val_accuracies.values()) / len(val_accuracies)

    print("-" * 60)
    print(f"{'Average':<15} | {avg_val_acc*100:>13.2f}% | Validation average")
    print(f"{'ENSEMBLE':<15} | {test_accuracy*100:>13.2f}% | **TEST SET** ✅")
    print("=" * 80)

    # Medical grade assessment
    print(f"\n🏥 Medical Grade Assessment:")
    print(f"   Test Accuracy: {test_accuracy*100:.2f}%")

    if test_accuracy >= 0.90:
        print(f"   Status: ✅✅ MEDICAL-GRADE (≥90%)")
    elif test_accuracy >= 0.85:
        print(f"   Status: ✅ RESEARCH QUALITY (≥85%)")
    else:
        print(f"   Status: ⚠️ BELOW TARGET (<85%)")

    print(f"\n💡 Interpretation:")
    print(f"   - Average validation accuracy (binary pairs): {avg_val_acc*100:.2f}%")
    print(f"   - Ensemble test accuracy: {test_accuracy*100:.2f}%")

    if test_accuracy > avg_val_acc:
        diff = (test_accuracy - avg_val_acc) * 100
        print(f"   - Test is {diff:.2f}% HIGHER than validation average ✅")
        print(f"   - Indicates: Good generalization, ensemble voting helps")
    elif test_accuracy < avg_val_acc:
        diff = (avg_val_acc - test_accuracy) * 100
        print(f"   - Test is {diff:.2f}% LOWER than validation average ⚠️")
        print(f"   - Indicates: Possible overfitting or validation/test distribution mismatch")
    else:
        print(f"   - Test matches validation average ✅")
        print(f"   - Indicates: Consistent performance across splits")

    print("\n" + "="*80)

if __name__ == "__main__":
    args = parse_args()
    evaluate_on_test_set(args.results_dir, args.dataset_path, args.batch_size)
