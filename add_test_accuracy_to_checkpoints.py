#!/usr/bin/env python3
"""
Add Test Accuracy to Existing Checkpoints

This script evaluates trained binary classifiers on test set and adds
test_accuracy to each checkpoint file.

Usage:
    python3 add_test_accuracy_to_checkpoints.py --results_dir ./mobilenet_5class_v2_fixed_results
"""

import os
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import sys

# Import from ensemble trainer
sys.path.insert(0, os.path.dirname(__file__))
from ensemble_5class_trainer import BinaryClassifier

def parse_args():
    parser = argparse.ArgumentParser(description='Add test accuracy to trained checkpoints')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Path to results directory')
    parser.add_argument('--dataset_path', type=str, default='./dataset_eyepacs_5class_balanced',
                       help='Path to dataset root')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    return parser.parse_args()

def create_binary_test_loader(test_dataset, class_a, class_b, batch_size, transform):
    """Create binary test loader for specific class pair"""
    indices = []
    labels = []

    for idx in range(len(test_dataset)):
        _, label = test_dataset.imgs[idx]
        if label == class_a:
            indices.append(idx)
            labels.append(0)  # Binary label 0
        elif label == class_b:
            indices.append(idx)
            labels.append(1)  # Binary label 1

    # Create subset
    binary_subset = Subset(test_dataset, indices)

    # Create data loader
    test_loader = DataLoader(
        binary_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return test_loader, labels

def evaluate_binary_classifier(model, test_loader, device):
    """Evaluate binary classifier on test set"""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            # Get predictions
            outputs = model(images)
            predictions = (outputs > 0.5).long().squeeze()

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate accuracy
    test_accuracy = accuracy_score(all_targets, all_predictions)
    return test_accuracy

def add_test_accuracy_to_checkpoints(results_dir, dataset_path, batch_size):
    """Add test accuracy to all binary classifier checkpoints"""

    print("\n" + "="*80)
    print("🧪 ADDING TEST ACCURACY TO CHECKPOINTS")
    print("="*80)

    models_dir = Path(results_dir) / "models"

    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return

    # Find all binary classifier checkpoints
    checkpoint_files = [f for f in models_dir.glob("best_*.pth") if 'ensemble' not in f.name.lower()]

    if not checkpoint_files:
        print(f"❌ No binary classifier checkpoints found in {models_dir}")
        return

    print(f"\n📊 Found {len(checkpoint_files)} binary classifier checkpoints")

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

    test_dataset = ImageFolder(str(test_path))

    print(f"✅ Test dataset loaded: {len(test_dataset)} images")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using device: {device}")

    updated_count = 0

    # Process each checkpoint
    for ckpt_path in tqdm(checkpoint_files, desc="Processing checkpoints"):
        try:
            # Load checkpoint
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

            # Check if test_accuracy already exists
            if 'test_accuracy' in checkpoint and checkpoint['test_accuracy'] is not None:
                print(f"   ⏭️ Skipping {ckpt_path.name} - test_accuracy already exists ({checkpoint['test_accuracy']:.4f})")
                continue

            # Extract model info from checkpoint
            if 'model_name' not in checkpoint or 'class_pair' not in checkpoint:
                print(f"   ⚠️ Skipping {ckpt_path.name} - missing model_name or class_pair")
                continue

            model_name = checkpoint['model_name']
            class_pair = checkpoint['class_pair']
            class_a, class_b = class_pair

            # Create binary classifier
            binary_model = BinaryClassifier(
                model_name=model_name,
                freeze_weights=True,
                dropout=0.5
            )

            # Load model weights
            binary_model.load_state_dict(checkpoint['model_state_dict'])
            binary_model = binary_model.to(device)

            # Create binary test loader for this specific pair
            test_loader, _ = create_binary_test_loader(
                test_dataset, class_a, class_b, batch_size, test_transform
            )

            # Evaluate on test set
            test_accuracy = evaluate_binary_classifier(binary_model, test_loader, device)

            # Add test_accuracy to checkpoint
            checkpoint['test_accuracy'] = test_accuracy

            # Save updated checkpoint
            torch.save(checkpoint, ckpt_path)

            updated_count += 1

            print(f"   ✅ {ckpt_path.name}: Val={checkpoint['best_val_accuracy']:.4f}, Test={test_accuracy:.4f}")

        except Exception as e:
            print(f"   ❌ Error processing {ckpt_path.name}: {e}")
            continue

    print("\n" + "="*80)
    print(f"✅ Successfully updated {updated_count} checkpoints with test accuracy!")
    print("="*80)

    # Now run model_analyzer to show the results
    print("\n📊 Running model_analyzer to verify...")
    print("="*80)

    import subprocess
    result = subprocess.run(
        ['python3', 'model_analyzer.py', '--model', str(models_dir)],
        capture_output=False
    )

if __name__ == "__main__":
    args = parse_args()
    add_test_accuracy_to_checkpoints(args.results_dir, args.dataset_path, args.batch_size)
