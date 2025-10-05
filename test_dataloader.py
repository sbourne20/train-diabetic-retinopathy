#!/usr/bin/env python3
"""
Test DataLoader to verify label correctness
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import Counter

def test_dataloader(dataset_path):
    """Test if dataloader is reading labels correctly"""

    print("="*80)
    print("DATALOADER LABEL VERIFICATION")
    print("="*80)
    print(f"\nDataset: {dataset_path}\n")

    # Simple transform (same as training)
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load training set
    train_dataset = datasets.ImageFolder(
        root=f"{dataset_path}/train",
        transform=transform
    )

    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    print(f"Dataset size: {len(train_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Class names: {train_dataset.classes}")
    print(f"Class to index mapping: {train_dataset.class_to_idx}")

    # Check first batch
    print("\n" + "="*80)
    print("FIRST BATCH ANALYSIS")
    print("="*80)

    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    print(f"\nBatch images shape: {images.shape}")
    print(f"Batch labels shape: {labels.shape}")
    print(f"Labels in first batch: {labels.tolist()}")
    print(f"Label distribution: {Counter(labels.tolist())}")

    # Check if images are valid
    print(f"\nImage tensor stats:")
    print(f"  Min value: {images.min():.4f}")
    print(f"  Max value: {images.max():.4f}")
    print(f"  Mean value: {images.mean():.4f}")
    print(f"  Std value: {images.std():.4f}")

    # Sample multiple batches to check label distribution
    print("\n" + "="*80)
    print("LABEL DISTRIBUTION (first 1000 samples)")
    print("="*80)

    all_labels = []
    for i, (_, labels_batch) in enumerate(train_loader):
        all_labels.extend(labels_batch.tolist())
        if i * 16 >= 1000:
            break

    label_counts = Counter(all_labels)
    print(f"\nLabel distribution:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        percentage = (count / len(all_labels)) * 100
        print(f"  Class {label}: {count:>4} samples ({percentage:>5.1f}%)")

    # Check if distribution is balanced
    expected_per_class = len(all_labels) / 5
    max_deviation = max(abs(count - expected_per_class) for count in label_counts.values())
    deviation_pct = (max_deviation / expected_per_class) * 100

    print(f"\nExpected per class: {expected_per_class:.1f}")
    print(f"Max deviation: {max_deviation:.1f} ({deviation_pct:.1f}%)")

    if deviation_pct > 10:
        print("⚠️  WARNING: Label distribution is NOT balanced!")
    else:
        print("✅ Label distribution is balanced")

    # Sample images from each class folder
    print("\n" + "="*80)
    print("SAMPLE FILES FROM EACH CLASS FOLDER")
    print("="*80)

    import os
    for cls in range(5):
        class_path = f"{dataset_path}/train/{cls}"
        if os.path.exists(class_path):
            files = [f for f in os.listdir(class_path) if f.endswith('.jpg') and not f.startswith('._')][:3]
            print(f"\nClass {cls} folder samples:")
            for f in files:
                print(f"  • {f}")

    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)

    if deviation_pct > 10:
        print("\n❌ UNBALANCED LABELS - Data loading issue!")
        print("\nPossible causes:")
        print("  1. ImageFolder not reading directories correctly")
        print("  2. macOS metadata files interfering")
        print("  3. Incorrect directory structure")
    elif images.std() < 0.1:
        print("\n❌ IMAGES ARE TOO UNIFORM - Normalization issue!")
        print("\nPossible causes:")
        print("  1. All images are blank/corrupted")
        print("  2. Augmentation created uniform images")
        print("  3. Wrong normalization values")
    else:
        print("\n✅ DataLoader appears to be working correctly")
        print("\nIf training still fails with ~50% accuracy:")
        print("  1. Labels might be WRONG (images in incorrect class folders)")
        print("  2. Augmented images don't preserve diagnostic features")
        print("  3. Model architecture issue")

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = './augmented_resized_V2_balanced'

    test_dataloader(dataset_path)
