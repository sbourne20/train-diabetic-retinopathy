#!/usr/bin/env python3
"""
Optimized 3-class DR dataset split using symlinks for speed.
Creates train/val/test splits (80/10/10) while maintaining class balance.
"""

import os
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Paths
source_dir = Path("/Volumes/Untitled/dr/dataset_eyepacs_3class/train")
target_dir = Path("/Volumes/Untitled/dr/dataset_eyepacs_3class_balanced")

# Split ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

classes = ["NORMAL", "NPDR", "PDR"]

print("=" * 80)
print("3-CLASS DR DATASET SPLIT (80/10/10) - OPTIMIZED")
print("=" * 80)

# Process each class
total_stats = {'train': {}, 'val': {}, 'test': {}}

for class_name in classes:
    print(f"\n📁 Processing class: {class_name}")

    # Get all image files
    class_dir = source_dir / class_name
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    print(f"   Scanning directory...", end=" ", flush=True)
    all_images = [f for f in class_dir.iterdir()
                  if f.is_file() and f.suffix in image_extensions]
    total_images = len(all_images)
    print(f"✅ Found {total_images:,} images")

    # Shuffle images randomly
    random.shuffle(all_images)

    # Calculate split sizes
    train_size = int(total_images * TRAIN_RATIO)
    val_size = int(total_images * VAL_RATIO)
    test_size = total_images - train_size - val_size

    print(f"   Train: {train_size:,} ({train_size/total_images*100:.1f}%)")
    print(f"   Val:   {val_size:,} ({val_size/total_images*100:.1f}%)")
    print(f"   Test:  {test_size:,} ({test_size/total_images*100:.1f}%)")

    # Split the images
    splits = {
        'train': all_images[:train_size],
        'val': all_images[train_size:train_size + val_size],
        'test': all_images[train_size + val_size:]
    }

    # Create symlinks (much faster than copying)
    for split_name, images in splits.items():
        target_class_dir = target_dir / split_name / class_name
        print(f"   Creating {len(images):,} symlinks in {split_name}...", end=" ", flush=True)

        for img_path in images:
            target_path = target_class_dir / img_path.name
            # Use symlink for speed
            os.symlink(img_path, target_path)

        total_stats[split_name][class_name] = len(images)
        print(f"✅")

print("\n" + "=" * 80)
print("📊 FINAL DATASET SUMMARY")
print("=" * 80)

# Print summary
for split_name in ['train', 'val', 'test']:
    print(f"\n{split_name.upper()}:")
    split_total = 0
    for class_name in classes:
        count = total_stats[split_name][class_name]
        print(f"  {class_name:8s}: {count:,}")
        split_total += count
    print(f"  {'TOTAL':8s}: {split_total:,}")

# Calculate balance info
print("\n" + "=" * 80)
print("📈 CLASS BALANCE ANALYSIS")
print("=" * 80)

original_counts = {
    'NORMAL': sum(total_stats[s].get('NORMAL', 0) for s in ['train', 'val', 'test']),
    'NPDR': sum(total_stats[s].get('NPDR', 0) for s in ['train', 'val', 'test']),
    'PDR': sum(total_stats[s].get('PDR', 0) for s in ['train', 'val', 'test'])
}

print("\nOriginal dataset distribution:")
total_all = sum(original_counts.values())
for class_name, count in original_counts.items():
    percentage = (count / total_all * 100) if total_all > 0 else 0
    print(f"  {class_name:8s}: {count:,} ({percentage:.1f}%)")

print(f"\n⚠️  Note: Using SYMLINKS (not copies) for faster operation")
print(f"    If you need actual copies, use: cp -RL source/ dest/")

print(f"\n✅ Dataset split completed successfully!")
print(f"📂 Output directory: {target_dir}")
