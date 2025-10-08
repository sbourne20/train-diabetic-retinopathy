#!/usr/bin/env python3
"""
3-class DR dataset split using ACTUAL FILE COPIES (no symlinks).
Creates train/val/test splits (80/10/10) while maintaining class balance.
"""

import shutil
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
print("3-CLASS DR DATASET SPLIT (80/10/10) - ACTUAL FILE COPIES")
print("=" * 80)
print("⚠️  This will take time as we're copying actual files, not symlinks.")
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

    # Copy actual files (not symlinks)
    for split_name, images in splits.items():
        target_class_dir = target_dir / split_name / class_name
        total_files = len(images)
        print(f"   Copying {total_files:,} files to {split_name}...", end=" ", flush=True)

        copied = 0
        for img_path in images:
            target_path = target_class_dir / img_path.name
            # Use shutil.copy2 for actual file copy (preserves metadata)
            shutil.copy2(img_path, target_path)
            copied += 1

            # Progress indicator every 1000 files
            if copied % 1000 == 0:
                print(f"\n      Progress: {copied:,}/{total_files:,} ({copied/total_files*100:.1f}%)", end=" ", flush=True)

        total_stats[split_name][class_name] = len(images)
        print(f"\n   ✅ Completed: {copied:,} files copied")

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

print(f"\n✅ Dataset split completed successfully!")
print(f"📂 Output directory: {target_dir}")
print(f"📋 All files are ACTUAL COPIES (not symlinks)")
