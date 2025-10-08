#!/usr/bin/env python3
"""
Split 3-class DR dataset into train/val/test (80/10/10) while maintaining class balance.
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

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
splits = ["train", "val", "test"]

print("=" * 80)
print("3-CLASS DR DATASET SPLIT (80/10/10)")
print("=" * 80)

# Process each class
for class_name in classes:
    print(f"\n📁 Processing class: {class_name}")

    # Get all image files for this class
    class_dir = source_dir / class_name
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    all_images = [f for f in class_dir.iterdir()
                  if f.is_file() and f.suffix in image_extensions]

    total_images = len(all_images)
    print(f"   Total images: {total_images:,}")

    # Shuffle images randomly
    random.shuffle(all_images)

    # Calculate split sizes
    train_size = int(total_images * TRAIN_RATIO)
    val_size = int(total_images * VAL_RATIO)
    test_size = total_images - train_size - val_size  # Remaining goes to test

    print(f"   Train: {train_size:,} ({train_size/total_images*100:.1f}%)")
    print(f"   Val:   {val_size:,} ({val_size/total_images*100:.1f}%)")
    print(f"   Test:  {test_size:,} ({test_size/total_images*100:.1f}%)")

    # Split the images
    train_images = all_images[:train_size]
    val_images = all_images[train_size:train_size + val_size]
    test_images = all_images[train_size + val_size:]

    # Copy files to respective directories
    split_data = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }

    for split_name, images in split_data.items():
        target_class_dir = target_dir / split_name / class_name
        print(f"   Copying {len(images):,} images to {split_name}...", end=" ")

        copied = 0
        for img_path in images:
            target_path = target_class_dir / img_path.name
            shutil.copy2(img_path, target_path)
            copied += 1

            # Progress indicator every 5000 images
            if copied % 5000 == 0:
                print(f"{copied:,}", end="...", flush=True)

        print(f"✅ Done ({copied:,} files)")

print("\n" + "=" * 80)
print("📊 FINAL DATASET SUMMARY")
print("=" * 80)

# Verify final counts
for split_name in splits:
    print(f"\n{split_name.upper()}:")
    split_total = 0
    for class_name in classes:
        class_dir = target_dir / split_name / class_name
        count = len(list(class_dir.glob("*.[jJ][pP]*[gG]")) + list(class_dir.glob("*.png")))
        print(f"  {class_name:8s}: {count:,}")
        split_total += count
    print(f"  {'TOTAL':8s}: {split_total:,}")

print("\n✅ Dataset split completed successfully!")
print(f"📂 Output directory: {target_dir}")
