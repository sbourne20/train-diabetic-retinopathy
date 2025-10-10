#!/usr/bin/env python3
"""
Balanced Dataset Splitter for 5-Class Diabetic Retinopathy
Splits dataset into train/val/test with balanced distribution
No augmentation - pure file copying

Source: /Volumes/WDC1TB/dataset_eyepacs/eyepacs
Target: /Volumes/WDC1TB/dataset_eyepacs_split

Strategy:
- Train: 70%
- Val: 15%
- Test: 15%
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Configuration
SOURCE_DIR = Path('/Volumes/WDC1TB/dataset_eyepacs/eyepacs')
TARGET_DIR = Path('/Volumes/WDC1TB/dataset_eyepacs_split')
NUM_CLASSES = 5
SPLIT_RATIOS = {
    'train': 0.70,
    'val': 0.15,
    'test': 0.15
}

# Set random seed for reproducibility
random.seed(42)

def create_directory_structure():
    """Create train/val/test directories with 5 classes each"""
    print("📁 Creating directory structure...")
    for split in ['train', 'val', 'test']:
        for class_idx in range(NUM_CLASSES):
            target_path = TARGET_DIR / split / str(class_idx)
            target_path.mkdir(parents=True, exist_ok=True)
            print(f"   Created: {target_path}")
    print()

def collect_image_files():
    """Collect all image files per class"""
    print("📊 Collecting image files...")
    class_files = defaultdict(list)

    for class_idx in range(NUM_CLASSES):
        class_path = SOURCE_DIR / str(class_idx)
        if not class_path.exists():
            print(f"   ⚠️  Warning: Class {class_idx} directory not found!")
            continue

        # Collect all image files
        image_files = []
        for ext in ['*.jpeg', '*.jpg', '*.png', '*.JPEG', '*.JPG', '*.PNG']:
            image_files.extend(list(class_path.glob(ext)))

        class_files[class_idx] = image_files
        print(f"   Class {class_idx}: {len(image_files)} images")

    print()
    return class_files

def calculate_splits(class_files):
    """Calculate balanced split sizes per class"""
    print("🎯 Calculating balanced splits...")
    split_plan = {}

    for class_idx, files in class_files.items():
        total = len(files)
        train_size = int(total * SPLIT_RATIOS['train'])
        val_size = int(total * SPLIT_RATIOS['val'])
        test_size = total - train_size - val_size  # Remaining goes to test

        split_plan[class_idx] = {
            'total': total,
            'train': train_size,
            'val': val_size,
            'test': test_size
        }

        print(f"   Class {class_idx}:")
        print(f"      Total: {total:5d} | Train: {train_size:5d} ({train_size/total*100:.1f}%) | "
              f"Val: {val_size:5d} ({val_size/total*100:.1f}%) | Test: {test_size:5d} ({test_size/total*100:.1f}%)")

    print()
    return split_plan

def split_and_copy_files(class_files, split_plan):
    """Split files and copy to target directories"""
    print("📋 Splitting and copying files...")

    for class_idx, files in class_files.items():
        print(f"\n   Processing Class {class_idx}...")

        # Shuffle files for random distribution
        shuffled_files = files.copy()
        random.shuffle(shuffled_files)

        # Calculate split indices
        train_size = split_plan[class_idx]['train']
        val_size = split_plan[class_idx]['val']

        train_files = shuffled_files[:train_size]
        val_files = shuffled_files[train_size:train_size + val_size]
        test_files = shuffled_files[train_size + val_size:]

        # Copy files
        splits_data = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }

        for split_name, file_list in splits_data.items():
            target_dir = TARGET_DIR / split_name / str(class_idx)
            print(f"      Copying {len(file_list)} files to {split_name}...")

            for idx, src_file in enumerate(file_list, 1):
                dst_file = target_dir / src_file.name
                shutil.copy2(src_file, dst_file)

                # Progress indicator
                if idx % 500 == 0 or idx == len(file_list):
                    print(f"         Progress: {idx}/{len(file_list)} files", end='\r')

            print(f"         ✅ Completed: {len(file_list)} files")

def verify_splits():
    """Verify the split was successful"""
    print("\n🔍 Verifying split distribution...")

    total_counts = {'train': 0, 'val': 0, 'test': 0}

    for split in ['train', 'val', 'test']:
        print(f"\n   {split.upper()}:")
        split_total = 0

        for class_idx in range(NUM_CLASSES):
            class_path = TARGET_DIR / split / str(class_idx)
            if class_path.exists():
                count = len(list(class_path.glob('*.*')))
                split_total += count
                print(f"      Class {class_idx}: {count:5d} images")
            else:
                print(f"      Class {class_idx}: ⚠️  Directory not found!")

        total_counts[split] = split_total
        print(f"      Total: {split_total} images")

    # Summary
    print("\n📊 SPLIT SUMMARY:")
    grand_total = sum(total_counts.values())
    for split, count in total_counts.items():
        percentage = (count / grand_total * 100) if grand_total > 0 else 0
        print(f"   {split.capitalize()}: {count:6d} images ({percentage:.1f}%)")
    print(f"   Grand Total: {grand_total} images")

def main():
    """Main execution"""
    print("=" * 80)
    print("🏥 DIABETIC RETINOPATHY DATASET SPLITTER - 5 CLASS BALANCED")
    print("=" * 80)
    print()

    # Check source directory
    if not SOURCE_DIR.exists():
        print(f"❌ Error: Source directory not found: {SOURCE_DIR}")
        return

    print(f"📂 Source: {SOURCE_DIR}")
    print(f"📂 Target: {TARGET_DIR}")
    print(f"🎯 Split Ratios: Train={SPLIT_RATIOS['train']*100:.0f}% | "
          f"Val={SPLIT_RATIOS['val']*100:.0f}% | Test={SPLIT_RATIOS['test']*100:.0f}%")
    print()

    # Check if target directory already exists
    if TARGET_DIR.exists():
        response = input(f"⚠️  Target directory already exists: {TARGET_DIR}\n"
                        f"   Do you want to continue? This will overwrite existing files. (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("❌ Operation cancelled.")
            return
        print()

    # Execute splitting pipeline
    create_directory_structure()
    class_files = collect_image_files()

    if not class_files:
        print("❌ Error: No image files found in source directory!")
        return

    split_plan = calculate_splits(class_files)
    split_and_copy_files(class_files, split_plan)
    verify_splits()

    print()
    print("=" * 80)
    print("✅ DATASET SPLITTING COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
