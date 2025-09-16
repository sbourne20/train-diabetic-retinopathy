#!/usr/bin/env python3
"""
Dataset Splitter for Diabetic Retinopathy Classification
Creates stratified train/val/test splits while maintaining class balance
"""

import os
import shutil
import random
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
from collections import defaultdict

def analyze_dataset(source_dir):
    """Analyze the current dataset structure and class distribution"""
    print("\n" + "="*60)
    print("📊 DATASET ANALYSIS")
    print("="*60)

    class_counts = {}
    total_images = 0

    for class_folder in sorted(os.listdir(source_dir)):
        if class_folder.startswith('.'):
            continue

        class_path = os.path.join(source_dir, class_folder)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            class_counts[int(class_folder)] = count
            total_images += count
            print(f"🔹 Class {class_folder}: {count:,} images ({count/total_images*100:.1f}%)")

    print(f"\n📈 Total Images: {total_images:,}")

    # Check for class imbalance
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    imbalance_ratio = max_count / min_count

    if imbalance_ratio > 10:
        print(f"⚠️  Severe class imbalance detected: {imbalance_ratio:.1f}:1 ratio")
        print("   Recommendation: Use class weights and SMOTE during training")
    elif imbalance_ratio > 3:
        print(f"⚠️  Moderate class imbalance: {imbalance_ratio:.1f}:1 ratio")
        print("   Recommendation: Use class weights during training")
    else:
        print(f"✅ Balanced dataset: {imbalance_ratio:.1f}:1 ratio")

    return class_counts, total_images

def collect_files_by_class(source_dir):
    """Collect all image files organized by class"""
    files_by_class = defaultdict(list)

    for class_folder in os.listdir(source_dir):
        if class_folder.startswith('.'):
            continue

        class_path = os.path.join(source_dir, class_folder)
        if os.path.isdir(class_path):
            class_id = int(class_folder)

            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(class_path, filename)
                    files_by_class[class_id].append((filename, file_path))

    return files_by_class

def create_stratified_split(files_by_class, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """Create stratified train/val/test splits maintaining class balance"""

    # Verify ratios sum to 1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    random.seed(random_seed)

    splits = {'train': defaultdict(list), 'val': defaultdict(list), 'test': defaultdict(list)}

    print("\n" + "="*60)
    print("📂 CREATING STRATIFIED SPLITS")
    print("="*60)
    print(f"🔹 Train: {train_ratio*100:.0f}% | Val: {val_ratio*100:.0f}% | Test: {test_ratio*100:.0f}%")
    print(f"🎲 Random seed: {random_seed}")
    print()

    for class_id, files in files_by_class.items():
        # Shuffle files for this class
        random.shuffle(files)

        total_files = len(files)
        train_count = int(total_files * train_ratio)
        val_count = int(total_files * val_ratio)
        test_count = total_files - train_count - val_count  # Remaining files go to test

        # Split files
        train_files = files[:train_count]
        val_files = files[train_count:train_count + val_count]
        test_files = files[train_count + val_count:]

        splits['train'][class_id] = train_files
        splits['val'][class_id] = val_files
        splits['test'][class_id] = test_files

        print(f"Class {class_id}: {len(train_files):4d} train | {len(val_files):4d} val | {len(test_files):4d} test")

    return splits

def copy_files_to_splits(splits, output_dir, source_dir):
    """Move files to their respective train/val/test directories inside dataset6"""

    print("\n" + "="*60)
    print("📁 MOVING FILES TO SPLIT DIRECTORIES")
    print("="*60)

    # Create val and test directories (train already exists)
    for split_name in ['val', 'test']:
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        # Create class subdirectories
        for class_id in range(5):  # DR classes 0-4
            class_dir = os.path.join(split_dir, str(class_id))
            os.makedirs(class_dir, exist_ok=True)

    # Process files
    total_moved = 0
    total_kept = 0

    for split_name, class_files in splits.items():
        split_total = 0

        for class_id, files in class_files.items():
            if split_name == 'train':
                # Keep training files in original location
                split_total += len(files)
                total_kept += len(files)
            else:
                # Move val/test files to new directories
                target_dir = os.path.join(output_dir, split_name, str(class_id))

                for filename, source_path in files:
                    target_path = os.path.join(target_dir, filename)
                    shutil.move(source_path, target_path)
                    split_total += 1
                    total_moved += 1

        if split_name == 'train':
            print(f"✅ {split_name.capitalize()}: {split_total:,} files kept in place")
        else:
            print(f"✅ {split_name.capitalize()}: {split_total:,} files moved")

    print(f"\n📊 Files moved: {total_moved:,}, Files kept: {total_kept:,}")
    return total_moved + total_kept

def verify_splits(output_dir):
    """Verify the created splits are correct"""

    print("\n" + "="*60)
    print("🔍 VERIFYING SPLITS")
    print("="*60)

    for split_name in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split_name)
        split_total = 0

        print(f"\n📁 {split_name.upper()} Split:")
        for class_id in range(5):
            class_dir = os.path.join(split_dir, str(class_id))
            if os.path.exists(class_dir):
                count = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                split_total += count
                print(f"   Class {class_id}: {count:,} images")

        print(f"   📊 {split_name.capitalize()} Total: {split_total:,} images")

def main():
    parser = argparse.ArgumentParser(description="Split diabetic retinopathy dataset into train/val/test")
    parser.add_argument("--source_dir", default="./dataset6/train",
                       help="Source directory containing class folders")
    parser.add_argument("--output_dir", default="./dataset6",
                       help="Output directory for split dataset (will create val and test inside)")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                       help="Training set ratio (default: 0.7)")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                       help="Validation set ratio (default: 0.15)")
    parser.add_argument("--test_ratio", type=float, default=0.15,
                       help="Test set ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--force", action="store_true",
                       help="Force overwrite existing output directory")

    args = parser.parse_args()

    # Validate source directory
    if not os.path.exists(args.source_dir):
        print(f"❌ Error: Source directory '{args.source_dir}' does not exist!")
        return 1

    # Check if output directory exists
    if os.path.exists(args.output_dir) and not args.force:
        print(f"❌ Error: Output directory '{args.output_dir}' already exists!")
        print("   Use --force to overwrite or choose a different output directory")
        return 1

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("🚀 DIABETIC RETINOPATHY DATASET SPLITTER")
    print(f"📁 Source: {args.source_dir}")
    print(f"📁 Output: {args.output_dir}")

    try:
        # Step 1: Analyze dataset
        class_counts, total_images = analyze_dataset(args.source_dir)

        # Check minimum requirements
        min_images_per_class = 50  # Minimum for meaningful splits
        insufficient_classes = [cls for cls, count in class_counts.items() if count < min_images_per_class]

        if insufficient_classes:
            print(f"\n⚠️  Warning: Classes {insufficient_classes} have fewer than {min_images_per_class} images")
            print("   This may result in very small validation/test sets for these classes")

        # Step 2: Collect files by class
        print("\n🔍 Collecting files by class...")
        files_by_class = collect_files_by_class(args.source_dir)

        # Step 3: Create stratified splits
        splits = create_stratified_split(
            files_by_class,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.seed
        )

        # Step 4: Copy files to split directories
        total_copied = copy_files_to_splits(splits, args.output_dir, args.source_dir)

        # Step 5: Verify splits
        verify_splits(args.output_dir)

        print("\n" + "="*60)
        print("✅ DATASET SPLITTING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📊 Total files processed: {total_copied:,}")
        print(f"📁 Output directory: {args.output_dir}")
        print("\nNext steps:")
        print("1. Verify the splits look correct")
        print("2. Update your training script to use the new dataset structure")
        print("3. Consider using class weights during training due to imbalance")

        # Generate sample training command
        print(f"\n🚀 Sample training command:")
        print(f"python ensemble_trainer.py --dataset_path {args.output_dir} --enable_class_weights")

        return 0

    except Exception as e:
        print(f"\n❌ Error during dataset splitting: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())