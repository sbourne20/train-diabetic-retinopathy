#!/usr/bin/env python3
"""
Dataset Validation Script
Diagnoses issues with augmented_resized_V2_balanced dataset
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict

def validate_dataset(dataset_path):
    """Validate dataset quality and identify issues"""

    print("="*80)
    print("DATASET VALIDATION DIAGNOSTIC")
    print("="*80)
    print(f"\nDataset path: {dataset_path}\n")

    issues = defaultdict(list)
    stats = {
        'total_images': 0,
        'corrupted': 0,
        'too_small': 0,
        'wrong_mode': 0,
        'zero_variance': 0,
        'macOS_metadata': 0
    }

    splits = ['train', 'val', 'test']
    classes = [0, 1, 2, 3, 4]

    for split in splits:
        print(f"\n{'='*80}")
        print(f"Validating {split.upper()} set")
        print(f"{'='*80}\n")

        split_path = Path(dataset_path) / split

        if not split_path.exists():
            print(f"❌ {split} directory not found!")
            continue

        for cls in classes:
            class_path = split_path / str(cls)
            if not class_path.exists():
                print(f"❌ Class {cls} directory not found!")
                continue

            # Get all jpg files
            image_files = list(class_path.glob('*.jpg'))

            # Filter out macOS metadata files
            real_images = [f for f in image_files if not f.name.startswith('._')]
            metadata_files = [f for f in image_files if f.name.startswith('._')]

            stats['macOS_metadata'] += len(metadata_files)

            print(f"Class {cls}: {len(real_images)} images ({len(metadata_files)} metadata files)")

            # Sample 10 images for detailed validation
            sample_size = min(10, len(real_images))
            samples = real_images[:sample_size]

            for img_path in samples:
                stats['total_images'] += 1

                try:
                    # Try to load image
                    img = Image.open(img_path)

                    # Check image properties
                    width, height = img.size
                    mode = img.mode

                    # Check for issues
                    if width < 100 or height < 100:
                        issues[f'{split}_class_{cls}'].append(f"Too small: {img_path.name} ({width}x{height})")
                        stats['too_small'] += 1

                    if mode not in ['RGB', 'L']:
                        issues[f'{split}_class_{cls}'].append(f"Wrong mode: {img_path.name} (mode: {mode})")
                        stats['wrong_mode'] += 1

                    # Convert to array and check variance
                    img_array = np.array(img)

                    if img_array.var() < 1.0:
                        issues[f'{split}_class_{cls}'].append(f"Zero variance (blank?): {img_path.name}")
                        stats['zero_variance'] += 1

                    # Check if image is mostly black or white
                    mean_intensity = img_array.mean()
                    if mean_intensity < 10 or mean_intensity > 245:
                        issues[f'{split}_class_{cls}'].append(f"Extreme intensity: {img_path.name} (mean: {mean_intensity:.1f})")

                except Exception as e:
                    issues[f'{split}_class_{cls}'].append(f"Corrupted: {img_path.name} - {str(e)}")
                    stats['corrupted'] += 1

    # Print summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    print(f"\nTotal images checked: {stats['total_images']}")
    print(f"macOS metadata files: {stats['macOS_metadata']} (ignored)")
    print(f"\nIssues found:")
    print(f"  Corrupted images: {stats['corrupted']}")
    print(f"  Too small (<100px): {stats['too_small']}")
    print(f"  Wrong color mode: {stats['wrong_mode']}")
    print(f"  Zero variance (blank): {stats['zero_variance']}")

    if issues:
        print("\n" + "="*80)
        print("DETAILED ISSUES BY CLASS")
        print("="*80)
        for key, issue_list in issues.items():
            if issue_list:
                print(f"\n{key}:")
                for issue in issue_list[:5]:  # Show first 5
                    print(f"  • {issue}")
                if len(issue_list) > 5:
                    print(f"  ... and {len(issue_list) - 5} more issues")

    # Check a few sample images in detail
    print("\n" + "="*80)
    print("SAMPLE IMAGE ANALYSIS (first 3 images from train/class_0)")
    print("="*80)

    train_class0 = Path(dataset_path) / 'train' / '0'
    if train_class0.exists():
        samples = [f for f in train_class0.glob('*.jpg') if not f.name.startswith('._')][:3]

        for img_path in samples:
            try:
                img = Image.open(img_path)
                img_array = np.array(img)

                print(f"\n{img_path.name}:")
                print(f"  Size: {img.size}")
                print(f"  Mode: {img.mode}")
                print(f"  Array shape: {img_array.shape}")
                print(f"  Data type: {img_array.dtype}")
                print(f"  Value range: [{img_array.min()}, {img_array.max()}]")
                print(f"  Mean: {img_array.mean():.2f}")
                print(f"  Std: {img_array.std():.2f}")
                print(f"  Variance: {img_array.var():.2f}")
            except Exception as e:
                print(f"\n{img_path.name}: ❌ ERROR - {e}")

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    if stats['corrupted'] > 0 or stats['zero_variance'] > 10:
        print("\n❌ DATASET HAS SERIOUS ISSUES!")
        print("\nPossible causes:")
        print("  1. Augmentation process corrupted images")
        print("  2. Images are resized incorrectly")
        print("  3. CLAHE preprocessing failed")
        print("  4. File transfer corruption")
        print("\nSuggested actions:")
        print("  1. Use original imbalanced dataset (./dataset_eyepacs)")
        print("  2. Re-create balanced dataset with proper validation")
        print("  3. Check augmentation script for bugs")
    else:
        print("\n✅ Dataset appears structurally sound")
        print("\nIf training still fails, the issue may be:")
        print("  1. Label mismatch (images in wrong class folders)")
        print("  2. Augmentation artifacts affecting features")
        print("  3. Model initialization problem")

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = './augmented_resized_V2_balanced'

    validate_dataset(dataset_path)
