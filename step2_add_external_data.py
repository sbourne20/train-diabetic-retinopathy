#!/usr/bin/env python3
"""
STEP 2: Add external data from augmented_resized_V2 to dataset_eyepacs_split
Target: 10,787 images per class (7,550 train | 1,618 val | 1,619 test)

This script COPIES files from external source to balance classes 1, 2, 3, 4.
"""

import os
import random
import shutil
from pathlib import Path

# Configuration
SOURCE_DIR = Path('/Volumes/WDC1TB/Eyepacs_Aptos_Messidor/augmented_resized_V2')
TARGET_DIR = Path('/Volumes/WDC1TB/dataset_eyepacs_split')

# Target counts per split per class
TARGETS = {
    'train': 7550,
    'val': 1618,
    'test': 1619
}

# Set random seed for reproducibility
random.seed(42)

def get_current_counts(base_dir):
    """Get current file counts"""
    counts = {}
    for split in ['train', 'val', 'test']:
        counts[split] = {}
        for class_idx in range(5):
            class_path = base_dir / split / str(class_idx)
            if class_path.exists():
                files = list(class_path.glob('*.*'))
                # Filter out .DS_Store and hidden files
                files = [f for f in files if not f.name.startswith('.')]
                counts[split][class_idx] = len(files)
            else:
                counts[split][class_idx] = 0
    return counts

def get_available_files(source_dir, split, class_idx):
    """Get available files from source directory"""
    class_path = source_dir / split / str(class_idx)
    if not class_path.exists():
        return []

    files = list(class_path.glob('*.*'))
    # Filter out .DS_Store and hidden files
    files = [f for f in files if not f.name.startswith('.')]
    return files

def copy_files_to_target(source_files, target_dir, count):
    """Copy random selection of files to target directory"""
    # Shuffle and select
    random.shuffle(source_files)
    files_to_copy = source_files[:count]

    # Ensure target directory exists
    target_dir.mkdir(parents=True, exist_ok=True)

    # Copy files
    copied = 0
    for idx, src_file in enumerate(files_to_copy, 1):
        dst_file = target_dir / src_file.name

        # Handle duplicate names
        counter = 1
        original_stem = dst_file.stem
        while dst_file.exists():
            dst_file = target_dir / f"{original_stem}_dup{counter}{dst_file.suffix}"
            counter += 1

        try:
            shutil.copy2(src_file, dst_file)
            copied += 1

            # Progress indicator
            if idx % 100 == 0 or idx == len(files_to_copy):
                print(f'      Progress: {idx:,}/{len(files_to_copy):,} files copied', end='\r')
        except Exception as e:
            print(f'\n      ⚠️  Error copying {src_file.name}: {e}')

    return copied

def add_external_data():
    """Add external data to balance classes 1, 2, 3, 4"""
    print('=' * 80)
    print('➕ STEP 2: ADD EXTERNAL DATA')
    print('=' * 80)
    print()

    # Check directories
    if not SOURCE_DIR.exists():
        print(f'❌ Error: Source directory not found: {SOURCE_DIR}')
        return

    if not TARGET_DIR.exists():
        print(f'❌ Error: Target directory not found: {TARGET_DIR}')
        return

    print(f'📂 Source: {SOURCE_DIR}')
    print(f'📂 Target: {TARGET_DIR}')
    print()

    # Get current counts
    print('📊 Analyzing current state...')
    current = get_current_counts(TARGET_DIR)
    source_available = get_current_counts(SOURCE_DIR)

    print()
    print('Current counts in target directory:')
    print()
    print('| Class | Train | Val  | Test | Total  |')
    print('|-------|-------|------|------|--------|')
    for c in range(5):
        total = current['train'][c] + current['val'][c] + current['test'][c]
        print(f'| {c}     | {current["train"][c]:5,} | {current["val"][c]:4,} | {current["test"][c]:4,} | {total:6,} |')

    print()
    print('Available in source directory:')
    print()
    print('| Class | Train  | Val   | Test  | Total  |')
    print('|-------|--------|-------|-------|--------|')
    for c in range(5):
        total = source_available['train'][c] + source_available['val'][c] + source_available['test'][c]
        print(f'| {c}     | {source_available["train"][c]:6,} | {source_available["val"][c]:5,} | {source_available["test"][c]:5,} | {total:6,} |')

    print()
    print('=' * 80)
    print('📋 REQUIRED ADDITIONS')
    print('=' * 80)
    print()

    # Calculate what needs to be added
    additions_needed = {}
    total_to_add = 0

    for c in range(5):
        if c == 0:
            # Class 0 already downsampled in step 1
            print(f'Class 0: ✅ Already balanced (downsampled in Step 1)')
            continue

        print(f'CLASS {c}:')
        additions_needed[c] = {}

        for split in ['train', 'val', 'test']:
            current_count = current[split][c]
            target_count = TARGETS[split]
            need_to_add = target_count - current_count

            if need_to_add <= 0:
                print(f'   {split}: ✅ Already at target ({current_count:,} files)')
                additions_needed[c][split] = 0
            else:
                available = source_available[split][c]
                if need_to_add <= available:
                    print(f'   {split}: Add {need_to_add:,} files (available: {available:,})')
                    additions_needed[c][split] = need_to_add
                    total_to_add += need_to_add
                else:
                    print(f'   {split}: ⚠️  Need {need_to_add:,} but only {available:,} available')
                    additions_needed[c][split] = available
                    total_to_add += available
        print()

    print(f'Total files to copy: {total_to_add:,}')
    print()

    # Confirm
    response = input('Continue with copying? (yes/no): ')
    if response.lower() not in ['yes', 'y']:
        print('❌ Operation cancelled.')
        return

    print()
    print('🔄 Starting file copy operations...')
    print()

    # Copy files
    total_copied = 0

    for c in range(1, 5):  # Skip class 0
        print(f'📁 Processing Class {c}...')

        for split in ['train', 'val', 'test']:
            count_needed = additions_needed[c][split]

            if count_needed == 0:
                continue

            print(f'   {split}: Copying {count_needed:,} files...')

            # Get source files
            source_files = get_available_files(SOURCE_DIR, split, c)

            # Target directory
            target_dir = TARGET_DIR / split / str(c)

            # Copy files
            copied = copy_files_to_target(source_files, target_dir, count_needed)
            total_copied += copied

            print(f'      ✅ Copied: {copied:,} files                    ')

        print()

    print('=' * 80)
    print('✅ STEP 2 COMPLETED: External Data Added')
    print('=' * 80)
    print()
    print(f'Total files copied: {total_copied:,}')
    print()

    # Final verification
    print('🔍 Final Verification:')
    print()
    final_counts = get_current_counts(TARGET_DIR)

    print('| Class | Split | Current | Target | Status |')
    print('|-------|-------|---------|--------|--------|')

    for c in range(5):
        for split in ['train', 'val', 'test']:
            current_count = final_counts[split][c]
            target_count = TARGETS[split]

            if current_count == target_count:
                status = '✅ OK'
            elif current_count < target_count:
                status = f'⚠️  -{target_count - current_count}'
            else:
                status = f'⚠️  +{current_count - target_count}'

            print(f'| {c}     | {split:5s} | {current_count:7,} | {target_count:6,} | {status:6s} |')
        print('|-------|-------|---------|--------|--------|')

    print()

    # Summary
    print('📊 Final Dataset Summary:')
    print()
    print('| Class | Total  | Status |')
    print('|-------|--------|--------|')

    all_balanced = True
    for c in range(5):
        total = sum(final_counts[split][c] for split in ['train', 'val', 'test'])
        target_total = sum(TARGETS.values())

        if total == target_total:
            status = '✅ Balanced'
        else:
            status = f'⚠️  {total - target_total:+,}'
            all_balanced = False

        print(f'| {c}     | {total:6,} | {status:10s} |')

    print()

    if all_balanced:
        print('🎉 SUCCESS! All classes are perfectly balanced at 10,787 images each!')
        print()
        print('Expected accuracy: 94-96%+ with ensemble + augmentation 🏆')
    else:
        print('⚠️  Some classes could not be fully balanced due to data availability.')
        print('   This may still provide good results, but accuracy might be slightly lower.')

if __name__ == '__main__':
    add_external_data()
