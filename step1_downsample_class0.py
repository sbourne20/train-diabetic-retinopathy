#!/usr/bin/env python3
"""
STEP 1: Downsample Class 0 in /Volumes/WDC1TB/dataset_eyepacs_split
Target: 10,787 images per class (7,550 train | 1,618 val | 1,619 test)

This script REMOVES excess Class 0 files to balance the dataset.
"""

import os
import random
from pathlib import Path
import shutil

# Configuration
BASE_DIR = Path('/Volumes/WDC1TB/dataset_eyepacs_split')
TARGET_CLASS = 0
TARGETS = {
    'train': 7550,
    'val': 1618,
    'test': 1619
}

# Set random seed for reproducibility
random.seed(42)

def get_current_counts():
    """Get current file counts for Class 0"""
    counts = {}
    for split in ['train', 'val', 'test']:
        class_path = BASE_DIR / split / str(TARGET_CLASS)
        if class_path.exists():
            files = list(class_path.glob('*.*'))
            counts[split] = len(files)
        else:
            counts[split] = 0
    return counts

def downsample_class0():
    """Downsample Class 0 to target counts"""
    print('=' * 80)
    print('🗑️  STEP 1: DOWNSAMPLE CLASS 0')
    print('=' * 80)
    print()

    # Check base directory
    if not BASE_DIR.exists():
        print(f'❌ Error: Base directory not found: {BASE_DIR}')
        return

    print(f'📂 Working directory: {BASE_DIR}')
    print()

    # Get current counts
    current = get_current_counts()

    print('📊 CURRENT vs TARGET (Class 0 only):')
    print()
    print('| Split | Current | Target | To Remove |')
    print('|-------|---------|--------|-----------|')

    total_to_remove = 0
    for split in ['train', 'val', 'test']:
        current_count = current[split]
        target_count = TARGETS[split]
        to_remove = current_count - target_count
        total_to_remove += to_remove
        print(f'| {split:5s} | {current_count:7,} | {target_count:6,} | {to_remove:9,} |')

    print()
    print(f'Total to remove: {total_to_remove:,} files')
    print()

    # Confirm
    response = input('⚠️  This will PERMANENTLY DELETE files. Continue? (yes/no): ')
    if response.lower() not in ['yes', 'y']:
        print('❌ Operation cancelled.')
        return

    print()
    print('🔄 Starting downsampling...')
    print()

    # Process each split
    for split in ['train', 'val', 'test']:
        class_path = BASE_DIR / split / str(TARGET_CLASS)
        current_count = current[split]
        target_count = TARGETS[split]
        to_remove = current_count - target_count

        if to_remove <= 0:
            print(f'✅ {split}: Already at target ({current_count} files)')
            continue

        print(f'🗑️  {split}: Removing {to_remove:,} files...')

        # Get all files
        all_files = list(class_path.glob('*.*'))

        # Shuffle and select files to remove
        random.shuffle(all_files)
        files_to_remove = all_files[:to_remove]

        # Remove files
        removed_count = 0
        for idx, file_path in enumerate(files_to_remove, 1):
            try:
                file_path.unlink()
                removed_count += 1

                # Progress indicator
                if idx % 500 == 0 or idx == len(files_to_remove):
                    print(f'   Progress: {idx:,}/{len(files_to_remove):,} files removed', end='\r')
            except Exception as e:
                print(f'\n   ⚠️  Error removing {file_path.name}: {e}')

        print(f'   ✅ Removed: {removed_count:,} files                    ')

    print()
    print('=' * 80)
    print('✅ STEP 1 COMPLETED: Class 0 Downsampling')
    print('=' * 80)
    print()

    # Verify
    print('🔍 Verification:')
    print()
    final_counts = get_current_counts()
    print('| Split | Target | Final | Status |')
    print('|-------|--------|-------|--------|')
    for split in ['train', 'val', 'test']:
        target = TARGETS[split]
        final = final_counts[split]
        status = '✅ OK' if final == target else f'⚠️  {final - target:+d}'
        print(f'| {split:5s} | {target:6,} | {final:5,} | {status:6s} |')

    print()
    print('Next step: Run step2_add_external_data.py to add files from external source')

if __name__ == '__main__':
    downsample_class0()
