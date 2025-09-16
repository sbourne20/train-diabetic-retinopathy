#!/usr/bin/env python3
"""
Dataset Balancer: Copy samples from dataset5 to balance dataset6
Improve class balance from 16.2:1 to acceptable medical standards (≤5:1)
"""

import os
import shutil
import random
from pathlib import Path
import argparse
from collections import defaultdict
import json

class DatasetBalancer:
    """Balance dataset6 using samples from dataset5"""

    def __init__(self, source_dataset='./dataset5', target_dataset='./dataset6'):
        self.source_path = Path(source_dataset)
        self.target_path = Path(target_dataset)
        self.balance_target = 'good'  # Target: ≤5:1 ratio
        self.target_standards = {
            'excellent': 3.0,  # ≤3:1 ratio
            'good': 5.0,       # ≤5:1 ratio
            'acceptable': 10.0  # ≤10:1 ratio
        }

    def analyze_current_state(self):
        """Analyze current class distribution in both datasets"""
        print("\n" + "="*70)
        print("📊 ANALYZING CURRENT CLASS DISTRIBUTION")
        print("="*70)

        # Dataset6 (target) analysis
        dataset6_counts = {}
        for split in ['train', 'val', 'test']:
            split_counts = {}
            split_path = self.target_path / split

            if split_path.exists():
                for class_id in range(5):
                    class_path = split_path / str(class_id)
                    if class_path.exists():
                        count = len([f for f in class_path.iterdir()
                                   if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
                        split_counts[class_id] = count
                    else:
                        split_counts[class_id] = 0

            dataset6_counts[split] = split_counts

        # Dataset5 (source) analysis
        dataset5_counts = {}
        for split in ['train', 'val', 'test']:
            split_counts = {}
            split_path = self.source_path / split

            if split_path.exists():
                for class_id in range(5):
                    class_path = split_path / str(class_id)
                    if class_path.exists():
                        count = len([f for f in class_path.iterdir()
                                   if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
                        split_counts[class_id] = count
                    else:
                        split_counts[class_id] = 0

            dataset5_counts[split] = split_counts

        # Print analysis
        print(f"\n📁 DATASET6 (TARGET) - Current State:")
        overall_d6 = defaultdict(int)
        for split, counts in dataset6_counts.items():
            print(f"   {split.upper()}:")
            for class_id, count in counts.items():
                print(f"     Class {class_id}: {count:,} images")
                overall_d6[class_id] += count

        print(f"\n📊 Dataset6 Overall Totals:")
        d6_total = sum(overall_d6.values())
        for class_id, count in overall_d6.items():
            percentage = (count / d6_total * 100) if d6_total > 0 else 0
            print(f"   Class {class_id}: {count:,} images ({percentage:.1f}%)")

        # Calculate current balance ratio
        d6_counts = list(overall_d6.values())
        current_ratio = max(d6_counts) / min(d6_counts) if min(d6_counts) > 0 else float('inf')
        print(f"\n⚖️ Current Balance Ratio: {current_ratio:.1f}:1")

        print(f"\n📁 DATASET5 (SOURCE) - Available Samples:")
        overall_d5 = defaultdict(int)
        for split, counts in dataset5_counts.items():
            for class_id, count in counts.items():
                overall_d5[class_id] += count

        for class_id, count in overall_d5.items():
            print(f"   Class {class_id}: {count:,} images available")

        return dataset6_counts, dataset5_counts, overall_d6, overall_d5, current_ratio

    def calculate_balance_strategy(self, d6_counts, d5_available, target_standard='good'):
        """Calculate how many samples to copy from each class"""
        print("\n" + "="*70)
        print("🎯 CALCULATING BALANCE STRATEGY")
        print("="*70)

        target_ratio = self.target_standards[target_standard]
        print(f"Target: {target_standard.upper()} balance (≤{target_ratio}:1 ratio)")

        # Current state
        current_min = min(d6_counts.values())
        current_max = max(d6_counts.values())

        print(f"\nCurrent state:")
        print(f"   Minimum class: {current_min:,} samples")
        print(f"   Maximum class: {current_max:,} samples")
        print(f"   Current ratio: {current_max/current_min:.1f}:1")

        # Strategy 1: Bring all classes to at least min_target
        # Strategy 2: Cap maximum class if needed

        # For medical balance, we want minority classes to have sufficient samples
        # Target: Bring smallest classes up to reasonable levels

        # Calculate targets
        strategy = {}

        # Find the target minimum based on our standards
        # For 'good' balance (5:1), if max class has 9,201, min should have ~1,840
        # But we want medical significance, so let's target higher

        # Medical-grade targets per class
        medical_targets = {
            'research': 1000,
            'clinical': 2000,
            'production': 3000
        }

        min_target = medical_targets['clinical']  # 2000 per class minimum
        max_allowed = min_target * target_ratio   # 10000 max for 5:1 ratio

        print(f"\nBalance Strategy:")
        print(f"   Target minimum per class: {min_target:,} samples")
        print(f"   Maximum allowed per class: {max_allowed:,} samples")

        copy_plan = {}

        for class_id, current_count in d6_counts.items():
            available = d5_available.get(class_id, 0)

            if current_count < min_target:
                # Need to add samples
                needed = min_target - current_count
                can_copy = min(needed, available)
                copy_plan[class_id] = can_copy

                print(f"   Class {class_id}: Need {needed:,}, can copy {can_copy:,} from dataset5")
            else:
                copy_plan[class_id] = 0
                print(f"   Class {class_id}: Sufficient ({current_count:,} samples)")

        # Calculate final balance ratio
        final_counts = {}
        for class_id in range(5):
            final_counts[class_id] = d6_counts[class_id] + copy_plan[class_id]

        final_min = min(final_counts.values())
        final_max = max(final_counts.values())
        final_ratio = final_max / final_min if final_min > 0 else float('inf')

        print(f"\nProjected final state:")
        for class_id, count in final_counts.items():
            increase = copy_plan[class_id]
            print(f"   Class {class_id}: {count:,} samples (+{increase:,})")

        print(f"\nFinal balance ratio: {final_ratio:.1f}:1")

        if final_ratio <= self.target_standards['excellent']:
            grade = "✅ EXCELLENT"
        elif final_ratio <= self.target_standards['good']:
            grade = "🟡 GOOD"
        elif final_ratio <= self.target_standards['acceptable']:
            grade = "⚠️ ACCEPTABLE"
        else:
            grade = "❌ STILL POOR"

        print(f"Final grade: {grade}")

        return copy_plan, final_counts, final_ratio

    def copy_samples_intelligently(self, copy_plan):
        """Copy samples from dataset5 to dataset6 with intelligent distribution"""
        print("\n" + "="*70)
        print("📁 COPYING SAMPLES FROM DATASET5 TO DATASET6")
        print("="*70)

        total_copied = 0

        for class_id, num_to_copy in copy_plan.items():
            if num_to_copy == 0:
                continue

            print(f"\n🔄 Processing Class {class_id} - copying {num_to_copy:,} samples...")

            # Collect available files from dataset5
            available_files = []
            for split in ['train', 'val', 'test']:
                source_class_dir = self.source_path / split / str(class_id)
                if source_class_dir.exists():
                    for file_path in source_class_dir.iterdir():
                        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            available_files.append((file_path, split))

            if len(available_files) < num_to_copy:
                print(f"   ⚠️ Warning: Only {len(available_files)} files available, copying all")
                num_to_copy = len(available_files)

            # Randomly select files to copy
            random.shuffle(available_files)
            selected_files = available_files[:num_to_copy]

            # Distribute copies across train/val/test maintaining ratios
            # Dataset6 current ratios: train=70%, val=15%, test=15%
            split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}

            copies_per_split = {
                'train': int(num_to_copy * split_ratios['train']),
                'val': int(num_to_copy * split_ratios['val']),
                'test': int(num_to_copy * split_ratios['test'])
            }

            # Adjust for rounding errors
            total_allocated = sum(copies_per_split.values())
            if total_allocated < num_to_copy:
                copies_per_split['train'] += (num_to_copy - total_allocated)

            print(f"   Distribution: Train={copies_per_split['train']}, Val={copies_per_split['val']}, Test={copies_per_split['test']}")

            # Copy files
            copied_count = 0
            for target_split, target_count in copies_per_split.items():
                if target_count == 0:
                    continue

                target_dir = self.target_path / target_split / str(class_id)
                target_dir.mkdir(parents=True, exist_ok=True)

                # Select files for this split
                files_for_split = selected_files[copied_count:copied_count + target_count]

                for source_file, source_split in files_for_split:
                    # Generate unique filename to avoid conflicts
                    base_name = source_file.stem
                    extension = source_file.suffix
                    counter = 1

                    target_file = target_dir / f"{base_name}_d5{extension}"
                    while target_file.exists():
                        target_file = target_dir / f"{base_name}_d5_{counter}{extension}"
                        counter += 1

                    # Copy file
                    shutil.copy2(source_file, target_file)
                    copied_count += 1
                    total_copied += 1

                print(f"     → {target_split}: {len(files_for_split)} files copied")

        print(f"\n✅ Total files copied: {total_copied:,}")
        return total_copied

    def verify_final_balance(self):
        """Verify the final class balance after copying"""
        print("\n" + "="*70)
        print("🔍 VERIFYING FINAL BALANCE")
        print("="*70)

        final_counts = defaultdict(int)

        for split in ['train', 'val', 'test']:
            split_path = self.target_path / split
            if not split_path.exists():
                continue

            print(f"\n📁 {split.upper()}:")
            for class_id in range(5):
                class_path = split_path / str(class_id)
                if class_path.exists():
                    count = len([f for f in class_path.iterdir()
                               if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
                    final_counts[class_id] += count
                    print(f"   Class {class_id}: {count:,} images")

        print(f"\n📊 FINAL OVERALL DISTRIBUTION:")
        total_images = sum(final_counts.values())

        for class_id, count in final_counts.items():
            percentage = (count / total_images * 100) if total_images > 0 else 0
            print(f"   Class {class_id}: {count:,} images ({percentage:.1f}%)")

        # Calculate final balance ratio
        counts = list(final_counts.values())
        if min(counts) > 0:
            final_ratio = max(counts) / min(counts)
            print(f"\n⚖️ Final Balance Ratio: {final_ratio:.1f}:1")

            if final_ratio <= self.target_standards['excellent']:
                grade = "✅ EXCELLENT BALANCE"
                status = "Ready for medical-grade training"
            elif final_ratio <= self.target_standards['good']:
                grade = "🟡 GOOD BALANCE"
                status = "Suitable for high-accuracy ensemble training"
            elif final_ratio <= self.target_standards['acceptable']:
                grade = "⚠️ ACCEPTABLE BALANCE"
                status = "Requires class weights during training"
            else:
                grade = "❌ STILL IMBALANCED"
                status = "Additional balancing needed"

            print(f"🏆 {grade}")
            print(f"📝 Status: {status}")

            return final_ratio, grade, status
        else:
            print("❌ Error: Some classes have 0 samples!")
            return float('inf'), "ERROR", "Critical error"

    def run_balancing(self, target_standard='good', dry_run=False):
        """Run the complete balancing process"""
        print("🚀 DATASET BALANCING PROCESS")
        print("="*70)
        print(f"📁 Source: {self.source_path}")
        print(f"📁 Target: {self.target_path}")
        print(f"🎯 Target Standard: {target_standard.upper()}")

        if dry_run:
            print("🧪 DRY RUN MODE - No files will be copied")

        try:
            # Step 1: Analyze current state
            d6_counts, d5_counts, d6_overall, d5_overall, current_ratio = self.analyze_current_state()

            # Step 2: Calculate strategy
            copy_plan, final_counts, projected_ratio = self.calculate_balance_strategy(
                d6_overall, d5_overall, target_standard
            )

            if not dry_run:
                # Step 3: Execute copying
                total_copied = self.copy_samples_intelligently(copy_plan)

                # Step 4: Verify results
                final_ratio, grade, status = self.verify_final_balance()

                print(f"\n" + "="*70)
                print("✅ BALANCING COMPLETED!")
                print("="*70)
                print(f"📊 Files copied: {total_copied:,}")
                print(f"⚖️ Balance improved: {current_ratio:.1f}:1 → {final_ratio:.1f}:1")
                print(f"🏆 Final grade: {grade}")
                print(f"📝 Status: {status}")

                return {
                    'success': True,
                    'files_copied': total_copied,
                    'old_ratio': current_ratio,
                    'new_ratio': final_ratio,
                    'grade': grade,
                    'status': status
                }
            else:
                print(f"\n🧪 DRY RUN RESULTS:")
                print(f"⚖️ Would improve balance: {current_ratio:.1f}:1 → {projected_ratio:.1f}:1")
                total_to_copy = sum(copy_plan.values())
                print(f"📊 Would copy: {total_to_copy:,} files")

                return {
                    'success': True,
                    'dry_run': True,
                    'files_to_copy': total_to_copy,
                    'old_ratio': current_ratio,
                    'projected_ratio': projected_ratio,
                    'copy_plan': copy_plan
                }

        except Exception as e:
            print(f"\n❌ Error during balancing: {str(e)}")
            return {'success': False, 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description='Balance dataset6 using samples from dataset5')

    parser.add_argument('--source_dataset', default='./dataset5',
                       help='Source dataset path (dataset5)')
    parser.add_argument('--target_dataset', default='./dataset6',
                       help='Target dataset path (dataset6)')
    parser.add_argument('--target_standard', choices=['excellent', 'good', 'acceptable'],
                       default='good', help='Target balance standard')
    parser.add_argument('--dry_run', action='store_true',
                       help='Dry run - analyze only, no files copied')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible file selection')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Validate paths
    if not Path(args.source_dataset).exists():
        print(f"❌ Error: Source dataset '{args.source_dataset}' not found!")
        return 1

    if not Path(args.target_dataset).exists():
        print(f"❌ Error: Target dataset '{args.target_dataset}' not found!")
        return 1

    # Run balancing
    balancer = DatasetBalancer(args.source_dataset, args.target_dataset)
    balancer.balance_target = args.target_standard

    result = balancer.run_balancing(args.target_standard, args.dry_run)

    if result['success']:
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit(main())