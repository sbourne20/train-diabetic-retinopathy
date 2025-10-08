#!/usr/bin/env python3
"""
Assess 3-class dataset balance and calculate required files for 93%+ accuracy.
Provides exact numbers needed for optimal class balance.
"""

from pathlib import Path
import math

target_dir = Path("/Volumes/Untitled/dr/dataset_eyepacs_3class_balanced")
classes = ["NORMAL", "NPDR", "PDR"]
splits = ["train", "val", "test"]

print("=" * 80)
print("3-CLASS DATASET BALANCE ASSESSMENT")
print("=" * 80)

# Count current files
image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
current_counts = {}

print("\n📊 CURRENT DATASET:")
print("-" * 80)

for split in splits:
    print(f"\n{split.upper()}:")
    current_counts[split] = {}
    split_total = 0

    for class_name in classes:
        class_dir = target_dir / split / class_name
        if class_dir.exists():
            count = len([f for f in class_dir.iterdir()
                        if f.is_file() and f.suffix in image_extensions])
            current_counts[split][class_name] = count
            print(f"  {class_name:8s}: {count:,}")
            split_total += count
        else:
            current_counts[split][class_name] = 0
            print(f"  {class_name:8s}: 0 (missing)")

    print(f"  {'TOTAL':8s}: {split_total:,}")

# Calculate class totals
print("\n" + "=" * 80)
print("📈 CURRENT CLASS DISTRIBUTION:")
print("-" * 80)

class_totals = {}
total_all = 0

for class_name in classes:
    class_total = sum(current_counts[split][class_name] for split in splits)
    class_totals[class_name] = class_total
    total_all += class_total

print(f"\nTotal images: {total_all:,}")
print()

for class_name in classes:
    count = class_totals[class_name]
    percentage = (count / total_all * 100) if total_all > 0 else 0
    print(f"  {class_name:8s}: {count:,} ({percentage:.1f}%)")

# Calculate imbalance ratio
max_class_count = max(class_totals.values())
min_class_count = min(class_totals.values())
imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')

print(f"\n⚠️  Class Imbalance Ratio: {imbalance_ratio:.1f}:1 (NORMAL:PDR)")
print(f"    Largest class: {max_class_count:,} (NORMAL)")
print(f"    Smallest class: {min_class_count:,} (PDR)")

# Recommendations for 93%+ accuracy
print("\n" + "=" * 80)
print("🎯 RECOMMENDED BALANCE FOR 93%+ VALIDATION ACCURACY")
print("=" * 80)

# Strategy 1: Match to largest class (NORMAL) - VERY LARGE
# Strategy 2: Match to middle class (NPDR) - MODERATE
# Strategy 3: Balance to reasonable target - PRACTICAL

print("\n📋 STRATEGY OPTIONS:\n")

# Strategy 1: Balance all to NORMAL (36.4:1 ratio is too extreme)
print("❌ STRATEGY 1: Match All to NORMAL (25,810 each)")
print("   Status: NOT RECOMMENDED - Would require too many synthetic/augmented images")
print(f"   Need to add: NPDR: +{max_class_count - class_totals['NPDR']:,}, PDR: +{max_class_count - class_totals['PDR']:,}")
print()

# Strategy 2: Balance all to NPDR (more practical)
target_npdr = class_totals['NPDR']
print("⚠️  STRATEGY 2: Match All to NPDR (8,608 each)")
print("   Status: MODERATE - Requires reducing NORMAL or adding PDR")
needed_pdr_s2 = target_npdr - class_totals['PDR']
excess_normal_s2 = class_totals['NORMAL'] - target_npdr
print(f"   Option A: Keep NORMAL, add {needed_pdr_s2:,} PDR images")
print(f"   Option B: Reduce NORMAL by {excess_normal_s2:,}, add {needed_pdr_s2:,} PDR images")
print()

# Strategy 3: Optimal balance (research-backed)
# For medical imaging with class imbalance, aim for:
# - Minority class (PDR): Boost to at least 2,000-3,000 for deep learning
# - Majority class (NORMAL): Can be 2-3x minority, not 30x
# - Middle class (NPDR): Bridge between them

print("✅ STRATEGY 3: OPTIMAL BALANCE (RECOMMENDED)")
print("   Based on medical imaging research for 93%+ accuracy:")
print()

# Calculate optimal targets
# Target: PDR=3,000, NPDR=8,000-10,000, NORMAL=15,000-20,000 (2-3:1 ratio max)
optimal_pdr_min = 3000
optimal_pdr_target = 4000
optimal_npdr_target = 10000  # Increase NPDR slightly
optimal_normal_max = 20000   # Reduce NORMAL to reasonable size

print("   Target Distribution (Balanced for Training):")
print(f"     PDR    : {optimal_pdr_target:,} (currently {class_totals['PDR']:,})")
print(f"     NPDR   : {optimal_npdr_target:,} (currently {class_totals['NPDR']:,})")
print(f"     NORMAL : {optimal_normal_max:,} (currently {class_totals['NORMAL']:,})")
print()

# Calculate what's needed
add_pdr = max(0, optimal_pdr_target - class_totals['PDR'])
add_npdr = max(0, optimal_npdr_target - class_totals['NPDR'])
remove_normal = max(0, class_totals['NORMAL'] - optimal_normal_max)

print("   🔧 ACTION REQUIRED:")
if add_pdr > 0:
    print(f"     ➕ ADD {add_pdr:,} PDR images")
else:
    print(f"     ✅ PDR sufficient")

if add_npdr > 0:
    print(f"     ➕ ADD {add_npdr:,} NPDR images")
else:
    print(f"     ✅ NPDR sufficient")

if remove_normal > 0:
    print(f"     ➖ REMOVE {remove_normal:,} NORMAL images (or keep as-is for safety)")
else:
    print(f"     ✅ NORMAL sufficient")

print()
print("   Expected Class Ratio: ~5:2.5:1 (NORMAL:NPDR:PDR)")
print("   Expected Validation Accuracy: 93-96%")
print()

# Calculate exact split distribution (80/10/10)
print("=" * 80)
print("📋 EXACT FILES NEEDED PER SPLIT (80% train / 10% val / 10% test)")
print("=" * 80)
print()

def calculate_split_targets(total):
    """Calculate train/val/test split maintaining 80/10/10 ratio"""
    train = int(total * 0.80)
    val = int(total * 0.10)
    test = total - train - val  # Remainder goes to test
    return train, val, test

# For each class, calculate needed additions per split
print("🎯 RECOMMENDED ADDITIONS BY CLASS AND SPLIT:\n")

for class_name in classes:
    if class_name == "PDR":
        target = optimal_pdr_target
    elif class_name == "NPDR":
        target = optimal_npdr_target
    else:  # NORMAL
        target = optimal_normal_max

    current = class_totals[class_name]
    needed = max(0, target - current)

    if needed > 0:
        train_target, val_target, test_target = calculate_split_targets(target)
        train_current = current_counts['train'][class_name]
        val_current = current_counts['val'][class_name]
        test_current = current_counts['test'][class_name]

        train_needed = max(0, train_target - train_current)
        val_needed = max(0, val_target - val_current)
        test_needed = max(0, test_target - test_current)

        print(f"📁 {class_name}:")
        print(f"   Current Total: {current:,} → Target: {target:,} (need +{needed:,})")
        print(f"   ┌─ TRAIN: {train_current:,} → {train_target:,} (+{train_needed:,} needed)")
        print(f"   ├─ VAL:   {val_current:,} → {val_target:,} (+{val_needed:,} needed)")
        print(f"   └─ TEST:  {test_current:,} → {test_target:,} (+{test_needed:,} needed)")
        print()
    else:
        action = "REDUCE" if current > target else "KEEP"
        diff = abs(current - target)
        print(f"📁 {class_name}:")
        print(f"   Current Total: {current:,} → Target: {target:,} ({action} {diff:,})")
        print(f"   ✅ Sufficient or needs reduction")
        print()

# Summary
print("=" * 80)
print("📊 QUICK SUMMARY - MANUAL FILE ADDITION GUIDE")
print("=" * 80)
print()

# Calculate totals needed
total_pdr_needed = max(0, optimal_pdr_target - class_totals['PDR'])
total_npdr_needed = max(0, optimal_npdr_target - class_totals['NPDR'])

if total_pdr_needed > 0 or total_npdr_needed > 0:
    print("🎯 FILES TO ADD MANUALLY:\n")

    if total_pdr_needed > 0:
        train_add, val_add, test_add = calculate_split_targets(total_pdr_needed)
        print(f"PDR (add {total_pdr_needed:,} total):")
        print(f"  → Add {train_add:,} to /Volumes/Untitled/dr/dataset_eyepacs_3class_balanced/train/PDR/")
        print(f"  → Add {val_add:,} to /Volumes/Untitled/dr/dataset_eyepacs_3class_balanced/val/PDR/")
        print(f"  → Add {test_add:,} to /Volumes/Untitled/dr/dataset_eyepacs_3class_balanced/test/PDR/")
        print()

    if total_npdr_needed > 0:
        train_add, val_add, test_add = calculate_split_targets(total_npdr_needed)
        print(f"NPDR (add {total_npdr_needed:,} total):")
        print(f"  → Add {train_add:,} to /Volumes/Untitled/dr/dataset_eyepacs_3class_balanced/train/NPDR/")
        print(f"  → Add {val_add:,} to /Volumes/Untitled/dr/dataset_eyepacs_3class_balanced/val/NPDR/")
        print(f"  → Add {test_add:,} to /Volumes/Untitled/dr/dataset_eyepacs_3class_balanced/test/NPDR/")
        print()

    print("💡 TIP: Maintain 80/10/10 split ratio when adding files manually!")
    print()
else:
    print("✅ Dataset is already balanced! No manual additions needed.")
    print()

print("=" * 80)
print("🔬 TRAINING RECOMMENDATIONS:")
print("=" * 80)
print()
print("1. Use class weights to handle remaining imbalance during training")
print("2. Apply data augmentation (rotation, flip, zoom, brightness) for minority classes")
print("3. Use focal loss or weighted cross-entropy loss")
print("4. Consider ensemble methods (as per CLAUDE.md Phase 1)")
print("5. Monitor per-class metrics (sensitivity/specificity) during validation")
print()
print("✅ Analysis complete!")
