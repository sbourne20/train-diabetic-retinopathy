#!/usr/bin/env python3
"""
Check NPDR shortage and recommend where to add remaining files.
"""

from pathlib import Path

target_dir = Path("/Volumes/Untitled/dr/dataset_eyepacs_3class_balanced")
image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

print("=" * 80)
print("NPDR CLASS ANALYSIS - SHORTAGE ASSESSMENT")
print("=" * 80)

# Target for NPDR: 10,000 total (80% train, 10% val, 10% test)
target_total = 10000
target_train = int(target_total * 0.80)  # 8,000
target_val = int(target_total * 0.10)    # 1,000
target_test = target_total - target_train - target_val  # 1,000

splits = ['train', 'val', 'test']
current_counts = {}

print("\n📊 CURRENT NPDR DISTRIBUTION:\n")

total_current = 0
for split in splits:
    class_dir = target_dir / split / "NPDR"
    count = len([f for f in class_dir.iterdir()
                if f.is_file() and f.suffix in image_extensions])
    current_counts[split] = count
    total_current += count

print(f"  Train: {current_counts['train']:,} / {target_train:,} (target)")
print(f"  Val:   {current_counts['val']:,} / {target_val:,} (target)")
print(f"  Test:  {current_counts['test']:,} / {target_test:,} (target)")
print(f"  {'─' * 40}")
print(f"  TOTAL: {total_current:,} / {target_total:,} (target)")

shortage = target_total - total_current
print(f"\n⚠️  Shortage: {shortage} files")

if shortage > 0:
    # Calculate how many needed per split to maintain 80/10/10
    train_shortage = target_train - current_counts['train']
    val_shortage = target_val - current_counts['val']
    test_shortage = target_test - current_counts['test']

    print("\n" + "=" * 80)
    print("🎯 RECOMMENDED DISTRIBUTION FOR REMAINING 20 FILES")
    print("=" * 80)
    print()

    # Option 1: Strict 80/10/10 ratio
    print("OPTION 1: Maintain Perfect 80/10/10 Ratio")
    print("-" * 80)

    if shortage == 20:
        # For 20 files: 16 train, 2 val, 2 test
        opt1_train = int(shortage * 0.80)  # 16
        opt1_val = int(shortage * 0.10)    # 2
        opt1_test = shortage - opt1_train - opt1_val  # 2

        print(f"  → Add {opt1_train} to train/NPDR/  ({current_counts['train']} → {current_counts['train'] + opt1_train})")
        print(f"  → Add {opt1_val} to val/NPDR/    ({current_counts['val']} → {current_counts['val'] + opt1_val})")
        print(f"  → Add {opt1_test} to test/NPDR/   ({current_counts['test']} → {current_counts['test'] + opt1_test})")

        final_train = current_counts['train'] + opt1_train
        final_val = current_counts['val'] + opt1_val
        final_test = current_counts['test'] + opt1_test

        print(f"\n  Final distribution:")
        print(f"    Train: {final_train:,} ({final_train/target_total*100:.1f}%)")
        print(f"    Val:   {final_val:,} ({final_val/target_total*100:.1f}%)")
        print(f"    Test:  {final_test:,} ({final_test/target_total*100:.1f}%)")

    print()

    # Option 2: Prioritize train set (most important for learning)
    print("OPTION 2: Prioritize Training Set (Recommended for Model Performance)")
    print("-" * 80)

    if train_shortage > 0:
        # Put all in train to reach target
        opt2_train = min(shortage, train_shortage)
        opt2_val = max(0, min(shortage - opt2_train, val_shortage))
        opt2_test = shortage - opt2_train - opt2_val

        print(f"  → Add {opt2_train} to train/NPDR/  ({current_counts['train']} → {current_counts['train'] + opt2_train})")
        if opt2_val > 0:
            print(f"  → Add {opt2_val} to val/NPDR/    ({current_counts['val']} → {current_counts['val'] + opt2_val})")
        if opt2_test > 0:
            print(f"  → Add {opt2_test} to test/NPDR/   ({current_counts['test']} → {current_counts['test'] + opt2_test})")

        final_train = current_counts['train'] + opt2_train
        final_val = current_counts['val'] + opt2_val
        final_test = current_counts['test'] + opt2_test

        print(f"\n  Final distribution:")
        print(f"    Train: {final_train:,} ({final_train/target_total*100:.1f}%)")
        print(f"    Val:   {final_val:,} ({final_val/target_total*100:.1f}%)")
        print(f"    Test:  {final_test:,} ({final_test/target_total*100:.1f}%)")

        print(f"\n  ✅ Rationale: Training set benefits most from additional data")
        print(f"     Model learns from train set, so maximizing it improves accuracy")

    print()

    # Option 3: Fill gaps to reach exact targets
    print("OPTION 3: Fill Gaps to Reach Exact Targets")
    print("-" * 80)
    print(f"  → Add {train_shortage} to train/NPDR/  (to reach {target_train})")
    print(f"  → Add {val_shortage} to val/NPDR/    (to reach {target_val})")
    print(f"  → Add {test_shortage} to test/NPDR/   (to reach {target_test})")
    print(f"\n  ⚠️  Note: This requires {train_shortage + val_shortage + test_shortage} files total")

    print("\n" + "=" * 80)
    print("💡 FINAL RECOMMENDATION")
    print("=" * 80)
    print()

    if shortage == 20:
        print("✅ For your 20 remaining NPDR files:")
        print()
        print("   BEST CHOICE: Option 1 (Maintain Perfect 80/10/10)")
        print()
        print("   📁 Add 16 files → train/NPDR/")
        print("   📁 Add 2 files  → val/NPDR/")
        print("   📁 Add 2 files  → test/NPDR/")
        print()
        print("   This maintains the ideal train/val/test ratio for optimal model training")
        print("   and ensures unbiased evaluation on validation and test sets.")
        print()
        print("   Alternative: If you prefer, add all 20 to train/NPDR/ for maximum")
        print("   training data (trade-off: slightly unbalanced split but better learning)")

else:
    print("\n✅ No shortage detected! Dataset is balanced.")

print("\n" + "=" * 80)
print("📊 COMPLETE DATASET SUMMARY (AFTER ADDING 20 FILES)")
print("=" * 80)

# Show what final dataset will look like
final_total_train = 20648 + 8000 + 3199
final_total_val = 2581 + 1000 + 399
final_total_test = 2581 + 1000 + 402

print()
print("Final dataset (assuming Option 1 - 16/2/2 split):")
print()
print(f"  TRAIN: {final_total_train + 16:,} images")
print(f"    NORMAL: 20,648")
print(f"    NPDR:   8,015")
print(f"    PDR:    3,199")
print()
print(f"  VAL:   {final_total_val + 2:,} images")
print(f"    NORMAL: 2,581")
print(f"    NPDR:   1,001")
print(f"    PDR:    399")
print()
print(f"  TEST:  {final_total_test + 2:,} images")
print(f"    NORMAL: 2,581")
print(f"    NPDR:   1,004")
print(f"    PDR:    402")
print()
print(f"  TOTAL: {final_total_train + final_total_val + final_total_test + 20:,} images")
print()
print("✅ Analysis complete!")
