#!/usr/bin/env python3
"""
Verify the 3-class balanced dataset structure and counts.
"""

from pathlib import Path

target_dir = Path("/Volumes/Untitled/dr/dataset_eyepacs_3class_balanced")
classes = ["NORMAL", "NPDR", "PDR"]
splits = ["train", "val", "test"]

print("=" * 80)
print("3-CLASS BALANCED DATASET VERIFICATION")
print("=" * 80)

image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

for split in splits:
    print(f"\n{split.upper()}:")
    split_total = 0

    for class_name in classes:
        class_dir = target_dir / split / class_name
        if not class_dir.exists():
            print(f"  {class_name:8s}: ❌ Directory not found")
            continue

        # Count image files
        count = len([f for f in class_dir.iterdir()
                    if f.is_file() and f.suffix in image_extensions])

        print(f"  {class_name:8s}: {count:,}")
        split_total += count

    print(f"  {'TOTAL':8s}: {split_total:,}")

print("\n" + "=" * 80)
print("CLASS DISTRIBUTION ACROSS SPLITS")
print("=" * 80)

for class_name in classes:
    print(f"\n{class_name}:")
    class_total = 0
    for split in splits:
        class_dir = target_dir / split / class_name
        if class_dir.exists():
            count = len([f for f in class_dir.iterdir()
                        if f.is_file() and f.suffix in image_extensions])
            percentage = 0
            print(f"  {split:5s}: {count:,}", end="")
            class_total += count

    # Calculate percentages
    print(f"\n  Total: {class_total:,}")
    for split in splits:
        class_dir = target_dir / split / class_name
        if class_dir.exists():
            count = len([f for f in class_dir.iterdir()
                        if f.is_file() and f.suffix in image_extensions])
            percentage = (count / class_total * 100) if class_total > 0 else 0
            print(f"  {split} %: {percentage:.1f}%")

print("\n" + "=" * 80)
print("✅ Dataset verification complete!")
print(f"📂 Directory: {target_dir}")
print("=" * 80)
