#!/usr/bin/env python3
"""
Check OVO training resume status - show completed and remaining binary classifiers.
"""

import os
import sys
from pathlib import Path
from itertools import combinations

def check_training_status(results_dir='./seresnext_5class_results'):
    """Check which binary classifiers are completed and which remain."""

    # OVO configuration
    num_classes = 5
    base_models = ['seresnext50_32x4d']
    class_pairs = list(combinations(range(num_classes), 2))

    total_classifiers = len(base_models) * len(class_pairs)

    print("=" * 70)
    print("🔍 OVO TRAINING RESUME STATUS CHECK")
    print("=" * 70)
    print(f"📊 Configuration:")
    print(f"   Classes: {num_classes}")
    print(f"   Base models: {base_models}")
    print(f"   Binary pairs needed: {len(class_pairs)}")
    print(f"   Total classifiers: {total_classifiers}")
    print()

    # Check for checkpoints
    models_dir = Path(results_dir) / 'models'

    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        print(f"   Starting from scratch - no checkpoints available")
        return

    completed = []
    remaining = []

    for model_name in base_models:
        for class_a, class_b in class_pairs:
            checkpoint_name = f'best_{model_name}_{class_a}_{class_b}.pth'
            checkpoint_path = models_dir / checkpoint_name

            classifier_key = f"{model_name}_{class_a}_{class_b}"

            if checkpoint_path.exists():
                # Get file size
                size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
                completed.append({
                    'key': classifier_key,
                    'pair': f"{class_a}_{class_b}",
                    'path': checkpoint_path,
                    'size_mb': size_mb
                })
            else:
                remaining.append({
                    'key': classifier_key,
                    'pair': f"{class_a}_{class_b}"
                })

    # Display results
    print(f"✅ COMPLETED CLASSIFIERS: {len(completed)}/{total_classifiers}")
    print("-" * 70)

    if completed:
        for item in completed:
            print(f"   ✓ {item['key']:40} ({item['size_mb']:.1f} MB)")
    else:
        print("   None - starting fresh training")

    print()
    print(f"⏳ REMAINING CLASSIFIERS: {len(remaining)}/{total_classifiers}")
    print("-" * 70)

    if remaining:
        for item in remaining:
            print(f"   ⏹ {item['key']:40} (not started)")
    else:
        print("   None - all classifiers completed! ✨")

    print()
    print("=" * 70)
    print(f"📈 Progress: {len(completed)}/{total_classifiers} ({len(completed)/total_classifiers*100:.1f}%)")
    print("=" * 70)

    if remaining:
        print()
        print("🔄 NEXT STEPS:")
        print(f"   Run training script with --resume flag:")
        print(f"   ./train_5class_seresnext.sh")
        print()
        print(f"   The script will automatically:")
        print(f"   ✅ Skip the {len(completed)} completed classifiers")
        print(f"   🚀 Train the remaining {len(remaining)} classifiers")
        print(f"   💾 Include all memory optimizations (no more OOM!)")
    else:
        print()
        print("✨ ALL TRAINING COMPLETE!")
        print("   You can now evaluate the OVO ensemble:")
        print("   python3 ensemble_5class_trainer.py --mode evaluate \\")
        print(f"       --output_dir {results_dir}")

    print()

    return {
        'total': total_classifiers,
        'completed': len(completed),
        'remaining': len(remaining),
        'completed_list': completed,
        'remaining_list': remaining
    }

if __name__ == '__main__':
    # Allow custom results directory as argument
    results_dir = sys.argv[1] if len(sys.argv) > 1 else './seresnext_5class_results'

    # Also check v2-model-dr location if exists
    v2_dir = './v2-model-dr/seresnext_5class_results'

    if Path(v2_dir).exists() and not Path(results_dir).exists():
        print(f"ℹ️  Using existing results from: {v2_dir}")
        results_dir = v2_dir

    status = check_training_status(results_dir)
