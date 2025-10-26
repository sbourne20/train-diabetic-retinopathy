#!/usr/bin/env python3
"""
Rebuild the OVO ensemble from trained binary classifiers with FIXED voting logic
"""

import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from ensemble_5class_trainer import OVOEnsemble

def rebuild_ensemble():
    """Rebuild ensemble from individual binary classifier checkpoints."""

    print("="*80)
    print("🔧 REBUILDING OVO ENSEMBLE WITH FIXED VOTING LOGIC")
    print("="*80)

    # Create fresh ensemble
    ovo_ensemble = OVOEnsemble(
        base_models=['coatnet_0_rw_224'],
        num_classes=5,
        freeze_weights=True,
        dropout=0.28
    )

    models_dir = Path('./coatnet_5class_results/models')

    # Load each binary classifier
    model_name = 'coatnet_0_rw_224'
    loaded_count = 0

    for class_a, class_b in ovo_ensemble.class_pairs:
        pair_name = f"pair_{class_a}_{class_b}"
        checkpoint_path = models_dir / f"best_{model_name}_{class_a}_{class_b}.pth"

        if checkpoint_path.exists():
            print(f"📥 Loading {checkpoint_path.name}...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            # Load state dict
            ovo_ensemble.classifiers[model_name][pair_name].load_state_dict(
                checkpoint['model_state_dict']
            )

            # Get accuracy
            val_acc = checkpoint.get('best_val_accuracy', 0)
            print(f"   ✅ Loaded: Val Acc = {val_acc:.2f}%")
            loaded_count += 1
        else:
            print(f"   ❌ NOT FOUND: {checkpoint_path}")

    print(f"\n{'='*80}")
    print(f"✅ Successfully loaded {loaded_count}/10 binary classifiers")

    # Save the rebuilt ensemble
    output_path = models_dir / "ovo_ensemble_best_FIXED.pth"
    torch.save(ovo_ensemble.state_dict(), output_path)
    print(f"💾 Saved rebuilt ensemble to: {output_path}")

    # Also backup old and rename new
    old_ensemble = models_dir / "ovo_ensemble_best.pth"
    backup_path = models_dir / "ovo_ensemble_best_OLD_BROKEN.pth"

    if old_ensemble.exists():
        old_ensemble.rename(backup_path)
        print(f"📦 Backed up old ensemble to: {backup_path}")

    output_path.rename(old_ensemble)
    print(f"✅ Renamed fixed ensemble to: {old_ensemble}")

    print(f"\n{'='*80}")
    print(f"🎉 ENSEMBLE REBUILD COMPLETE!")
    print(f"{'='*80}")
    print(f"\nNow run: bash test_ovo_evaluation.sh")
    print(f"Expected accuracy: ~96-98% (matching binary classifier performance)")

if __name__ == '__main__':
    rebuild_ensemble()
