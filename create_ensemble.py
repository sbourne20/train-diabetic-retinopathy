#!/usr/bin/env python3
"""
Create OVO Ensemble from Individual Binary Models
"""

import torch
from pathlib import Path
from ensemble_local_trainer import OVOEnsemble
import json

def create_ensemble_from_models():
    """Create ensemble file from individual binary classifiers."""

    results_dir = Path("./ovo_ensemble_results_v2")
    models_dir = results_dir / "models"

    # Load configuration
    config_path = results_dir / "ovo_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default config based on your training - ALL 3 architectures
        config = {
            'model': {'base_models': ['mobilenet_v2', 'inception_v3', 'densenet121']},
            'data': {'num_classes': 5}
        }

    print(f"🏗️ Creating OVO ensemble with {config['model']['base_models']}")

    # Create ensemble
    ensemble = OVOEnsemble(
        base_models=config['model']['base_models'],
        num_classes=config['data']['num_classes'],
        freeze_weights=True,
        dropout=0.3
    )

    # Load individual binary classifiers
    loaded_count = 0
    missing_models = []

    for model_name in config['model']['base_models']:
        for class_a, class_b in ensemble.class_pairs:
            model_path = models_dir / f"best_{model_name}_{class_a}_{class_b}.pth"
            classifier_name = f"pair_{class_a}_{class_b}"

            if model_path.exists():
                try:
                    checkpoint = torch.load(model_path, map_location='cpu')

                    # Handle both old and new checkpoint formats
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        ensemble.classifiers[model_name][classifier_name].load_state_dict(checkpoint['model_state_dict'])
                        best_acc = checkpoint.get('best_val_accuracy', 0.0)
                        print(f"✅ Loaded: {model_path.name} (Best Val Acc: {best_acc:.2f}%)")
                    else:
                        ensemble.classifiers[model_name][classifier_name].load_state_dict(checkpoint)
                        print(f"✅ Loaded: {model_path.name} (legacy format)")

                    loaded_count += 1
                except Exception as e:
                    print(f"❌ Failed to load {model_path.name}: {e}")
                    missing_models.append(str(model_path))
            else:
                print(f"❌ Missing: {model_path.name}")
                missing_models.append(str(model_path))

    print(f"\n📦 Loaded {loaded_count}/30 binary classifiers into ensemble")

    if missing_models:
        print(f"⚠️ Missing {len(missing_models)} models:")
        for model in missing_models:
            print(f"   {model}")

    # Save complete ensemble
    ensemble_path = models_dir / "ovo_ensemble_best.pth"
    torch.save(ensemble.state_dict(), ensemble_path)
    print(f"💾 OVO ensemble saved: {ensemble_path}")

    return ensemble_path, loaded_count

if __name__ == "__main__":
    ensemble_path, count = create_ensemble_from_models()
    print(f"\n✅ Ensemble creation complete: {count} models loaded")