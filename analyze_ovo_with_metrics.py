#!/usr/bin/env python3
"""
Enhanced OVO Model Analysis with Training Metrics
Analyzes all binary classifiers and extracts comprehensive training data.
"""

import os
import sys
import json
import torch
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_checkpoint_with_metrics(model_path):
    """Comprehensive analysis of checkpoint with training metrics."""
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')

        # Basic file info
        file_size = os.path.getsize(model_path)

        result = {
            'model_path': str(model_path),
            'file_size_mb': file_size / (1024 * 1024),
            'status': 'loaded'
        }

        # Handle new checkpoint format with metrics
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New format with comprehensive metrics
            result.update({
                'format': 'comprehensive',
                'epoch': checkpoint.get('epoch', 'unknown'),
                'best_val_accuracy': checkpoint.get('best_val_accuracy', 0.0),
                'current_val_accuracy': checkpoint.get('current_val_accuracy', 0.0),
                'current_train_accuracy': checkpoint.get('current_train_accuracy', 0.0),
                'class_pair': checkpoint.get('class_pair', 'unknown'),
                'model_name': checkpoint.get('model_name', 'unknown'),
                'has_training_history': 'train_history' in checkpoint
            })

            # Extract training history if available
            if 'train_history' in checkpoint:
                history = checkpoint['train_history']
                result['training_summary'] = {
                    'total_epochs': len(history.get('train_accuracies', [])),
                    'final_train_acc': history['train_accuracies'][-1] if history.get('train_accuracies') else 0.0,
                    'final_val_acc': history['val_accuracies'][-1] if history.get('val_accuracies') else 0.0,
                    'max_val_acc': max(history['val_accuracies']) if history.get('val_accuracies') else 0.0,
                    'min_train_loss': min(history['train_losses']) if history.get('train_losses') else float('inf'),
                    'min_val_loss': min(history['val_losses']) if history.get('val_losses') else float('inf')
                }

        elif isinstance(checkpoint, dict):
            # Old format or partial metrics
            result['format'] = 'legacy_dict'

            # Look for any accuracy keys
            accuracy_keys = [k for k in checkpoint.keys() if 'accuracy' in k.lower() or 'acc' in k.lower()]
            if accuracy_keys:
                result['found_accuracies'] = {k: checkpoint[k] for k in accuracy_keys}
            else:
                result['found_accuracies'] = {}

        else:
            # Just state dict
            result['format'] = 'state_dict_only'

        return result

    except Exception as e:
        return {
            'model_path': str(model_path),
            'status': 'error',
            'error': str(e)
        }

def generate_training_report(results):
    """Generate comprehensive training report."""
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE TRAINING ANALYSIS REPORT")
    print("=" * 80)

    # Summary statistics
    base_models = ['mobilenet_v2', 'inception_v3', 'densenet121']
    class_pairs = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]

    for base_model in base_models:
        print(f"\n🏗️ {base_model.upper()}")
        print("-" * 60)

        model_accuracies = []
        total_models = 0
        trained_models = 0

        for pair in class_pairs:
            pair_key = f"{pair[0]}_{pair[1]}"
            if pair_key in results[base_model]:
                analysis = results[base_model][pair_key]
                total_models += 1

                if analysis['status'] == 'loaded' and analysis.get('format') == 'comprehensive':
                    trained_models += 1
                    best_acc = analysis.get('best_val_accuracy', 0.0)
                    model_accuracies.append(best_acc)

                    # Print individual model performance
                    train_acc = analysis.get('current_train_accuracy', 0.0)
                    val_acc = analysis.get('current_val_accuracy', 0.0)
                    epoch = analysis.get('epoch', 'unknown')

                    status_icon = "✅" if best_acc > 80.0 else "⚠️" if best_acc > 70.0 else "❌"
                    print(f"  {status_icon} Classes {pair[0]}-{pair[1]}: {best_acc:5.1f}% (Train: {train_acc:5.1f}%, Epoch: {epoch})")

                else:
                    print(f"  ❌ Classes {pair[0]}-{pair[1]}: No training data")

        # Model summary
        if model_accuracies:
            avg_acc = sum(model_accuracies) / len(model_accuracies)
            min_acc = min(model_accuracies)
            max_acc = max(model_accuracies)

            print(f"\n  📈 SUMMARY: {trained_models}/{total_models} models trained")
            print(f"     Average Accuracy: {avg_acc:5.1f}%")
            print(f"     Range: {min_acc:5.1f}% - {max_acc:5.1f}%")

            # Quality assessment
            if avg_acc > 85:
                print(f"     🏆 EXCELLENT - Medical grade quality")
            elif avg_acc > 80:
                print(f"     ✅ GOOD - Production ready")
            elif avg_acc > 75:
                print(f"     ⚠️ MODERATE - Needs improvement")
            else:
                print(f"     ❌ POOR - Requires retraining")
        else:
            print(f"  ❌ NO TRAINED MODELS FOUND")

    return results

def main():
    """Main analysis function."""
    models_dir = Path('./ovo_ensemble_results_v2/models')

    if not models_dir.exists():
        models_dir = Path('./ovo_ensemble_results/models')

    if not models_dir.exists():
        print(f"❌ No models directory found")
        return

    print(f"🔍 Analyzing models in: {models_dir}")

    # Get all binary classifier models
    binary_models = []
    for model_file in models_dir.glob('best_*.pth'):
        if model_file.name != 'ovo_ensemble_best.pth':
            binary_models.append(model_file)

    print(f"📋 Found {len(binary_models)} binary classifiers")

    # Analyze each model
    results = {}
    base_models = ['mobilenet_v2', 'inception_v3', 'densenet121']
    class_pairs = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]

    for base_model in base_models:
        results[base_model] = {}

        for pair in class_pairs:
            model_name = f"best_{base_model}_{pair[0]}_{pair[1]}.pth"
            model_path = models_dir / model_name

            if model_path.exists():
                analysis = analyze_checkpoint_with_metrics(model_path)
                results[base_model][f"{pair[0]}_{pair[1]}"] = analysis
            else:
                results[base_model][f"{pair[0]}_{pair[1]}"] = {'status': 'missing'}

    # Generate comprehensive report
    generate_training_report(results)

    # Save detailed results
    output_file = 'ovo_comprehensive_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Detailed analysis saved: {output_file}")

if __name__ == "__main__":
    main()