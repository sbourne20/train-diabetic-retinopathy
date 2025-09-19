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
    """Generate comprehensive training report with overfitting detection."""
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
        overfitting_count = 0

        for pair in class_pairs:
            pair_key = f"{pair[0]}_{pair[1]}"
            if pair_key in results[base_model]:
                analysis = results[base_model][pair_key]
                total_models += 1

                if analysis['status'] == 'loaded' and analysis.get('format') == 'comprehensive':
                    trained_models += 1

                    # Get accuracies
                    val_acc = analysis.get('best_val_accuracy', 0.0)
                    train_acc = analysis.get('current_train_accuracy', 0.0)
                    epoch = analysis.get('epoch', 'unknown')

                    model_accuracies.append(val_acc)

                    # Calculate overfitting gap
                    overfitting_gap = train_acc - val_acc

                    # Determine status icon based on performance and overfitting (MEDICAL GRADE STANDARDS)
                    if val_acc > 85.0:
                        if overfitting_gap <= 5.0:
                            status_icon = "✅"  # Excellent - Medical grade
                        elif overfitting_gap <= 8.0:
                            status_icon = "⚠️"  # Good but approaching critical
                            overfitting_count += 1
                        else:
                            status_icon = "🔴"  # Critical overfitting
                            overfitting_count += 1
                    elif val_acc > 75.0:
                        if overfitting_gap <= 6.0:
                            status_icon = "⚠️"  # Moderate
                        else:
                            status_icon = "🔴"  # Poor + overfitting
                            overfitting_count += 1
                    else:
                        status_icon = "❌"  # Poor performance
                        if overfitting_gap > 6.0:
                            overfitting_count += 1

                    # Add overfitting indicator to output (MEDICAL GRADE THRESHOLDS)
                    overfitting_indicator = ""
                    if overfitting_gap >= 8.0:
                        overfitting_indicator = " 🚨 CRITICAL OVERFITTING (≥8%)"
                    elif overfitting_gap > 6.0:
                        overfitting_indicator = " ⚠️ OVERFITTING"
                    elif overfitting_gap > 4.0:
                        overfitting_indicator = " 📈 MILD OVERFITTING"

                    print(f"  {status_icon} Classes {pair[0]}-{pair[1]}: {val_acc:5.1f}% (Train: {train_acc:5.1f}%, Epoch: {epoch}){overfitting_indicator}")

                else:
                    print(f"  ❌ Classes {pair[0]}-{pair[1]}: No training data")

        # Model summary with overfitting analysis
        if model_accuracies:
            avg_acc = sum(model_accuracies) / len(model_accuracies)
            min_acc = min(model_accuracies)
            max_acc = max(model_accuracies)

            print(f"\n  📈 SUMMARY: {trained_models}/{total_models} models trained")
            print(f"     Average Accuracy: {avg_acc:5.1f}%")
            print(f"     Range: {min_acc:5.1f}% - {max_acc:5.1f}%")

            # Overfitting analysis
            if overfitting_count > 0:
                print(f"     🚨 Overfitting detected in {overfitting_count}/{trained_models} models")
            else:
                print(f"     ✅ No significant overfitting detected")

            # Quality assessment with overfitting consideration
            if avg_acc > 90 and overfitting_count <= trained_models * 0.2:
                print(f"     🏆 EXCELLENT - Medical grade quality")
            elif avg_acc > 85 and overfitting_count <= trained_models * 0.3:
                print(f"     ✅ GOOD - Production ready")
            elif avg_acc > 80 or overfitting_count > trained_models * 0.5:
                print(f"     ⚠️ MODERATE - Needs improvement (accuracy or overfitting)")
            else:
                print(f"     ❌ POOR - Requires retraining")
        else:
            print(f"  ❌ NO TRAINED MODELS FOUND")

    return results

def main():
    """Main analysis function."""
    # Check for v4 (fixed), then v3, v2, v1
    models_dir = Path('./ovo_ensemble_results_v4/models')
    version = "v4"

    if not models_dir.exists():
        models_dir = Path('./ovo_ensemble_results_v3/models')
        version = "v3"

    if not models_dir.exists():
        models_dir = Path('./ovo_ensemble_results_v2/models')
        version = "v2"

    if not models_dir.exists():
        models_dir = Path('./ovo_ensemble_results/models')
        version = "v1"

    if not models_dir.exists():
        print(f"❌ No models directory found. Checked:")
        print(f"   - ./ovo_ensemble_results_v4/models (ENHANCED)")
        print(f"   - ./ovo_ensemble_results_v3/models")
        print(f"   - ./ovo_ensemble_results_v2/models")
        print(f"   - ./ovo_ensemble_results/models")
        return

    print(f"🔍 Analyzing models in: {models_dir}")
    if version == "v4":
        print(f"🛡️ Using V4 ENHANCED results with overfitting prevention")
    else:
        print(f"📦 Using {version} dataset results")

    # Get all binary classifier models
    binary_models = []
    for model_file in models_dir.glob('best_*.pth'):
        if model_file.name != 'ovo_ensemble_best.pth':
            binary_models.append(model_file)

    print(f"📋 Found {len(binary_models)} binary classifiers")

    # Show which models exist
    if binary_models:
        print("📄 Available models:")
        for model in sorted(binary_models):
            size_mb = model.stat().st_size / (1024 * 1024)
            print(f"   - {model.name} ({size_mb:.1f} MB)")
    else:
        print("⚠️  No binary classifier models found in directory")

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

    # Save detailed results with version info
    output_file = f'ovo_comprehensive_analysis_{version}.json'
    analysis_results = {
        'version': version,
        'models_directory': str(models_dir),
        'total_models_found': len(binary_models),
        'analysis_timestamp': str(Path().resolve()),
        'results': results
    }

    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)

    print(f"\n💾 Detailed analysis saved: {output_file}")

    # Summary of findings
    total_trained = sum(1 for base_model in results.values()
                       for pair_result in base_model.values()
                       if pair_result.get('status') == 'loaded' and
                          pair_result.get('format') == 'comprehensive')
    total_expected = len(base_models) * len(class_pairs)

    print(f"\n📊 FINAL SUMMARY:")
    print(f"   Models trained: {total_trained}/{total_expected}")
    print(f"   Training progress: {total_trained/total_expected*100:.1f}%")

    if total_trained < total_expected:
        print(f"   🔄 Training appears to be in progress for {version.upper()}")
        print(f"   💡 Run this script again after more models are trained")

    if version == "v3":
        print(f"\n⚠️ V3 ISSUES DETECTED:")
        print(f"   🚨 Severe overfitting in multiple models")
        print(f"   ❌ No overfitting prevention was applied")
        print(f"   💡 Use train_improved_ovo_fixed.sh for V4 with proper overfitting prevention")

if __name__ == "__main__":
    main()