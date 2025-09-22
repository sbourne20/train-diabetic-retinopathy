#!/usr/bin/env python3
"""
Comprehensive Analysis of All OVO Binary Classifiers and Ensemble Models
Checks validation accuracy and training quality for every model type.
"""

import os
import sys
import json
import torch
from pathlib import Path

def analyze_model_basic(model_path):
    """Basic analysis of a model checkpoint."""
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')

        # Get file size
        file_size = os.path.getsize(model_path)

        # Try to get accuracy information
        accuracy_info = {}
        if isinstance(checkpoint, dict):
            # Look for accuracy keys
            for key in checkpoint.keys():
                if 'accuracy' in key.lower() or 'acc' in key.lower():
                    accuracy_info[key] = checkpoint[key]

        return {
            'model_path': str(model_path),
            'file_size_mb': file_size / (1024 * 1024),
            'checkpoint_type': type(checkpoint).__name__,
            'checkpoint_keys': list(checkpoint.keys()) if isinstance(checkpoint, dict) else [],
            'accuracy_info': accuracy_info,
            'status': 'loaded'
        }
    except Exception as e:
        return {
            'model_path': str(model_path),
            'status': 'error',
            'error': str(e)
        }

def main():
    """Analyze all OVO binary classifiers and ensemble models."""
    # Try different possible model directories
    possible_dirs = [
        './super_ensemble_results/models',
        './ovo_ensemble_results_balanced/models',
        './ovo_ensemble_results_v4/models',
        './ovo_ensemble_results_v3/models',
        './ovo_ensemble_results_v2/models',
        './ovo_ensemble_results/models'
    ]

    models_dir = None
    analysis_type = "unknown"

    for dir_path in possible_dirs:
        if Path(dir_path).exists():
            models_dir = Path(dir_path)
            if 'super_ensemble' in dir_path:
                analysis_type = "ensemble"
            else:
                analysis_type = "ovo"
            break

    if not models_dir:
        print(f"❌ No models directory found. Checked:")
        for dir_path in possible_dirs:
            print(f"   - {dir_path}")
        return

    print(f"🔍 Analyzing models in: {models_dir}")
    print(f"📋 Analysis type: {analysis_type.upper()}")

    # Get all model files
    model_files = []
    for model_file in models_dir.glob('best_*.pth'):
        if model_file.name not in ['ovo_ensemble_best.pth', 'super_ensemble_best.pth']:
            model_files.append(model_file)

    print(f"📋 Found {len(model_files)} models")

    # Show which models exist
    if model_files:
        print("📄 Available models:")
        for model in sorted(model_files):
            size_mb = model.stat().st_size / (1024 * 1024)
            print(f"   - {model.name} ({size_mb:.1f} MB)")
    else:
        print("⚠️  No models found in directory")

    # Analyze based on type
    results = {}
    size_summary = {}

    if analysis_type == "ensemble":
        # Analyze ensemble models (MedSigLIP, EfficientNet)
        ensemble_models = ['medsiglip_448', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5']
        sizes = []

        print("\n" + "=" * 80)
        print("🏆 ENSEMBLE MODELS ANALYSIS")
        print("=" * 80)

        for model_name in ensemble_models:
            model_path = models_dir / f"best_{model_name}.pth"

            if model_path.exists():
                analysis = analyze_model_basic(model_path)
                results[model_name] = analysis
                sizes.append(analysis.get('file_size_mb', 0))

                # Print individual result
                status = "✅" if analysis['status'] == 'loaded' else "❌"
                size_mb = analysis.get('file_size_mb', 0)
                acc_info = analysis.get('accuracy_info', {})
                acc_str = f"({len(acc_info)} accuracy fields)" if acc_info else "(no accuracy saved)"

                print(f"{status} {model_name:<25} | {size_mb:>8.1f}MB | {acc_str}")
            else:
                print(f"❌ {model_name:<25} | MISSING")
                results[model_name] = {'status': 'missing'}

        # Size summary for ensemble
        if sizes:
            size_summary['ensemble'] = {
                'min_mb': min(sizes),
                'max_mb': max(sizes),
                'avg_mb': sum(sizes) / len(sizes),
                'all_same': len(set(f"{s:.1f}" for s in sizes)) == 1
            }

    else:
        # Original OVO analysis
        base_models = ['mobilenet_v2', 'inception_v3', 'densenet121']
        class_pairs = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]

        print("\n" + "=" * 80)
        print("🔗 OVO BINARY CLASSIFIERS ANALYSIS")
        print("=" * 80)

        for base_model in base_models:
            results[base_model] = {}
            sizes = []

            for pair in class_pairs:
                model_name = f"best_{base_model}_{pair[0]}_{pair[1]}.pth"
                model_path = models_dir / model_name

                if model_path.exists():
                    analysis = analyze_model_basic(model_path)
                    results[base_model][f"{pair[0]}_{pair[1]}"] = analysis
                    sizes.append(analysis.get('file_size_mb', 0))

                    # Print individual result
                    status = "✅" if analysis['status'] == 'loaded' else "❌"
                    size_mb = analysis.get('file_size_mb', 0)
                    acc_info = analysis.get('accuracy_info', {})
                    acc_str = f"({len(acc_info)} accuracy fields)" if acc_info else "(no accuracy saved)"

                    print(f"{status} {model_name:<35} | {size_mb:>6.1f}MB | {acc_str}")
                else:
                    print(f"❌ {model_name:<35} | MISSING")
                    results[base_model][f"{pair[0]}_{pair[1]}"] = {'status': 'missing'}

            # Size summary for this base model
            if sizes:
                size_summary[base_model] = {
                    'min_mb': min(sizes),
                    'max_mb': max(sizes),
                    'avg_mb': sum(sizes) / len(sizes),
                    'all_same': len(set(f"{s:.1f}" for s in sizes)) == 1
                }

    print("\n" + "=" * 80)
    print("📊 SIZE SUMMARY")
    print("=" * 80)

    for model_type, size_info in size_summary.items():
        all_same_indicator = "⚠️ ALL IDENTICAL" if size_info['all_same'] else "✅ VARIED"
        print(f"{model_type:<15} | {size_info['avg_mb']:>6.1f}MB avg | {all_same_indicator}")

    # Check for accuracy information
    print("\n" + "=" * 80)
    print("🎯 ACCURACY ANALYSIS")
    print("=" * 80)

    total_with_accuracy = 0
    total_models = 0

    if analysis_type == "ensemble":
        # Count ensemble models with accuracy
        for model_name, analysis in results.items():
            if analysis.get('status') == 'loaded':
                total_models += 1
                if analysis.get('accuracy_info'):
                    total_with_accuracy += 1

        print(f"ENSEMBLE MODELS | {total_with_accuracy}/{total_models} models have accuracy data")

    else:
        # Original OVO counting
        for base_model in ['mobilenet_v2', 'inception_v3', 'densenet121']:
            if base_model in results:
                models_with_acc = 0
                for pair_key, analysis in results[base_model].items():
                    if analysis.get('status') == 'loaded':
                        total_models += 1
                        if analysis.get('accuracy_info'):
                            models_with_acc += 1
                            total_with_accuracy += 1

                print(f"{base_model:<15} | {models_with_acc}/10 models have accuracy data")

    print(f"\n🎯 OVERALL: {total_with_accuracy}/{total_models} models have training metrics")

    # Identify potential issues
    print("\n" + "=" * 80)
    print("🔍 POTENTIAL ISSUES")
    print("=" * 80)

    issues = []

    # Check if all models are identical size (indicating no training)
    for model_type, size_info in size_summary.items():
        if size_info['all_same']:
            issues.append(f"❌ {model_type}: All models identical size ({size_info['avg_mb']:.1f}MB) - likely untrained")

    # Check missing accuracy data
    if total_with_accuracy < total_models:
        issues.append(f"⚠️ Missing accuracy data in {total_models - total_with_accuracy}/{total_models} models")

    if not issues:
        print("✅ No obvious issues detected")
    else:
        for issue in issues:
            print(issue)

    # Save detailed results
    output_file = f'{analysis_type}_models_analysis.json'
    with open(output_file, 'w') as f:
        json.dump({
            'analysis_type': analysis_type,
            'models_directory': str(models_dir),
            'results': results,
            'size_summary': size_summary,
            'total_models': total_models,
            'models_with_accuracy': total_with_accuracy,
            'issues': issues
        }, f, indent=2)

    print(f"\n💾 Detailed analysis saved: {output_file}")

    # Final assessment
    if analysis_type == "ensemble":
        print(f"\n🏆 ENSEMBLE MODEL SUMMARY:")
        print(f"   Found {total_models} ensemble models")
        if total_models > 0:
            completion_rate = (total_models / 4) * 100  # 4 expected ensemble models
            print(f"   Training completion: {completion_rate:.1f}%")
            if completion_rate < 100:
                print(f"   🔄 Super ensemble training in progress")

if __name__ == "__main__":
    main()