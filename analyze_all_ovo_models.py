#!/usr/bin/env python3
"""
Comprehensive Analysis of All OVO Binary Classifiers
Checks validation accuracy and training quality for every binary model.
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
    """Analyze all OVO binary classifiers."""
    models_dir = Path('./ovo_ensemble_results/models')

    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return

    # Get all binary classifier models
    binary_models = []
    for model_file in models_dir.glob('best_*.pth'):
        if model_file.name != 'ovo_ensemble_best.pth':
            binary_models.append(model_file)

    print(f"🔍 Analyzing {len(binary_models)} binary classifiers...")
    print("=" * 80)

    # Analyze each model
    results = {}
    base_models = ['mobilenet_v2', 'inception_v3', 'densenet121']
    class_pairs = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]

    # Track file sizes by base model
    size_summary = {}

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

    for base_model, size_info in size_summary.items():
        all_same_indicator = "⚠️ ALL IDENTICAL" if size_info['all_same'] else "✅ VARIED"
        print(f"{base_model:<15} | {size_info['avg_mb']:>6.1f}MB avg | {all_same_indicator}")

    # Check for accuracy information
    print("\n" + "=" * 80)
    print("🎯 ACCURACY ANALYSIS")
    print("=" * 80)

    total_with_accuracy = 0
    total_models = 0

    for base_model in base_models:
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
    for base_model, size_info in size_summary.items():
        if size_info['all_same']:
            issues.append(f"❌ {base_model}: All models identical size ({size_info['avg_mb']:.1f}MB) - likely untrained")

    # Check missing accuracy data
    if total_with_accuracy < total_models:
        issues.append(f"⚠️ Missing accuracy data in {total_models - total_with_accuracy}/{total_models} models")

    if not issues:
        print("✅ No obvious issues detected")
    else:
        for issue in issues:
            print(issue)

    # Save detailed results
    output_file = 'ovo_models_analysis.json'
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'size_summary': size_summary,
            'total_models': total_models,
            'models_with_accuracy': total_with_accuracy,
            'issues': issues
        }, f, indent=2)

    print(f"\n💾 Detailed analysis saved: {output_file}")

if __name__ == "__main__":
    main()