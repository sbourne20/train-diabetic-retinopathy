#!/usr/bin/env python3
"""
OVO Ensemble Checkpoint Analyzer
Analyzes .pth checkpoint files with PyTorch
"""

import torch
import sys
from pathlib import Path

def analyze_checkpoint(checkpoint_path):
    """Load and analyze checkpoint with PyTorch"""
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')

        return {
            'epoch': ckpt.get('epoch', 'N/A'),
            'best_val_accuracy': ckpt.get('best_val_accuracy', 0.0),
            'val_accuracy': ckpt.get('val_accuracy', 0.0),
            'train_accuracy': ckpt.get('train_accuracy', 'N/A'),
            'success': True
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def main():
    models_dir = Path('./densenet_5class_results/models')

    # List of OVO pair models
    pairs = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (1, 3), (1, 4),
        (2, 3), (2, 4),
        (3, 4)
    ]

    print('=' * 80)
    print('DENSENET121 OVO ENSEMBLE - ALL 10 BINARY CLASSIFIERS ANALYSIS')
    print('=' * 80)
    print()

    results = []

    for i, j in pairs:
        model_name = f'best_densenet121_{i}_{j}.pth'
        model_path = models_dir / model_name

        if not model_path.exists():
            print(f'❌ Missing: Pair ({i}, {j})')
            continue

        data = analyze_checkpoint(model_path)

        if not data['success']:
            print(f'❌ Error loading Pair ({i}, {j}): {data["error"]}')
            continue

        val_acc_raw = data['best_val_accuracy']
        epoch = data['epoch']
        train_acc_raw = data.get('train_accuracy', 'N/A')

        # Normalize to 0-1 range (checkpoint stores as percentage 0-100)
        val_acc = val_acc_raw / 100.0 if val_acc_raw > 1.0 else val_acc_raw

        # Calculate overfitting gap
        gap_str = 'N/A'
        if isinstance(train_acc_raw, float):
            train_acc = train_acc_raw / 100.0 if train_acc_raw > 1.0 else train_acc_raw
            gap = (train_acc - val_acc) * 100
            gap_str = f'{gap:+.2f}%'
        else:
            train_acc = 'N/A'

        results.append({
            'pair': f'{i}-{j}',
            'classes': (i, j),
            'val_acc': val_acc,  # Store as 0-1
            'train_acc': train_acc,
            'gap': gap_str,
            'epoch': epoch
        })

        # Status indicator based on 0-1 range
        if val_acc >= 0.90:
            status = '✅'
        elif val_acc >= 0.85:
            status = '⚠️'
        else:
            status = '❌'

        print(f'{status} Pair ({i},{j}): {val_acc*100:5.2f}% | Epoch: {epoch:3} | Gap: {gap_str:>8}')

    print()
    print('=' * 80)
    print('SUMMARY STATISTICS')
    print('=' * 80)
    print()

    if not results:
        print('❌ No valid checkpoints found')
        return

    val_accs = [r['val_acc'] for r in results]
    avg_acc = sum(val_accs) / len(val_accs)
    min_acc = min(val_accs)
    max_acc = max(val_accs)

    print(f'📊 Average Pair Accuracy: {avg_acc*100:.2f}%')
    print(f'🏆 Best Pair Accuracy:    {max_acc*100:.2f}%')
    print(f'⚠️  Worst Pair Accuracy:   {min_acc*100:.2f}%')
    print(f'📈 Accuracy Range:        {(max_acc-min_acc)*100:.2f}%')
    print()

    # Medical grade assessment
    if avg_acc >= 0.90:
        grade = '✅ EXCELLENT - Medical Grade (≥90%)'
        clinical_ready = True
    elif avg_acc >= 0.85:
        grade = '⚠️  GOOD - Near Medical Grade (85-90%)'
        clinical_ready = False
    elif avg_acc >= 0.80:
        grade = '📈 MODERATE - Research Quality (80-85%)'
        clinical_ready = False
    else:
        grade = '❌ NEEDS IMPROVEMENT (<80%)'
        clinical_ready = False

    print(f'🏥 Overall Assessment: {grade}')
    print(f'🎯 Clinical Ready: {"YES" if clinical_ready else "NO"}')
    print()

    # Identify weak pairs
    weak_pairs = [r for r in results if r['val_acc'] < 0.85]
    if weak_pairs:
        print('⚠️  Weak Pairs (<85% accuracy):')
        for r in weak_pairs:
            i, j = r['classes']
            print(f'   • Pair ({i},{j}): {r["val_acc"]*100:.2f}% - Needs improvement')
        print()

    # Identify strong pairs
    strong_pairs = [r for r in results if r['val_acc'] >= 0.90]
    if strong_pairs:
        print('✅ Strong Pairs (≥90% accuracy):')
        for r in strong_pairs:
            i, j = r['classes']
            print(f'   • Pair ({i},{j}): {r["val_acc"]*100:.2f}% - Excellent')
        print()

    # Class-specific analysis
    print('=' * 80)
    print('CLASS-SPECIFIC PERFORMANCE')
    print('=' * 80)
    print()

    class_names = {
        0: 'No DR',
        1: 'Mild NPDR',
        2: 'Moderate NPDR',
        3: 'Severe NPDR',
        4: 'PDR'
    }

    for class_id in range(5):
        class_pairs = [r for r in results if class_id in r['classes']]
        if class_pairs:
            class_avg = sum(r['val_acc'] for r in class_pairs) / len(class_pairs)
            print(f'Class {class_id} ({class_names[class_id]}):')
            print(f'  Average accuracy across {len(class_pairs)} pairs: {class_avg*100:.2f}%')

            # Show individual pairs
            for r in class_pairs:
                i, j = r['classes']
                print(f'    • vs Class {j if class_id == i else i}: {r["val_acc"]*100:.2f}%')
            print()

    print('=' * 80)

if __name__ == '__main__':
    main()
