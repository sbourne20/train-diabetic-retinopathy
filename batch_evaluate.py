#!/usr/bin/env python3
"""
Batch Evaluation Script for mata-dr.py
Tests ensemble on entire validation/test set and generates confusion matrix
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import MATADR class
from mata_dr import MATADR

def batch_evaluate(dataset_path, output_dir='./evaluation_results'):
    """
    Evaluate ensemble on entire test set

    Args:
        dataset_path: Path to dataset (e.g., ./dataset_eyepacs)
        output_dir: Where to save results
    """

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Initialize MATA-DR
    print("🏥 Initializing MATA-DR Ensemble...")
    mata = MATADR(device='cuda')

    # Class names
    class_names = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'PDR']

    # Collect all test images
    test_dir = Path(dataset_path) / 'test'

    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return

    print(f"\n📁 Scanning test directory: {test_dir}")

    all_images = []
    all_labels = []

    for class_idx in range(5):
        class_dir = test_dir / str(class_idx)
        if not class_dir.exists():
            print(f"⚠️  Warning: Class {class_idx} directory not found")
            continue

        # Get all image files
        image_files = list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))

        print(f"  Class {class_idx} ({class_names[class_idx]}): {len(image_files)} images")

        for img_path in image_files:
            all_images.append(str(img_path))
            all_labels.append(class_idx)

    print(f"\n📊 Total test images: {len(all_images)}")

    # Run predictions
    print("\n🔍 Running ensemble predictions...\n")

    predictions = []
    confidences = []
    failed_images = []

    for img_path in tqdm(all_images, desc="Processing images"):
        try:
            results = mata.predict_ensemble(img_path)
            predictions.append(results['ensemble']['class'])
            confidences.append(results['ensemble']['confidence'])
        except Exception as e:
            print(f"\n⚠️  Failed on {img_path}: {e}")
            failed_images.append(img_path)
            predictions.append(-1)  # Mark as failed
            confidences.append(0.0)

    # Remove failed predictions
    valid_indices = [i for i, pred in enumerate(predictions) if pred != -1]
    predictions = [predictions[i] for i in valid_indices]
    confidences = [confidences[i] for i in valid_indices]
    all_labels_valid = [all_labels[i] for i in valid_indices]
    all_images_valid = [all_images[i] for i in valid_indices]

    print(f"\n✅ Successfully processed: {len(predictions)}/{len(all_images)}")
    if failed_images:
        print(f"❌ Failed images: {len(failed_images)}")

    # Calculate metrics
    accuracy = accuracy_score(all_labels_valid, predictions)

    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\n🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📊 Total Samples: {len(predictions)}")
    print(f"💯 Average Confidence: {np.mean(confidences):.4f} ({np.mean(confidences)*100:.2f}%)")

    # Classification report
    print("\n📋 Detailed Classification Report:")
    print(classification_report(
        all_labels_valid,
        predictions,
        target_names=class_names,
        digits=4
    ))

    # Confusion matrix
    cm = confusion_matrix(all_labels_valid, predictions)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'Confusion Matrix - Ensemble Accuracy: {accuracy*100:.2f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    confusion_matrix_path = output_dir / 'confusion_matrix.png'
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Confusion matrix saved to: {confusion_matrix_path}")

    # Find misclassified images
    misclassified = []
    for i, (true_label, pred_label, conf, img_path) in enumerate(
        zip(all_labels_valid, predictions, confidences, all_images_valid)
    ):
        if true_label != pred_label:
            misclassified.append({
                'image': img_path,
                'true_class': int(true_label),
                'true_class_name': class_names[true_label],
                'predicted_class': int(pred_label),
                'predicted_class_name': class_names[pred_label],
                'confidence': float(conf)
            })

    print(f"\n❌ Misclassified images: {len(misclassified)}/{len(predictions)} ({len(misclassified)/len(predictions)*100:.2f}%)")

    # Save detailed results
    results_summary = {
        'overall_accuracy': float(accuracy),
        'total_samples': len(predictions),
        'average_confidence': float(np.mean(confidences)),
        'per_class_accuracy': {},
        'confusion_matrix': cm.tolist(),
        'misclassified_count': len(misclassified),
        'failed_images': failed_images
    }

    # Per-class accuracy
    for class_idx in range(5):
        class_mask = [label == class_idx for label in all_labels_valid]
        if sum(class_mask) > 0:
            class_predictions = [predictions[i] for i, mask in enumerate(class_mask) if mask]
            class_labels = [all_labels_valid[i] for i, mask in enumerate(class_mask) if mask]
            class_acc = accuracy_score(class_labels, class_predictions)
            results_summary['per_class_accuracy'][class_names[class_idx]] = float(class_acc)

    # Save JSON results
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_summary, indent=2, fp=f)
    print(f"💾 Results saved to: {results_path}")

    # Save misclassified images list
    misclassified_path = output_dir / 'misclassified_images.json'
    with open(misclassified_path, 'w') as f:
        json.dump(misclassified, indent=2, fp=f)
    print(f"💾 Misclassified images list saved to: {misclassified_path}")

    # Print top 10 most confident misclassifications
    if misclassified:
        print("\n🔍 Top 10 Most Confident Misclassifications:")
        sorted_misclassified = sorted(misclassified, key=lambda x: x['confidence'], reverse=True)[:10]
        for i, item in enumerate(sorted_misclassified, 1):
            print(f"\n  {i}. {Path(item['image']).name}")
            print(f"     True: {item['true_class_name']} → Predicted: {item['predicted_class_name']}")
            print(f"     Confidence: {item['confidence']*100:.2f}%")

    print("\n" + "="*80)
    print("✅ BATCH EVALUATION COMPLETE!")
    print("="*80)

    return results_summary


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Batch evaluation of MATA-DR ensemble')
    parser.add_argument('--dataset_path', default='./dataset_eyepacs',
                       help='Path to dataset with test folder')
    parser.add_argument('--output_dir', default='./evaluation_results',
                       help='Where to save results')

    args = parser.parse_args()

    batch_evaluate(args.dataset_path, args.output_dir)
