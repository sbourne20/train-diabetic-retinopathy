#!/usr/bin/env python3
"""
Meta-Ensemble for OVO Models
Combines multiple OVO ensemble models (EfficientNetB2, DenseNet121, ResNet50, etc.)
by ensembling their 5-class probability outputs.

Usage:
    python meta_ensemble_ovo.py --models efficientnetb2 densenet121 \
        --model_dirs ./efficientnetb2_5class_results ./densenet_5class_results \
        --dataset_path ./dataset_eyepacs_5class_balanced \
        --method weighted
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Setup logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetaEnsemble:
    """Meta-ensemble that combines multiple OVO ensemble models."""

    def __init__(self, ovo_ensembles, method='average', weights=None):
        """
        Args:
            ovo_ensembles: List of (model_name, ovo_ensemble) tuples
            method: 'average', 'weighted', or 'vote'
            weights: Optional weights for weighted averaging
        """
        self.ovo_ensembles = ovo_ensembles
        self.method = method
        self.weights = weights

        if weights is None and method == 'weighted':
            # Equal weights by default
            self.weights = [1.0 / len(ovo_ensembles)] * len(ovo_ensembles)

        logger.info(f"✅ Meta-Ensemble initialized with {len(ovo_ensembles)} models")
        logger.info(f"   Method: {method}")
        if weights:
            for i, (name, _) in enumerate(ovo_ensembles):
                logger.info(f"   {name}: weight = {weights[i]:.3f}")

    def predict(self, images, return_individual=False):
        """
        Predict using meta-ensemble.

        Args:
            images: Input images tensor
            return_individual: If True, return individual model predictions

        Returns:
            final_probs: Combined probabilities [batch_size, num_classes]
            individual_results: Optional dict of individual model results
        """
        device = images.device
        batch_size = images.size(0)
        num_classes = 5

        all_probs = []
        individual_results = {}

        # Get predictions from each OVO ensemble
        for model_name, ovo_ensemble in self.ovo_ensembles:
            ovo_ensemble.eval()
            with torch.no_grad():
                outputs = ovo_ensemble(images, return_individual=False)

                # Get probabilities (softmax of logits)
                if 'logits' in outputs:
                    probs = torch.softmax(outputs['logits'], dim=1)
                else:
                    probs = torch.softmax(outputs, dim=1)

                all_probs.append(probs)

                if return_individual:
                    individual_results[model_name] = {
                        'probs': probs.cpu().numpy(),
                        'pred': torch.argmax(probs, dim=1).cpu().numpy()
                    }

        # Combine predictions based on method
        if self.method == 'average':
            # Simple average
            final_probs = torch.stack(all_probs).mean(dim=0)

        elif self.method == 'weighted':
            # Weighted average
            weighted_probs = []
            for i, probs in enumerate(all_probs):
                weighted_probs.append(probs * self.weights[i])
            final_probs = torch.stack(weighted_probs).sum(dim=0)

        elif self.method == 'vote':
            # Majority voting on predictions
            all_preds = torch.stack([torch.argmax(p, dim=1) for p in all_probs])
            # Convert to one-hot and sum
            votes = torch.zeros(batch_size, num_classes, device=device)
            for preds in all_preds:
                votes.scatter_add_(1, preds.unsqueeze(1), torch.ones(batch_size, 1, device=device))
            # Normalize to probabilities
            final_probs = votes / len(all_probs)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        if return_individual:
            return final_probs, individual_results
        return final_probs


def load_ovo_ensemble(model_dir, model_name, device):
    """Load a single OVO ensemble model."""

    model_path = Path(model_dir) / "models" / "ovo_ensemble_best.pth"
    config_path = Path(model_dir) / "ovo_config.json"

    if not model_path.exists():
        raise FileNotFoundError(f"OVO ensemble not found: {model_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    logger.info(f"📦 Loading {model_name} OVO ensemble from {model_path}")

    # Import OVO ensemble class
    try:
        from ensemble_5class_trainer import OVOEnsemble
    except ImportError:
        logger.error("❌ Cannot import OVOEnsemble from ensemble_5class_trainer.py")
        sys.exit(1)

    # Create OVO ensemble
    ovo_ensemble = OVOEnsemble(
        base_models=[model_name],
        num_classes=config['data']['num_classes'],
        dropout=config['model'].get('dropout', 0.3),
        freeze_weights=config['model'].get('freeze_weights', False)
    )

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    ovo_ensemble.load_state_dict(checkpoint)
    ovo_ensemble.to(device)
    ovo_ensemble.eval()

    logger.info(f"✅ {model_name} OVO ensemble loaded successfully")

    return ovo_ensemble, config


def evaluate_meta_ensemble(meta_ensemble, test_loader, device, num_classes=5):
    """Evaluate meta-ensemble on test set."""

    logger.info("🔬 Evaluating Meta-Ensemble")

    all_preds = []
    all_targets = []
    all_individual_preds = {name: [] for name, _ in meta_ensemble.ovo_ensembles}

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            targets = targets.to(device)

            # Get meta-ensemble predictions
            final_probs, individual_results = meta_ensemble.predict(images, return_individual=True)

            # Final predictions
            final_pred = torch.argmax(final_probs, dim=1)
            all_preds.extend(final_pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Store individual predictions
            for model_name, results in individual_results.items():
                all_individual_preds[model_name].extend(results['pred'])

    # Calculate metrics
    meta_accuracy = accuracy_score(all_targets, all_preds)

    individual_accuracies = {}
    for model_name, preds in all_individual_preds.items():
        individual_accuracies[model_name] = accuracy_score(all_targets, preds)

    # Medical grade validation
    medical_grade_pass = meta_accuracy >= 0.90

    # Generate reports
    class_names = [f'Class_{i}' for i in range(num_classes)]
    report = classification_report(all_targets, all_preds, target_names=class_names)
    cm = confusion_matrix(all_targets, all_preds)

    results = {
        'meta_ensemble_accuracy': meta_accuracy,
        'individual_accuracies': individual_accuracies,
        'medical_grade_pass': medical_grade_pass,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Meta-Ensemble OVO Models')

    parser.add_argument('--models', nargs='+', required=True,
                       help='Model names (e.g., efficientnetb2 densenet121)')
    parser.add_argument('--model_dirs', nargs='+', required=True,
                       help='Model result directories')
    parser.add_argument('--dataset_path', required=True,
                       help='Path to dataset (with test folder)')
    parser.add_argument('--method', choices=['average', 'weighted', 'vote'],
                       default='average',
                       help='Ensemble method')
    parser.add_argument('--weights', nargs='+', type=float, default=None,
                       help='Weights for weighted ensemble (e.g., 0.45 0.55)')
    parser.add_argument('--img_size', type=int, default=260,
                       help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--output_file', default='./meta_ensemble_results.json',
                       help='Output file for results')

    args = parser.parse_args()

    # Validation
    if len(args.models) != len(args.model_dirs):
        logger.error("❌ Number of models must match number of model directories")
        sys.exit(1)

    if args.weights and len(args.weights) != len(args.models):
        logger.error("❌ Number of weights must match number of models")
        sys.exit(1)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  Device: {device}")

    # Load all OVO ensembles
    logger.info("📦 Loading OVO ensembles...")
    ovo_ensembles = []

    for model_name, model_dir in zip(args.models, args.model_dirs):
        ovo_ensemble, config = load_ovo_ensemble(model_dir, model_name, device)
        ovo_ensembles.append((model_name, ovo_ensemble))

    # Create meta-ensemble
    logger.info("🔗 Creating Meta-Ensemble...")
    meta_ensemble = MetaEnsemble(
        ovo_ensembles=ovo_ensembles,
        method=args.method,
        weights=args.weights
    )

    # Load test dataset
    logger.info("📊 Loading test dataset...")
    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = ImageFolder(
        root=Path(args.dataset_path) / "test",
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logger.info(f"   Test samples: {len(test_dataset)}")

    # Evaluate
    results = evaluate_meta_ensemble(meta_ensemble, test_loader, device)

    # Display results
    logger.info("\n" + "="*80)
    logger.info("🎆 META-ENSEMBLE RESULTS")
    logger.info("="*80)
    logger.info(f"Method: {args.method}")
    logger.info(f"\n📊 Individual Model Accuracies:")
    for model_name, acc in results['individual_accuracies'].items():
        logger.info(f"   {model_name}: {acc*100:.2f}%")

    logger.info(f"\n🎯 Meta-Ensemble Accuracy: {results['meta_ensemble_accuracy']*100:.2f}%")
    logger.info(f"🏥 Medical Grade (≥90%): {'✅ PASS' if results['medical_grade_pass'] else '❌ FAIL'}")

    logger.info(f"\n📋 Classification Report:")
    logger.info(results['classification_report'])

    logger.info(f"\n📊 Confusion Matrix:")
    cm = np.array(results['confusion_matrix'])
    for i, row in enumerate(cm):
        logger.info(f"   Class {i}: {row}")

    # Save results
    output_path = Path(args.output_file)
    with open(output_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            'meta_ensemble_accuracy': float(results['meta_ensemble_accuracy']),
            'individual_accuracies': {k: float(v) for k, v in results['individual_accuracies'].items()},
            'medical_grade_pass': results['medical_grade_pass'],
            'method': args.method,
            'models': args.models,
            'weights': args.weights,
            'classification_report': results['classification_report'],
            'confusion_matrix': results['confusion_matrix']
        }
        json.dump(json_results, f, indent=2)

    logger.info(f"\n💾 Results saved to: {output_path}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
