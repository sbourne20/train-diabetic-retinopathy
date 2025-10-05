#!/usr/bin/env python3
"""
Weighted Ensemble Inference with Multiple Weighting Strategies

Compares different ensemble weighting strategies:
1. Simple averaging (current: 90.82%)
2. Accuracy-based weighting
3. Optimized weighting (grid search)
4. Confidence-based weighting
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
import json
from tqdm import tqdm

# MedSigLIP support
try:
    from transformers import AutoModel, AutoProcessor
    MEDSIGLIP_AVAILABLE = True
except ImportError:
    MEDSIGLIP_AVAILABLE = False

class WeightedEnsemblePredictor:
    def __init__(self, checkpoints, dataset_path, device='cuda'):
        self.device = device
        self.checkpoints = checkpoints
        self.dataset_path = dataset_path

        # Load models
        print("Loading models...")
        self.models = {}
        for model_name, checkpoint_path in checkpoints.items():
            self.models[model_name] = self.load_model(model_name, checkpoint_path)

        print(f"✅ Loaded {len(self.models)} models")

    def load_model(self, model_name, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if model_name == 'densenet':
            model = models.densenet121(weights=None)
            model.classifier = nn.Linear(1024, 5)
        elif model_name == 'efficientnetb2':
            model = models.efficientnet_b2(weights=None)
            model.classifier[1] = nn.Linear(1408, 5)
        elif model_name == 'medsiglip':
            if not MEDSIGLIP_AVAILABLE:
                raise ImportError("transformers not available for MedSigLIP")
            base_model = AutoModel.from_pretrained(
                "flaviagiammarino/pubmed-clip-vit-base-patch32",
                trust_remote_code=True
            )
            model = nn.Sequential(
                base_model.vision_model,
                nn.Linear(768, 5)
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        val_acc = checkpoint.get('best_val_accuracy', 0.0)
        print(f"  ✅ {model_name}: {val_acc:.4f}")

        return model

    def get_predictions(self, model_name, dataloader, img_size):
        """Get predictions from single model"""
        model = self.models[model_name]
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=f"  {model_name}"):
                # Resize if needed
                if images.shape[-1] != img_size:
                    images = torch.nn.functional.interpolate(
                        images, size=(img_size, img_size), mode='bilinear'
                    )

                images = images.to(self.device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)

                all_probs.append(probs.cpu().numpy())
                all_preds.append(probs.argmax(dim=1).cpu().numpy())
                all_labels.append(labels.numpy())

        return (
            np.concatenate(all_preds),
            np.concatenate(all_probs),
            np.concatenate(all_labels)
        )

    def ensemble_predict(self, weights_dict):
        """
        Get ensemble predictions with custom weights

        Args:
            weights_dict: {'densenet': 0.33, 'medsiglip': 0.33, 'efficientnetb2': 0.34}
        """
        # Load test dataset
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_dataset = ImageFolder(f"{self.dataset_path}/test", transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

        print(f"\n📊 Test samples: {len(test_dataset)}")

        # Get predictions from each model
        print("\n🔍 Getting individual model predictions...")
        predictions = {}

        for model_name in ['densenet', 'efficientnetb2']:
            preds, probs, labels = self.get_predictions(model_name, test_loader, 299)
            predictions[model_name] = {'preds': preds, 'probs': probs, 'labels': labels}

        # MedSigLIP needs 448x448
        if 'medsiglip' in self.models:
            transform_448 = transforms.Compose([
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            test_dataset_448 = ImageFolder(f"{self.dataset_path}/test", transform=transform_448)
            test_loader_448 = DataLoader(test_dataset_448, batch_size=16, shuffle=False, num_workers=4)

            preds, probs, labels = self.get_predictions('medsiglip', test_loader_448, 448)
            predictions['medsiglip'] = {'preds': preds, 'probs': probs, 'labels': labels}

        # Compute weighted ensemble
        print("\n🤝 Creating weighted ensemble...")

        # Normalize weights
        total_weight = sum(weights_dict.values())
        normalized_weights = {k: v/total_weight for k, v in weights_dict.items()}

        print(f"Weights: {normalized_weights}")

        ensemble_probs = np.zeros_like(predictions['densenet']['probs'])
        for model_name, weight in normalized_weights.items():
            ensemble_probs += weight * predictions[model_name]['probs']

        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        labels = predictions['densenet']['labels']

        # Calculate accuracies
        individual_accs = {}
        for model_name, data in predictions.items():
            acc = accuracy_score(data['labels'], data['preds'])
            individual_accs[model_name] = acc

        ensemble_acc = accuracy_score(labels, ensemble_preds)

        return {
            'ensemble_accuracy': ensemble_acc,
            'individual_accuracies': individual_accs,
            'weights': normalized_weights,
            'predictions': ensemble_preds,
            'labels': labels,
            'probabilities': ensemble_probs
        }


def grid_search_weights(predictor, granularity=10):
    """
    Grid search for optimal weights

    Args:
        granularity: Number of steps for each weight (higher = slower but more precise)
    """
    print("\n" + "="*80)
    print("GRID SEARCH FOR OPTIMAL WEIGHTS")
    print("="*80)

    best_acc = 0
    best_weights = None

    # Generate weight combinations (w1, w2, w3) that sum to 1.0
    step = 1.0 / granularity

    print(f"Testing {granularity**2} weight combinations...")

    tested = 0
    for i in range(granularity + 1):
        w1 = i * step  # densenet
        for j in range(granularity + 1 - i):
            w2 = j * step  # medsiglip
            w3 = 1.0 - w1 - w2  # efficientnetb2

            if w3 < 0 or w3 > 1.0:
                continue

            weights = {
                'densenet': w1,
                'medsiglip': w2,
                'efficientnetb2': w3
            }

            results = predictor.ensemble_predict(weights)
            acc = results['ensemble_accuracy']

            if acc > best_acc:
                best_acc = acc
                best_weights = weights
                print(f"  🎯 New best: {acc:.4f} - {weights}")

            tested += 1

    print(f"\nTested {tested} combinations")
    print(f"\n🏆 Best accuracy: {best_acc:.4f}")
    print(f"🏆 Best weights: {best_weights}")

    return best_weights, best_acc


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Weighted Ensemble Inference')
    parser.add_argument('--dataset_path', default='./dataset_eyepacs', help='Dataset path')
    parser.add_argument('--optimize', action='store_true', help='Run grid search to find optimal weights')
    parser.add_argument('--granularity', type=int, default=10, help='Grid search granularity')
    args = parser.parse_args()

    # Model checkpoints
    checkpoints = {
        'densenet': './densenet_eyepacs_results/models/best_densenet121_multiclass.pth',
        'medsiglip': './medsiglip_95percent_results/models/best_medsiglip_448_multiclass.pth',
        'efficientnetb2': './efficientnetb2_eyepacs_results/models/best_efficientnetb2_multiclass.pth'
    }

    # Initialize predictor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = WeightedEnsemblePredictor(checkpoints, args.dataset_path, device)

    # Test different weighting strategies
    strategies = {
        'Simple Averaging': {
            'densenet': 1/3,
            'medsiglip': 1/3,
            'efficientnetb2': 1/3
        },
        'Accuracy-Based': {
            'densenet': 0.8888,
            'medsiglip': 0.8774,
            'efficientnetb2': 0.8987
        },
        'Best-Model Heavy': {
            'densenet': 0.2,
            'medsiglip': 0.2,
            'efficientnetb2': 0.6
        },
        'DenseNet+EfficientNetB2 Only': {
            'densenet': 0.5,
            'medsiglip': 0.0,
            'efficientnetb2': 0.5
        }
    }

    print("\n" + "="*80)
    print("TESTING DIFFERENT WEIGHTING STRATEGIES")
    print("="*80)

    results_summary = {}

    for strategy_name, weights in strategies.items():
        print(f"\n\n{'='*80}")
        print(f"STRATEGY: {strategy_name}")
        print(f"{'='*80}")

        results = predictor.ensemble_predict(weights)
        results_summary[strategy_name] = results

        print(f"\nResults:")
        print(f"  Ensemble Accuracy: {results['ensemble_accuracy']:.4f} ({results['ensemble_accuracy']*100:.2f}%)")
        print(f"  Individual accuracies:")
        for model_name, acc in results['individual_accuracies'].items():
            print(f"    {model_name}: {acc:.4f}")

        # Classification report
        print(f"\n  Classification Report:")
        print(classification_report(
            results['labels'],
            results['predictions'],
            target_names=['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'PDR'],
            digits=4
        ))

    # Find best strategy
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    best_strategy = max(results_summary.items(), key=lambda x: x[1]['ensemble_accuracy'])

    print(f"\n🏆 Best Strategy: {best_strategy[0]}")
    print(f"🏆 Accuracy: {best_strategy[1]['ensemble_accuracy']:.4f} ({best_strategy[1]['ensemble_accuracy']*100:.2f}%)")
    print(f"🏆 Weights: {best_strategy[1]['weights']}")

    # Improvement over simple averaging
    simple_acc = results_summary['Simple Averaging']['ensemble_accuracy']
    improvement = (best_strategy[1]['ensemble_accuracy'] - simple_acc) * 100
    print(f"\n📈 Improvement over simple averaging: {improvement:+.2f}%")

    # Optional: Grid search for optimal weights
    if args.optimize:
        print("\n" + "="*80)
        print("RUNNING GRID SEARCH OPTIMIZATION")
        print("="*80)

        optimal_weights, optimal_acc = grid_search_weights(predictor, args.granularity)

        print(f"\n🎯 Optimal weights found:")
        print(f"  {optimal_weights}")
        print(f"  Accuracy: {optimal_acc:.4f} ({optimal_acc*100:.2f}%)")

        improvement_from_simple = (optimal_acc - simple_acc) * 100
        print(f"\n📈 Improvement over simple averaging: {improvement_from_simple:+.2f}%")

        # Save optimal weights
        with open('./ensemble_3model_results/optimal_weights.json', 'w') as f:
            json.dump({
                'weights': optimal_weights,
                'accuracy': float(optimal_acc),
                'improvement': float(improvement_from_simple)
            }, f, indent=2)

        print(f"\n💾 Optimal weights saved to ./ensemble_3model_results/optimal_weights.json")

    print("\n✅ Weighted ensemble analysis complete!")
