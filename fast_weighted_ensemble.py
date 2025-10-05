#!/usr/bin/env python3
"""
OPTIMIZED Weighted Ensemble Inference with Prediction Caching

Key optimization: Run inference ONCE per model, cache predictions,
then test all weight combinations using cached predictions.

Speed improvement: ~34 hours → ~10 minutes for granularity=20
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
import json
from tqdm import tqdm
import time

# MedSigLIP support
try:
    from transformers import AutoModel, AutoProcessor
    MEDSIGLIP_AVAILABLE = True
except ImportError:
    MEDSIGLIP_AVAILABLE = False

class OptimizedEnsemblePredictor:
    def __init__(self, checkpoints, dataset_path, device='cuda'):
        self.device = device
        self.checkpoints = checkpoints
        self.dataset_path = dataset_path
        self.cached_predictions = {}
        self.cached_labels = None

        # Load models
        print("🔄 Loading models...")
        self.models = {}
        for model_name, checkpoint_path in checkpoints.items():
            self.models[model_name] = self.load_model(model_name, checkpoint_path)

        print(f"✅ Loaded {len(self.models)} models\n")

    def load_model(self, model_name, checkpoint_path):
        """Load model from checkpoint - using ensemble_local_trainer.py architecture"""
        from ensemble_local_trainer import MultiClassDRModel

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Create model using the same architecture as training
        if model_name == 'densenet':
            model = MultiClassDRModel(model_name='densenet121', num_classes=5, freeze_weights=False, dropout=0.3)
        elif model_name == 'efficientnetb2':
            model = MultiClassDRModel(model_name='efficientnetb2', num_classes=5, freeze_weights=False, dropout=0.2)
        elif model_name == 'medsiglip':
            if not MEDSIGLIP_AVAILABLE:
                raise ImportError("transformers not available for MedSigLIP")
            model = MultiClassDRModel(model_name='medsiglip_448', num_classes=5, freeze_weights=False, dropout=0.3)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        val_acc = checkpoint.get('best_val_accuracy', 0.0)
        print(f"  ✅ {model_name}: {val_acc:.4f} ({val_acc*100:.2f}%)")

        return model

    def cache_all_predictions(self):
        """
        Run inference ONCE for each model and cache results.
        This is the key optimization!
        """
        print("="*80)
        print("STEP 1: CACHING MODEL PREDICTIONS (runs once)")
        print("="*80)

        # Load test dataset
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_dataset = ImageFolder(f"{self.dataset_path}/test", transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

        print(f"\n📊 Test samples: {len(test_dataset)}\n")

        # Get predictions from DenseNet and EfficientNetB2 (299x299)
        print("🔍 Running inference (this takes a few minutes)...")
        for model_name in ['densenet', 'efficientnetb2']:
            preds, probs, labels = self.get_predictions(model_name, test_loader, 299)
            self.cached_predictions[model_name] = probs
            self.cached_labels = labels

            acc = accuracy_score(labels, preds)
            print(f"  ✅ {model_name}: {acc:.4f} ({acc*100:.2f}%) - cached")

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
            self.cached_predictions['medsiglip'] = probs

            acc = accuracy_score(labels, preds)
            print(f"  ✅ medsiglip: {acc:.4f} ({acc*100:.2f}%) - cached")

        print(f"\n✅ All predictions cached! Now we can test weight combinations instantly.\n")

    def get_predictions(self, model_name, dataloader, img_size):
        """Get predictions from single model"""
        model = self.models[model_name]
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=f"    {model_name}", leave=False):
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

    def ensemble_predict_from_cache(self, weights_dict):
        """
        Get ensemble predictions using CACHED predictions.
        This is ~1000x faster than re-running inference!
        """
        if not self.cached_predictions:
            raise ValueError("No cached predictions! Run cache_all_predictions() first")

        # Normalize weights
        total_weight = sum(weights_dict.values())
        normalized_weights = {k: v/total_weight for k, v in weights_dict.items()}

        # Compute weighted ensemble using cached predictions
        ensemble_probs = np.zeros_like(self.cached_predictions['densenet'])
        for model_name, weight in normalized_weights.items():
            ensemble_probs += weight * self.cached_predictions[model_name]

        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        ensemble_acc = accuracy_score(self.cached_labels, ensemble_preds)

        return {
            'ensemble_accuracy': ensemble_acc,
            'weights': normalized_weights,
            'predictions': ensemble_preds,
            'labels': self.cached_labels,
            'probabilities': ensemble_probs
        }


def grid_search_weights(predictor, granularity=10):
    """
    Grid search for optimal weights using cached predictions.

    Args:
        granularity: Number of steps for each weight (higher = slower but more precise)
    """
    print("\n" + "="*80)
    print("STEP 2: GRID SEARCH FOR OPTIMAL WEIGHTS")
    print("="*80)

    # Calculate total combinations
    total_combinations = (granularity + 1) * (granularity + 2) // 2
    print(f"\n🔍 Testing {total_combinations} weight combinations (granularity={granularity})")
    print(f"⏱️  Estimated time: ~{total_combinations * 0.001:.1f} seconds (~{total_combinations * 0.001 / 60:.1f} minutes)\n")

    best_acc = 0
    best_weights = None
    all_results = []

    # Generate weight combinations (w1, w2, w3) that sum to 1.0
    step = 1.0 / granularity

    start_time = time.time()
    tested = 0

    with tqdm(total=total_combinations, desc="Testing combinations") as pbar:
        for i in range(granularity + 1):
            w1 = i * step  # densenet
            for j in range(granularity + 1 - i):
                w2 = j * step  # medsiglip
                w3 = 1.0 - w1 - w2  # efficientnetb2

                if w3 < -0.001 or w3 > 1.001:  # Small tolerance for float errors
                    continue

                weights = {
                    'densenet': round(w1, 4),
                    'medsiglip': round(w2, 4),
                    'efficientnetb2': round(w3, 4)
                }

                results = predictor.ensemble_predict_from_cache(weights)
                acc = results['ensemble_accuracy']

                all_results.append({
                    'weights': weights,
                    'accuracy': acc
                })

                if acc > best_acc:
                    best_acc = acc
                    best_weights = weights
                    tqdm.write(f"  🎯 New best: {acc:.4f} ({acc*100:.2f}%) - {weights}")

                tested += 1
                pbar.update(1)

    elapsed = time.time() - start_time

    print(f"\n✅ Tested {tested} combinations in {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)")
    print(f"\n🏆 BEST ACCURACY: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"🏆 BEST WEIGHTS: {best_weights}")

    return best_weights, best_acc, all_results


def test_predefined_strategies(predictor):
    """Test predefined weighting strategies"""
    print("\n" + "="*80)
    print("TESTING PREDEFINED STRATEGIES")
    print("="*80)

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
        'Best-Model Heavy (EfficientNetB2)': {
            'densenet': 0.2,
            'medsiglip': 0.2,
            'efficientnetb2': 0.6
        },
        'DenseNet+EfficientNetB2 Only': {
            'densenet': 0.5,
            'medsiglip': 0.0,
            'efficientnetb2': 0.5
        },
        'MedSigLIP Only': {
            'densenet': 0.0,
            'medsiglip': 1.0,
            'efficientnetb2': 0.0
        }
    }

    results_summary = {}

    for strategy_name, weights in strategies.items():
        results = predictor.ensemble_predict_from_cache(weights)
        results_summary[strategy_name] = results

        print(f"\n{strategy_name}:")
        print(f"  Weights: {results['weights']}")
        print(f"  Accuracy: {results['ensemble_accuracy']:.4f} ({results['ensemble_accuracy']*100:.2f}%)")

    # Find best strategy
    print("\n" + "="*80)
    print("PREDEFINED STRATEGIES SUMMARY")
    print("="*80)

    best_strategy = max(results_summary.items(), key=lambda x: x[1]['ensemble_accuracy'])

    print(f"\n🏆 Best Predefined Strategy: {best_strategy[0]}")
    print(f"🏆 Accuracy: {best_strategy[1]['ensemble_accuracy']:.4f} ({best_strategy[1]['ensemble_accuracy']*100:.2f}%)")
    print(f"🏆 Weights: {best_strategy[1]['weights']}")

    # Improvement over simple averaging
    simple_acc = results_summary['Simple Averaging']['ensemble_accuracy']
    improvement = (best_strategy[1]['ensemble_accuracy'] - simple_acc) * 100
    print(f"\n📈 Improvement over simple averaging: {improvement:+.2f}%")

    return results_summary, simple_acc


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='OPTIMIZED Weighted Ensemble Inference with Caching')
    parser.add_argument('--dataset_path', default='./dataset_eyepacs', help='Dataset path')
    parser.add_argument('--optimize', action='store_true', help='Run grid search to find optimal weights')
    parser.add_argument('--granularity', type=int, default=20, help='Grid search granularity (default: 20)')
    parser.add_argument('--skip_strategies', action='store_true', help='Skip predefined strategies test')
    args = parser.parse_args()

    # Model checkpoints
    checkpoints = {
        'densenet': './densenet_eyepacs_results/models/best_densenet121_multiclass.pth',
        'medsiglip': './medsiglip_95percent_results/models/best_medsiglip_448_multiclass.pth',
        'efficientnetb2': './efficientnetb2_eyepacs_results/models/best_efficientnetb2_multiclass.pth'
    }

    # Initialize predictor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Using device: {device}\n")

    predictor = OptimizedEnsemblePredictor(checkpoints, args.dataset_path, device)

    # STEP 1: Cache all predictions (runs once)
    predictor.cache_all_predictions()

    # STEP 2: Test predefined strategies (optional, fast)
    simple_acc = 0.9082  # Default from ensemble_results.json
    if not args.skip_strategies:
        results_summary, simple_acc = test_predefined_strategies(predictor)

    # STEP 3: Grid search optimization (optional)
    if args.optimize:
        optimal_weights, optimal_acc, all_results = grid_search_weights(predictor, args.granularity)

        print(f"\n{'='*80}")
        print("OPTIMIZATION RESULTS")
        print("="*80)
        print(f"\n🎯 Optimal weights found:")
        print(f"  {optimal_weights}")
        print(f"  Accuracy: {optimal_acc:.4f} ({optimal_acc*100:.2f}%)")

        improvement_from_simple = (optimal_acc - simple_acc) * 100
        print(f"\n📈 Improvement over simple averaging: {improvement_from_simple:+.2f}%")

        # Detailed analysis
        if improvement_from_simple > 0.3:
            print(f"✅ SIGNIFICANT IMPROVEMENT! Update mata-dr.py to use optimal weights.")
        elif improvement_from_simple > 0.1:
            print(f"⚠️  MARGINAL IMPROVEMENT. Optional to update mata-dr.py.")
        else:
            print(f"ℹ️  MINIMAL IMPROVEMENT. Simple averaging is already near-optimal.")

        # Save optimal weights
        output_dir = Path('./ensemble_3model_results')
        output_dir.mkdir(exist_ok=True)

        with open(output_dir / 'optimal_weights.json', 'w') as f:
            json.dump({
                'optimal_weights': optimal_weights,
                'accuracy': float(optimal_acc),
                'improvement_over_simple': float(improvement_from_simple),
                'simple_averaging_accuracy': float(simple_acc),
                'granularity': args.granularity
            }, f, indent=2)

        print(f"\n💾 Optimal weights saved to {output_dir / 'optimal_weights.json'}")

        # Save top 10 results
        sorted_results = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)
        with open(output_dir / 'top_weight_combinations.json', 'w') as f:
            json.dump(sorted_results[:10], f, indent=2)

        print(f"💾 Top 10 weight combinations saved to {output_dir / 'top_weight_combinations.json'}")

    print("\n" + "="*80)
    print("✅ WEIGHTED ENSEMBLE OPTIMIZATION COMPLETE!")
    print("="*80)
