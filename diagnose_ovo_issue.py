#!/usr/bin/env python3
"""
OVO Issue Diagnostic Script

This script diagnoses why the individual binary classifiers are performing
much lower than expected (76-80% vs 91-92% from analysis).
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from itertools import combinations
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BinaryClassifier(nn.Module):
    """Binary classifier - matches your trained model architecture exactly"""

    def __init__(self, model_name='mobilenet_v2', freeze_weights=True, dropout=0.5):
        super().__init__()
        self.model_name = model_name

        if model_name == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=True)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif model_name == 'inception_v3':
            self.backbone = models.inception_v3(pretrained=True)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.backbone.training = True
        elif model_name == 'densenet121':
            self.backbone = models.densenet121(pretrained=True)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        if freeze_weights:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # MATCH TRAINED MODEL ARCHITECTURE
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),                    # Layer 0: Dropout
            nn.Linear(num_features, 512),          # Layer 1: Linear reduction
            nn.ReLU(),                             # Layer 2: Activation
            nn.BatchNorm1d(512),                   # Layer 3: BatchNorm
            nn.Dropout(dropout),                   # Layer 4: Dropout
            nn.Linear(512, 128),                   # Layer 5: Further reduction
            nn.ReLU(),                             # Layer 6: Activation
            nn.BatchNorm1d(128),                   # Layer 7: BatchNorm
            nn.Dropout(dropout),                   # Layer 8: Dropout
            nn.Linear(128, 1),                     # Layer 9: Final output
            nn.Sigmoid()                           # Layer 10: Sigmoid activation
        )

    def forward(self, x):
        if self.model_name == 'inception_v3' and x.size(-1) < 75:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        if self.model_name == 'inception_v3' and self.training:
            features, aux_features = self.backbone(x)
        else:
            features = self.backbone(x)

        if isinstance(features, tuple):
            features = features[0]

        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        return self.classifier(features)

def create_binary_test_dataset(base_dataset, class_a, class_b, transform):
    """Create binary dataset for specific class pair testing"""
    from torch.utils.data import Dataset

    class BinaryTestDataset(Dataset):
        def __init__(self, base_dataset, class_a, class_b, transform=None):
            self.base_dataset = base_dataset
            self.class_a = class_a
            self.class_b = class_b
            self.transform = transform

            # Filter indices for binary classes
            self.indices = []
            self.labels = []

            for idx in range(len(base_dataset)):
                _, label = base_dataset[idx]
                if label == class_a:
                    self.indices.append(idx)
                    self.labels.append(0)  # Binary label 0
                elif label == class_b:
                    self.indices.append(idx)
                    self.labels.append(1)  # Binary label 1

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            original_idx = self.indices[idx]
            image, _ = self.base_dataset[original_idx]
            binary_label = self.labels[idx]

            if self.transform:
                image = self.transform(image)

            return image, binary_label

    return BinaryTestDataset(base_dataset, class_a, class_b, transform)

def test_individual_binary_classifier(model_name, class_a, class_b, results_dir, dataset_path):
    """Test individual binary classifier performance"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load trained model
    model_path = Path(results_dir) / "models" / f"best_{model_name}_{class_a}_{class_b}.pth"

    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}")
        return None

    # Create and load model
    model = BinaryClassifier(model_name=model_name, freeze_weights=True, dropout=0.5)

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            training_accuracy = checkpoint.get('best_val_accuracy', 'Unknown')
        else:
            model.load_state_dict(checkpoint)
            training_accuracy = 'Unknown'

    except Exception as e:
        logger.error(f"Failed to load {model_path}: {e}")
        return None

    model = model.to(device)
    model.eval()

    # Prepare test data for this specific binary pair
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load base test dataset (without transform to avoid double transform)
    base_test_dataset = ImageFolder(f"{dataset_path}/test", transform=None)

    # Create binary dataset for this specific pair
    binary_test_dataset = create_binary_test_dataset(
        base_test_dataset, class_a, class_b, test_transform
    )

    if len(binary_test_dataset) == 0:
        logger.warning(f"No test samples found for classes {class_a} vs {class_b}")
        return None

    test_loader = DataLoader(
        binary_test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Evaluate binary classifier
    all_predictions = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images).squeeze()

            # Handle single sample case
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)

            # Binary predictions (threshold = 0.5)
            predictions = (outputs > 0.5).float()

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(outputs.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)

    # Calculate confidence distribution
    probabilities = np.array(all_probabilities)
    confidence = np.abs(probabilities - 0.5) * 2  # True confidence

    result = {
        'model_name': model_name,
        'class_pair': f"{class_a}_{class_b}",
        'test_accuracy': accuracy,
        'training_accuracy': training_accuracy,
        'test_samples': len(binary_test_dataset),
        'class_distribution': {
            f'class_{class_a}': sum(1 for t in all_targets if t == 0),
            f'class_{class_b}': sum(1 for t in all_targets if t == 1)
        },
        'confidence_stats': {
            'mean_confidence': float(np.mean(confidence)),
            'std_confidence': float(np.std(confidence)),
            'low_confidence_pct': float(np.mean(confidence < 0.3) * 100)
        },
        'probability_stats': {
            'mean_probability': float(np.mean(probabilities)),
            'std_probability': float(np.std(probabilities))
        }
    }

    return result

def diagnose_ovo_performance():
    """Complete diagnostic analysis of OVO performance issues"""

    print("🔍 OVO PERFORMANCE DIAGNOSTIC")
    print("=" * 50)

    results_dir = "./ovo_ensemble_results_v2"
    dataset_path = "./dataset6"

    # Test all binary classifiers individually
    base_models = ['mobilenet_v2', 'inception_v3', 'densenet121']
    num_classes = 5
    class_pairs = list(combinations(range(num_classes), 2))

    all_results = []

    print(f"🧪 Testing {len(base_models) * len(class_pairs)} binary classifiers...")

    for model_name in base_models:
        print(f"\n📊 Testing {model_name}:")
        model_results = []

        for class_a, class_b in class_pairs:
            result = test_individual_binary_classifier(
                model_name, class_a, class_b, results_dir, dataset_path
            )

            if result:
                model_results.append(result)
                print(f"  Classes {class_a}-{class_b}: {result['test_accuracy']:.3f} "
                      f"(Training: {result['training_accuracy']}) "
                      f"Samples: {result['test_samples']}")

        all_results.extend(model_results)

        # Model summary
        if model_results:
            accuracies = [r['test_accuracy'] for r in model_results]
            print(f"  📈 {model_name} Average: {np.mean(accuracies):.3f} "
                  f"(Range: {min(accuracies):.3f}-{max(accuracies):.3f})")

    # Overall analysis
    print(f"\n🎯 DIAGNOSTIC SUMMARY:")
    print("=" * 50)

    if all_results:
        # Accuracy analysis
        test_accuracies = [r['test_accuracy'] for r in all_results]
        print(f"📊 Individual Binary Classifier Performance:")
        print(f"   Average Test Accuracy: {np.mean(test_accuracies):.3f}")
        print(f"   Std Dev: {np.std(test_accuracies):.3f}")
        print(f"   Range: {min(test_accuracies):.3f} - {max(test_accuracies):.3f}")

        # Compare with your original analysis results
        expected_accuracies = [0.923, 0.911, 0.996, 0.992, 0.846, 0.903, 0.911, 0.950, 0.933, 0.821,  # mobilenet_v2
                             0.921, 0.891, 0.995, 0.987, 0.851, 0.910, 0.905, 0.951, 0.922, 0.813,    # inception_v3
                             0.920, 0.913, 0.998, 0.993, 0.852, 0.913, 0.923, 0.953, 0.950, 0.844]    # densenet121

        print(f"\n📈 COMPARISON WITH ORIGINAL ANALYSIS:")
        print(f"   Expected Average: {np.mean(expected_accuracies):.3f}")
        print(f"   Current Average:  {np.mean(test_accuracies):.3f}")
        print(f"   Gap: {np.mean(expected_accuracies) - np.mean(test_accuracies):.3f}")

        # Confidence analysis
        confidences = [r['confidence_stats']['mean_confidence'] for r in all_results if 'confidence_stats' in r]
        if confidences:
            print(f"\n🎯 CONFIDENCE ANALYSIS:")
            print(f"   Average Confidence: {np.mean(confidences):.3f}")
            print(f"   Low Confidence (<30%) Rate: {np.mean([r['confidence_stats']['low_confidence_pct'] for r in all_results]):.1f}%")

        # Sample size analysis
        sample_sizes = [r['test_samples'] for r in all_results]
        print(f"\n📊 TEST SAMPLE ANALYSIS:")
        print(f"   Average Samples per Binary Test: {np.mean(sample_sizes):.1f}")
        print(f"   Range: {min(sample_sizes)} - {max(sample_sizes)}")

        # Save detailed results
        output_path = Path(results_dir) / "results" / "diagnostic_analysis.json"
        output_path.parent.mkdir(exist_ok=True)

        diagnostic_summary = {
            'individual_results': all_results,
            'summary_stats': {
                'avg_test_accuracy': float(np.mean(test_accuracies)),
                'std_test_accuracy': float(np.std(test_accuracies)),
                'min_test_accuracy': float(min(test_accuracies)),
                'max_test_accuracy': float(max(test_accuracies)),
                'expected_avg_accuracy': float(np.mean(expected_accuracies)),
                'performance_gap': float(np.mean(expected_accuracies) - np.mean(test_accuracies))
            }
        }

        with open(output_path, 'w') as f:
            json.dump(diagnostic_summary, f, indent=2)

        print(f"\n💾 Detailed results saved: {output_path}")

        # Diagnostic conclusions
        performance_gap = np.mean(expected_accuracies) - np.mean(test_accuracies)

        print(f"\n🔍 DIAGNOSTIC CONCLUSIONS:")
        if performance_gap > 0.1:
            print("❌ ISSUE IDENTIFIED: Significant performance gap detected")
            print("   Possible causes:")
            print("   1. Test set different from original analysis")
            print("   2. Model loading/architecture mismatch")
            print("   3. Different evaluation protocol")
        elif performance_gap > 0.05:
            print("⚠️ MODERATE GAP: Some performance difference detected")
            print("   Voting mechanism improvements may still help")
        else:
            print("✅ MODELS PERFORMING AS EXPECTED")
            print("   Issue is primarily in voting aggregation")

    print("=" * 50)

if __name__ == "__main__":
    diagnose_ovo_performance()