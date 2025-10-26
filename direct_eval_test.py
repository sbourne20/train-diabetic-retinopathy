#!/usr/bin/env python3
"""
Direct evaluation to debug the 73% vs 93% discrepancy
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, confusion_matrix
import sys

sys.path.insert(0, str(Path(__file__).parent))
from ensemble_5class_trainer import OVOEnsemble, CLAHETransform

def direct_evaluation():
    """Run evaluation exactly like the official code."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup test dataset (EXACTLY like official evaluation)
    test_transform = transforms.Compose([
        CLAHETransform(clip_limit=3.0),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(
        './dataset_eyepacs_5class_balanced_enhanced_v2/test',
        transform=test_transform
    )

    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=4)

    print(f"Test dataset: {len(test_dataset)} images")
    print(f"Batches: {len(test_loader)}")

    # Load ensemble (EXACTLY like official evaluation)
    ovo_ensemble = OVOEnsemble(
        base_models=['coatnet_0_rw_224'],
        num_classes=5,
        freeze_weights=True,
        dropout=0.28
    )

    ensemble_path = Path('./coatnet_5class_results/models/ovo_ensemble_best.pth')
    state_dict = torch.load(ensemble_path, map_location='cpu', weights_only=False)
    ovo_ensemble.load_state_dict(state_dict)
    ovo_ensemble = ovo_ensemble.to(device)
    ovo_ensemble.eval()

    print("✅ Ensemble loaded")

    all_predictions = []
    all_targets = []
    sample_votes = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(device)
            targets = targets.to(device)

            # Get ensemble predictions (EXACTLY like official code)
            outputs = ovo_ensemble(images, return_individual=True)

            # Get final predictions (argmax of votes)
            _, ensemble_pred = torch.max(outputs['logits'], 1)
            all_predictions.extend(ensemble_pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Save first batch vote details
            if batch_idx == 0:
                sample_votes = outputs['logits'].cpu().numpy()
                print(f"\nFirst batch votes:")
                for i in range(min(3, len(sample_votes))):
                    print(f"  Sample {i}: votes={sample_votes[i]}, pred={ensemble_pred[i].item()}, true={targets[i].item()}")

    # Calculate metrics
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)

    accuracy = accuracy_score(all_targets, all_predictions)
    cm = confusion_matrix(all_targets, all_predictions)

    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Total samples: {len(all_targets)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print(f"\nPer-class accuracy:")
    for i in range(5):
        class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"  Class {i}: {class_acc:.4f} ({class_acc*100:.2f}%) - {cm[i].sum()} samples")

    print(f"\nConfusion Matrix:")
    print(cm)

    # Check vote statistics
    print(f"\nVote statistics from first batch:")
    print(f"  Min votes: {sample_votes.min()}")
    print(f"  Max votes: {sample_votes.max()}")
    print(f"  Vote sums: {sample_votes.sum(axis=1)[:3]}")

if __name__ == '__main__':
    direct_evaluation()
