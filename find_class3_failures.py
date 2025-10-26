#!/usr/bin/env python3
"""
Find Class 3 samples that are being misclassified to understand the failure mode
"""

import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import sys

sys.path.insert(0, str(Path(__file__).parent))
from ensemble_5class_trainer import OVOEnsemble, CLAHETransform

def find_failures():
    """Find Class 3 samples that are misclassified."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup test dataset
    transform = transforms.Compose([
        CLAHETransform(clip_limit=3.0),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(
        './dataset_eyepacs_5class_balanced_enhanced_v2/test',
        transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load ensemble
    ensemble_path = Path('./coatnet_5class_results/models/ovo_ensemble_best.pth')
    ovo_ensemble = OVOEnsemble(
        base_models=['coatnet_0_rw_224'],
        num_classes=5,
        freeze_weights=True,
        dropout=0.28
    )
    state_dict = torch.load(ensemble_path, map_location='cpu', weights_only=False)
    ovo_ensemble.load_state_dict(state_dict)
    ovo_ensemble = ovo_ensemble.to(device)
    ovo_ensemble.eval()

    class_3_correct = 0
    class_3_total = 0
    class_3_pred_as_2 = 0
    class_3_pred_as_4 = 0
    failures = []

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Finding Class 3 failures"):
            images = images.to(device)
            targets = targets.to(device)

            # Get predictions
            outputs = ovo_ensemble(images)
            preds = outputs['logits'].argmax(dim=1)

            # Track Class 3 samples
            class_3_mask = (targets == 3)
            if class_3_mask.sum() > 0:
                class_3_total += class_3_mask.sum().item()
                class_3_correct += ((preds == 3) & class_3_mask).sum().item()
                class_3_pred_as_2 += ((preds == 2) & class_3_mask).sum().item()
                class_3_pred_as_4 += ((preds == 4) & class_3_mask).sum().item()

                # Store first 5 failures
                for i in range(len(targets)):
                    if targets[i] == 3 and preds[i] != 3 and len(failures) < 5:
                        failures.append({
                            'true': 3,
                            'pred': preds[i].item(),
                            'votes': outputs['logits'][i].cpu().numpy(),
                            'image': images[i]
                        })

    print(f"\n{'='*80}")
    print(f"CLASS 3 FAILURE ANALYSIS")
    print(f"{'='*80}")
    print(f"Total Class 3 samples: {class_3_total}")
    print(f"Correctly classified: {class_3_correct} ({class_3_correct/class_3_total*100:.1f}%)")
    print(f"Misclassified as Class 2: {class_3_pred_as_2} ({class_3_pred_as_2/class_3_total*100:.1f}%)")
    print(f"Misclassified as Class 4: {class_3_pred_as_4} ({class_3_pred_as_4/class_3_total*100:.1f}%)")
    print(f"Other misclassifications: {class_3_total - class_3_correct - class_3_pred_as_2 - class_3_pred_as_4}")

    # Analyze failures
    if failures:
        print(f"\n{'='*80}")
        print(f"DETAILED ANALYSIS OF FIRST {len(failures)} FAILURES")
        print(f"{'='*80}")

        for idx, failure in enumerate(failures):
            print(f"\nFailure {idx+1}:")
            print(f"  True: Class 3, Predicted: Class {failure['pred']}")
            print(f"  Vote distribution: {failure['votes']}")
            print(f"  Winner: Class {failure['votes'].argmax()} with {failure['votes'].max():.1f} votes")

            # Analyze pair-by-pair for this failure
            print(f"\n  Analyzing individual binary classifiers:")
            image = failure['image'].unsqueeze(0).to(device)

            model_name = 'coatnet_0_rw_224'
            for class_a, class_b in ovo_ensemble.class_pairs:
                pair_name = f"pair_{class_a}_{class_b}"
                classifier = ovo_ensemble.classifiers[model_name][pair_name]

                logits = classifier(image).squeeze()
                prob = torch.sigmoid(logits).item()
                winner = class_b if prob > 0.5 else class_a

                # Mark pairs involving Class 3
                if 3 in [class_a, class_b]:
                    status = "✅" if winner == 3 else "❌"
                    print(f"    {status} {pair_name}: prob={prob:.3f}, winner=Class {winner}")

if __name__ == '__main__':
    find_failures()
