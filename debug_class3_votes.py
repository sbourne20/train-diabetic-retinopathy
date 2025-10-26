#!/usr/bin/env python3
"""
Trace Class 3 voting to find where it's failing
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

def analyze_class_3_voting():
    """Analyze how Class 3 samples are being voted on."""

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

    # Filter for Class 3 only
    class_3_indices = [i for i, (_, label) in enumerate(test_dataset) if label == 3]
    print(f"Found {len(class_3_indices)} Class 3 samples")

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

    # Analyze first 20 Class 3 samples
    vote_analysis = {i: [] for i in range(5)}
    predictions = []

    with torch.no_grad():
        for idx in class_3_indices[:20]:
            image, _ = test_dataset[idx]
            image = image.unsqueeze(0).to(device)

            # Get predictions from each binary classifier
            model_name = 'coatnet_0_rw_224'
            pair_votes = {}

            for class_a, class_b in ovo_ensemble.class_pairs:
                pair_name = f"pair_{class_a}_{class_b}"
                classifier = ovo_ensemble.classifiers[model_name][pair_name]

                logits = classifier(image).squeeze()
                prob = torch.sigmoid(logits).item()

                # Determine vote
                if prob > 0.5:
                    winner = class_b
                else:
                    winner = class_a

                pair_votes[pair_name] = {
                    'prob': prob,
                    'winner': winner,
                    'class_a': class_a,
                    'class_b': class_b
                }

            # Get ensemble prediction
            outputs = ovo_ensemble(image)
            votes = outputs['logits'][0].cpu().numpy()
            pred = votes.argmax()

            predictions.append(pred)

            # Track votes
            for class_idx in range(5):
                vote_analysis[class_idx].append(votes[class_idx])

            # Print first 5 samples in detail
            if idx < class_3_indices[5]:
                print(f"\n{'='*80}")
                print(f"Class 3 Sample {idx}")
                print(f"Predicted: Class {pred} ({'✅ CORRECT' if pred == 3 else '❌ WRONG'})")
                print(f"Vote distribution: {votes}")
                print(f"\nPair-by-pair votes:")
                for pair_name, info in pair_votes.items():
                    marker = "✅" if (3 in [info['class_a'], info['class_b']] and info['winner'] == 3) else \
                             "❌" if (3 in [info['class_a'], info['class_b']] and info['winner'] != 3) else \
                             "⚪"
                    print(f"  {marker} {pair_name}: prob={info['prob']:.3f}, winner=Class {info['winner']}")

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY FOR 20 CLASS 3 SAMPLES")
    print(f"{'='*80}")

    predictions = np.array(predictions)
    print(f"\nPrediction distribution:")
    for class_idx in range(5):
        count = (predictions == class_idx).sum()
        pct = count / len(predictions) * 100
        marker = "✅" if class_idx == 3 else "❌"
        print(f"  {marker} Predicted as Class {class_idx}: {count}/20 ({pct:.1f}%)")

    print(f"\nAverage votes per class:")
    for class_idx in range(5):
        avg_votes = np.mean(vote_analysis[class_idx])
        std_votes = np.std(vote_analysis[class_idx])
        marker = "🎯" if class_idx == 3 else "  "
        print(f"  {marker} Class {class_idx}: {avg_votes:.2f} ± {std_votes:.2f} votes")

if __name__ == '__main__':
    analyze_class_3_voting()
