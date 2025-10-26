#!/usr/bin/env python3
"""
Deep dive into pair_2_3 classifier to understand why Class 3 fails
"""

import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix, classification_report
import sys

sys.path.insert(0, str(Path(__file__).parent))
from ensemble_5class_trainer import BinaryClassifier, CLAHETransform

def analyze_pair_2_3():
    """Analyze pair_2_3 classifier in detail."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model_path = Path('./coatnet_5class_results/models/best_coatnet_0_rw_224_2_3.pth')
    model = BinaryClassifier('coatnet_0_rw_224', freeze_weights=True, dropout=0.28)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

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

    # Filter for classes 2 and 3 only
    indices = [i for i, (_, label) in enumerate(test_dataset) if label in [2, 3]]
    filtered_dataset = Subset(test_dataset, indices)
    test_loader = DataLoader(filtered_dataset, batch_size=32, shuffle=False, num_workers=4)

    print("="*80)
    print("🔍 ANALYZING PAIR 2_3 CLASSIFIER")
    print("="*80)
    print(f"Val Accuracy from checkpoint: {checkpoint.get('best_val_accuracy', 0):.2f}%")
    print()

    all_predictions = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            targets = targets.to(device)

            # Convert to binary (0 = class 2, 1 = class 3)
            binary_targets = (targets == 3).long()

            # Get predictions
            logits = model(images).squeeze()
            probs = torch.sigmoid(logits)
            binary_preds = (probs > 0.5).long()

            all_predictions.extend(binary_preds.cpu().numpy())
            all_targets.extend(binary_targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)

    # Overall metrics
    print(f"Total samples: {len(all_targets)}")
    print(f"Class 2 samples: {(all_targets == 0).sum()}")
    print(f"Class 3 samples: {(all_targets == 1).sum()}")
    print()

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    print("Confusion Matrix (rows=true, cols=pred):")
    print("           Pred_2  Pred_3")
    print(f"True_2:    {cm[0,0]:<7} {cm[0,1]:<7}")
    print(f"True_3:    {cm[1,0]:<7} {cm[1,1]:<7}")
    print()

    # Per-class metrics
    class_2_acc = cm[0,0] / cm[0].sum() * 100
    class_3_acc = cm[1,1] / cm[1].sum() * 100
    print(f"Class 2 (Moderate NPDR) Accuracy: {class_2_acc:.2f}%")
    print(f"Class 3 (Severe NPDR) Accuracy: {class_3_acc:.2f}%")
    print(f"Overall Accuracy: {(cm[0,0] + cm[1,1]) / cm.sum() * 100:.2f}%")
    print()

    # Probability distribution
    class_2_probs = all_probs[all_targets == 0]
    class_3_probs = all_probs[all_targets == 1]

    print("Probability Distribution:")
    print(f"Class 2 samples - Mean prob: {class_2_probs.mean():.3f}, Std: {class_2_probs.std():.3f}")
    print(f"Class 3 samples - Mean prob: {class_3_probs.mean():.3f}, Std: {class_3_probs.std():.3f}")
    print()

    # Threshold analysis
    print("Threshold Analysis:")
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds_at_thresh = (all_probs > threshold).astype(int)
        cm_thresh = confusion_matrix(all_targets, preds_at_thresh)
        acc = (cm_thresh[0,0] + cm_thresh[1,1]) / cm_thresh.sum() * 100
        class_2_acc_t = cm_thresh[0,0] / cm_thresh[0].sum() * 100 if cm_thresh[0].sum() > 0 else 0
        class_3_acc_t = cm_thresh[1,1] / cm_thresh[1].sum() * 100 if cm_thresh[1].sum() > 0 else 0
        print(f"  Threshold {threshold:.1f}: Overall={acc:.2f}%, Class2={class_2_acc_t:.2f}%, Class3={class_3_acc_t:.2f}%")

    # Check if there's a bias
    if class_2_acc > class_3_acc + 10:
        print("\n⚠️ WARNING: Classifier is biased towards Class 2!")
        print(f"   Class 2 accuracy ({class_2_acc:.1f}%) >> Class 3 accuracy ({class_3_acc:.1f}%)")
    elif class_3_acc > class_2_acc + 10:
        print("\n⚠️ WARNING: Classifier is biased towards Class 3!")
        print(f"   Class 3 accuracy ({class_3_acc:.1f}%) >> Class 2 accuracy ({class_2_acc:.1f}%)")

if __name__ == '__main__':
    analyze_pair_2_3()
