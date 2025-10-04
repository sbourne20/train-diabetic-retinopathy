#!/usr/bin/env python3
"""
Simple 2-Model Ensemble Inference: DenseNet121 + MedSigLIP-448
Averages predictions from both models for improved accuracy
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
from tqdm import tqdm

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import model architectures
from torchvision.models import densenet121, DenseNet121_Weights, efficientnet_b2
from transformers import AutoModel, AutoProcessor

class DenseNet121_DR(nn.Module):
    """DenseNet121 for DR classification - matches ensemble_local_trainer.py"""
    def __init__(self, num_classes=5, dropout=0.2):
        super().__init__()
        self.backbone = densenet121(weights=None)
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()  # Remove original classifier

        # Match the classifier from ensemble_local_trainer.py
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout/2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout/3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout/4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class MedSigLIP_DR(nn.Module):
    """MedSigLIP-448 for DR classification - matches ensemble_local_trainer.py"""
    def __init__(self, num_classes=5, dropout=0.2):
        super().__init__()
        # Load MedSigLIP backbone
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not hf_token:
            raise ValueError("HUGGINGFACE_TOKEN not found in environment")

        self.backbone = AutoModel.from_pretrained(
            "google/medsiglip-448",
            token=hf_token,
            trust_remote_code=True
        )
        num_features = self.backbone.config.vision_config.hidden_size

        # Match the classifier from ensemble_local_trainer.py
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout/2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout/3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout/4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Handle input size
        if x.size(-1) != 448:
            x = F.interpolate(x, size=(448, 448), mode='bilinear', align_corners=False)

        # Get vision features
        vision_outputs = self.backbone.vision_model(pixel_values=x)
        pooled_output = vision_outputs.pooler_output
        return self.classifier(pooled_output)

class EfficientNetB2_DR(nn.Module):
    """EfficientNetB2 for DR classification - matches ensemble_local_trainer.py"""
    def __init__(self, num_classes=5, dropout=0.2):
        super().__init__()
        self.backbone = efficientnet_b2(weights=None)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        # Match the classifier from ensemble_local_trainer.py
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout/2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout/3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout/4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def load_model(checkpoint_path, model_class, num_classes=5, device='cuda'):
    """Load a model from checkpoint"""
    model = model_class(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, checkpoint.get('best_val_accuracy', 0.0)

def get_predictions(model, data_loader, device='cuda', resize_to=None):
    """Get predictions from a single model"""
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Inference"):
            images = images.to(device)

            # Resize if needed (for different model input sizes)
            if resize_to and images.size(-1) != resize_to:
                images = F.interpolate(images, size=(resize_to, resize_to),
                                     mode='bilinear', align_corners=False)

            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_probs), np.array(all_labels)

def ensemble_predictions(probs_list, method='average'):
    """Combine predictions from multiple models"""
    if method == 'average':
        # Simple averaging of probabilities
        ensemble_probs = np.mean(probs_list, axis=0)
    elif method == 'voting':
        # Majority voting
        preds_list = [np.argmax(probs, axis=1) for probs in probs_list]
        ensemble_preds = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=np.array(preds_list)
        )
        return ensemble_preds
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    return ensemble_preds, ensemble_probs

def evaluate_ensemble(
    densenet_path="./densenet_eyepacs_results/models/best_densenet121_multiclass.pth",
    medsiglip_path="./medsiglip_95percent_results/models/best_medsiglip_448_multiclass.pth",
    efficientnetb2_path=None,
    dataset_path="./dataset_eyepacs",
    batch_size=16,
    device='cuda',
    output_dir='./ensemble_results'
):
    """Evaluate 2 or 3-model ensemble"""

    num_models = 3 if efficientnetb2_path else 2

    print("="*60)
    print(f"SIMPLE {num_models}-MODEL ENSEMBLE EVALUATION")
    print("="*60)

    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load models
    print("\n📁 Loading models...")
    densenet_model, densenet_acc = load_model(densenet_path, DenseNet121_DR, device=device)
    print(f"✅ DenseNet121 loaded (Val Acc: {densenet_acc:.4f})")

    medsiglip_model, medsiglip_acc = load_model(medsiglip_path, MedSigLIP_DR, device=device)
    print(f"✅ MedSigLIP-448 loaded (Val Acc: {medsiglip_acc:.4f})")

    efficientnetb2_model = None
    efficientnetb2_acc = 0.0
    if efficientnetb2_path:
        efficientnetb2_model, efficientnetb2_acc = load_model(efficientnetb2_path, EfficientNetB2_DR, device=device)
        print(f"✅ EfficientNetB2 loaded (Val Acc: {efficientnetb2_acc:.4f})")

    # Prepare data loader with appropriate size
    # Load at a reasonable intermediate size, then resize per model
    print(f"\n📊 Loading test dataset from {dataset_path}")
    test_transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Use DenseNet size as base
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(
        root=f"{dataset_path}/test",
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # Use 0 for CPU to avoid multiprocessing issues
    )
    print(f"Test samples: {len(test_dataset)}")

    # Get individual model predictions
    print("\n🔍 Getting DenseNet predictions (299x299)...")
    densenet_preds, densenet_probs, true_labels = get_predictions(
        densenet_model, test_loader, device, resize_to=299
    )
    densenet_individual_acc = accuracy_score(true_labels, densenet_preds)

    print("🔍 Getting MedSigLIP predictions (448x448)...")
    medsiglip_preds, medsiglip_probs, _ = get_predictions(
        medsiglip_model, test_loader, device, resize_to=448
    )
    medsiglip_individual_acc = accuracy_score(true_labels, medsiglip_preds)

    # Get EfficientNetB2 predictions if available
    probs_list = [densenet_probs, medsiglip_probs]
    efficientnetb2_individual_acc = 0.0
    if efficientnetb2_model:
        print("🔍 Getting EfficientNetB2 predictions (299x299)...")
        efficientnetb2_preds, efficientnetb2_probs, _ = get_predictions(
            efficientnetb2_model, test_loader, device, resize_to=299
        )
        efficientnetb2_individual_acc = accuracy_score(true_labels, efficientnetb2_preds)
        probs_list.append(efficientnetb2_probs)

    # Ensemble predictions
    print(f"\n🤝 Creating {num_models}-model ensemble predictions (averaging)...")
    ensemble_preds, ensemble_probs = ensemble_predictions(
        probs_list,
        method='average'
    )
    ensemble_acc = accuracy_score(true_labels, ensemble_preds)

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"DenseNet121 Accuracy:   {densenet_individual_acc:.4f} ({densenet_individual_acc*100:.2f}%)")
    print(f"MedSigLIP-448 Accuracy: {medsiglip_individual_acc:.4f} ({medsiglip_individual_acc*100:.2f}%)")
    if efficientnetb2_model:
        print(f"EfficientNetB2 Accuracy: {efficientnetb2_individual_acc:.4f} ({efficientnetb2_individual_acc*100:.2f}%)")
    print(f"{num_models}-Model Ensemble:     {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")
    print(f"\nImprovement over DenseNet: {(ensemble_acc - densenet_individual_acc)*100:+.2f}%")
    print(f"Improvement over MedSigLIP: {(ensemble_acc - medsiglip_individual_acc)*100:+.2f}%")
    if efficientnetb2_model:
        print(f"Improvement over EfficientNetB2: {(ensemble_acc - efficientnetb2_individual_acc)*100:+.2f}%")
    print("="*60)

    # Classification report
    class_names = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'PDR']
    print("\n📊 ENSEMBLE CLASSIFICATION REPORT:")
    print(classification_report(true_labels, ensemble_preds, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(true_labels, ensemble_preds)
    print("📊 CONFUSION MATRIX:")
    print(cm)

    # Save results
    results = {
        'individual_models': {
            'densenet121': {
                'checkpoint': str(densenet_path),
                'val_accuracy': float(densenet_acc),
                'test_accuracy': float(densenet_individual_acc)
            },
            'medsiglip_448': {
                'checkpoint': str(medsiglip_path),
                'val_accuracy': float(medsiglip_acc),
                'test_accuracy': float(medsiglip_individual_acc)
            }
        },
        'ensemble': {
            'method': 'average',
            'num_models': num_models,
            'test_accuracy': float(ensemble_acc),
            'improvement_over_densenet': float((ensemble_acc - densenet_individual_acc)*100),
            'improvement_over_medsiglip': float((ensemble_acc - medsiglip_individual_acc)*100)
        },
        'classification_report': classification_report(
            true_labels, ensemble_preds, target_names=class_names, output_dict=True
        ),
        'confusion_matrix': cm.tolist()
    }

    if efficientnetb2_model:
        results['individual_models']['efficientnetb2'] = {
            'checkpoint': str(efficientnetb2_path),
            'val_accuracy': float(efficientnetb2_acc),
            'test_accuracy': float(efficientnetb2_individual_acc)
        }
        results['ensemble']['improvement_over_efficientnetb2'] = float((ensemble_acc - efficientnetb2_individual_acc)*100)

    results_file = f"{output_dir}/ensemble_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {results_file}")

    # Medical-grade assessment
    print("\n🏥 MEDICAL-GRADE ASSESSMENT:")
    if ensemble_acc >= 0.90:
        print("✅ FULL PASS - Production Ready (≥90%)")
    elif ensemble_acc >= 0.85:
        print("⚠️ NEAR PASS - Close to Production (≥85%)")
    elif ensemble_acc >= 0.80:
        print("📈 PROMISING LEVEL - Research Quality (≥80%)")
    else:
        print("❌ NEEDS IMPROVEMENT - Below Standards (<80%)")

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple 2 or 3-Model Ensemble Inference")
    parser.add_argument("--densenet_checkpoint", type=str,
                       default="./densenet_eyepacs_results/models/best_densenet121_multiclass.pth",
                       help="Path to DenseNet checkpoint")
    parser.add_argument("--medsiglip_checkpoint", type=str,
                       default="./medsiglip_95percent_results/models/best_medsiglip_448_multiclass.pth",
                       help="Path to MedSigLIP checkpoint")
    parser.add_argument("--efficientnetb2_checkpoint", type=str, default=None,
                       help="Path to EfficientNetB2 checkpoint (optional - for 3-model ensemble)")
    parser.add_argument("--dataset_path", type=str, default="./dataset_eyepacs",
                       help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--output_dir", type=str, default="./ensemble_results",
                       help="Output directory for results")

    args = parser.parse_args()

    results = evaluate_ensemble(
        densenet_path=args.densenet_checkpoint,
        medsiglip_path=args.medsiglip_checkpoint,
        efficientnetb2_path=args.efficientnetb2_checkpoint,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir
    )

    print("\n✅ Ensemble evaluation complete!")
