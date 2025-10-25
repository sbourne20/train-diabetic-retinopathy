#!/usr/bin/env python3
"""
Test Individual OVO Binary Classifiers - DR 5-Class
Verify each binary classifier (0-1, 0-2, ..., 3-4) independently
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import logging
from PIL import Image
import cv2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CLAHETransform:
    """Apply CLAHE to RGB fundus images"""
    def __init__(self, clip_limit=3.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        # Convert PIL to numpy
        img_np = np.array(img)

        # Apply CLAHE per channel
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

        if len(img_np.shape) == 3:
            # RGB image
            img_np[:,:,0] = clahe.apply(img_np[:,:,0])
            img_np[:,:,1] = clahe.apply(img_np[:,:,1])
            img_np[:,:,2] = clahe.apply(img_np[:,:,2])
        else:
            # Grayscale
            img_np = clahe.apply(img_np)

        return Image.fromarray(img_np)


class BinaryClassifier(nn.Module):
    """Binary classifier for OVO ensemble - matches training architecture"""

    def __init__(self, model_name, dropout=0.5):
        super().__init__()
        self.model_name = model_name

        # Load backbone
        if model_name == 'seresnext50_32x4d':
            try:
                import timm
                self.backbone = timm.create_model('seresnext50_32x4d', pretrained=False)
                num_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()
            except ImportError:
                raise ValueError("timm required for SEResNeXt50")

        elif model_name == 'mobilenet_v2':
            from torchvision.models import mobilenet_v2
            self.backbone = mobilenet_v2(pretrained=False)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        elif model_name == 'densenet121':
            from torchvision.models import densenet121
            self.backbone = densenet121(pretrained=False)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()

        elif model_name == 'inception_v3':
            from torchvision.models import inception_v3
            self.backbone = inception_v3(pretrained=False, aux_logits=False)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Binary classification head (matches training)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1)  # Output raw logit
        )

    def forward(self, x):
        # Extract features
        features = self.backbone(x)

        # Handle tuple outputs
        if isinstance(features, tuple):
            features = features[0]

        # Global average pooling if needed
        if len(features.shape) > 2:
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)

        # Binary classification
        return self.classifier(features)


def create_test_transforms(img_size=224, enable_clahe=False):
    """Create test transforms"""
    transform_list = []

    if enable_clahe:
        transform_list.append(CLAHETransform(clip_limit=3.0))

    transform_list.extend([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transforms.Compose(transform_list)


def test_binary_classifier(model, dataloader, device):
    """Test a single binary classifier"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            targets = targets.to(device)

            # Get predictions
            logits = model(images).squeeze()
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)

            preds = (torch.sigmoid(logits) > 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate accuracy
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    accuracy = (all_preds == all_targets).mean()

    return accuracy


def test_all_binary_classifiers(models_dir, dataset_path, enable_clahe=False, img_size=224):
    """Test all binary classifiers in directory"""

    models_dir = Path(models_dir)
    dataset_path = Path(dataset_path)

    # Find all binary classifier checkpoints
    binary_models = list(models_dir.glob('best_*_*_*.pth'))
    binary_models = [m for m in binary_models if m.name != 'ovo_ensemble_best.pth']

    logger.info(f"📁 Found {len(binary_models)} binary classifiers")

    # Class names for DR 5-class
    class_names = ['No_DR', 'Mild_NPDR', 'Moderate_NPDR', 'Severe_NPDR', 'PDR']

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 Using device: {device}")

    # Load full dataset
    test_transform = create_test_transforms(img_size=img_size, enable_clahe=enable_clahe)
    full_test_dataset = ImageFolder(root=dataset_path / "test", transform=test_transform)

    logger.info(f"📊 Full test dataset: {len(full_test_dataset)} images across {len(full_test_dataset.classes)} classes")

    results = []

    for model_file in sorted(binary_models):
        try:
            # Parse model name and class pair from filename
            # Format: best_seresnext50_32x4d_0_1.pth
            parts = model_file.stem.replace('best_', '').split('_')

            # Handle model names with underscores (like seresnext50_32x4d)
            class_0 = int(parts[-2])
            class_1 = int(parts[-1])
            model_name = '_'.join(parts[:-2])

            logger.info(f"\n{'='*60}")
            logger.info(f"Testing: {model_file.name}")
            logger.info(f"  Model: {model_name}")
            logger.info(f"  Classes: {class_0} ({class_names[class_0]}) vs {class_1} ({class_names[class_1]})")

            # Load checkpoint
            checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)

            # Extract val accuracy if available
            val_accuracy = None
            if isinstance(checkpoint, dict):
                val_accuracy = checkpoint.get('best_val_accuracy') or checkpoint.get('val_accuracy')
                epoch = checkpoint.get('epoch', 'unknown')
                logger.info(f"  Checkpoint epoch: {epoch}")
                if val_accuracy is not None:
                    logger.info(f"  Val Accuracy: {val_accuracy*100:.2f}%")

            # Create binary dataset (only class_0 and class_1)
            class_0_indices = [i for i, (_, label) in enumerate(full_test_dataset.samples) if label == class_0]
            class_1_indices = [i for i, (_, label) in enumerate(full_test_dataset.samples) if label == class_1]

            # Combine and create subset
            binary_indices = class_0_indices + class_1_indices
            binary_dataset = Subset(full_test_dataset, binary_indices)

            # Create binary targets (0 for class_0, 1 for class_1)
            binary_targets = [0] * len(class_0_indices) + [1] * len(class_1_indices)

            # Wrap in dataset with binary labels
            class BinaryDataset(torch.utils.data.Dataset):
                def __init__(self, subset, binary_targets):
                    self.subset = subset
                    self.binary_targets = binary_targets

                def __len__(self):
                    return len(self.subset)

                def __getitem__(self, idx):
                    image, _ = self.subset[idx]
                    return image, self.binary_targets[idx]

            binary_dataset_wrapped = BinaryDataset(binary_dataset, binary_targets)

            logger.info(f"  Binary test set: {len(binary_dataset_wrapped)} images ({len(class_0_indices)} class {class_0}, {len(class_1_indices)} class {class_1})")

            # Create dataloader
            test_loader = DataLoader(
                binary_dataset_wrapped,
                batch_size=16,
                shuffle=False,
                num_workers=2
            )

            # Create model
            model = BinaryClassifier(model_name=model_name, dropout=0.5)

            # Load weights
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)

            model = model.to(device)

            # Test
            test_accuracy = test_binary_classifier(model, test_loader, device)

            logger.info(f"  ✅ Test Accuracy: {test_accuracy*100:.2f}%")
            if val_accuracy is not None:
                gap = abs(val_accuracy - test_accuracy)
                logger.info(f"  📊 Val-Test Gap: {gap*100:.2f}%")

            # Save result
            result = {
                'model_file': str(model_file.name),
                'model_name': model_name,
                'class_pair': f"{class_0}-{class_1}",
                'class_names': [class_names[class_0], class_names[class_1]],
                'val_accuracy': float(val_accuracy) if val_accuracy is not None else None,
                'test_accuracy': float(test_accuracy),
                'test_samples': len(binary_dataset_wrapped),
                'class_0_samples': len(class_0_indices),
                'class_1_samples': len(class_1_indices)
            }
            results.append(result)

        except Exception as e:
            logger.error(f"❌ Failed to test {model_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test Individual OVO Binary Classifiers for DR 5-Class')
    parser.add_argument('--models_dir', type=str, default='./seresnext50_5class_results/models',
                       help='Directory containing binary classifier checkpoints')
    parser.add_argument('--dataset_path', type=str, default='./dataset3_augmented_resized',
                       help='Path to 5-class dataset (must have test/ subdirectory)')
    parser.add_argument('--enable_clahe', action='store_true',
                       help='Enable CLAHE preprocessing')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size for testing')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')

    args = parser.parse_args()

    # Test all binary classifiers
    results = test_all_binary_classifiers(
        args.models_dir,
        args.dataset_path,
        args.enable_clahe,
        args.img_size
    )

    # Print summary
    logger.info(f"\n{'='*100}")
    logger.info("📊 BINARY CLASSIFIER TEST RESULTS SUMMARY")
    logger.info(f"{'='*100}\n")

    logger.info(f"{'Model':<25} | {'Class Pair':<25} | {'Val Acc':<12} | {'Test Acc':<12} | {'Samples':<10}")
    logger.info('-' * 100)

    for result in results:
        model_name = result['model_name'][:23]
        class_pair_str = f"{result['class_names'][0][:10]} vs {result['class_names'][1][:10]}"
        val_acc = f"{result['val_accuracy']*100:.2f}%" if result['val_accuracy'] is not None else "N/A"
        test_acc = f"{result['test_accuracy']*100:.2f}%"
        samples = result['test_samples']

        logger.info(f"{model_name:<25} | {class_pair_str:<25} | {val_acc:<12} | {test_acc:<12} | {samples:<10}")

    # Calculate averages
    if results:
        avg_val_acc = np.mean([r['val_accuracy'] for r in results if r['val_accuracy'] is not None])
        avg_test_acc = np.mean([r['test_accuracy'] for r in results])

        logger.info('-' * 100)
        logger.info(f"{'AVERAGE':<25} | {'':<25} | {avg_val_acc*100:.2f}%{'':<7} | {avg_test_acc*100:.2f}%")
        logger.info('=' * 100)

        # Medical grade assessment
        logger.info(f"\n🏥 MEDICAL GRADE ASSESSMENT:")
        if avg_test_acc >= 0.95:
            logger.info(f"   ✅ EXCELLENT - Binary classifiers meet medical-grade standards (≥95%)")
        elif avg_test_acc >= 0.90:
            logger.info(f"   ✅ GOOD - Binary classifiers meet acceptable standards (≥90%)")
        elif avg_test_acc >= 0.85:
            logger.info(f"   ⚠️ MODERATE - Binary classifiers need improvement (≥85%)")
        else:
            logger.info(f"   ❌ POOR - Binary classifiers require retraining (<85%)")

    # Save results
    if args.output is None:
        output_file = Path(args.models_dir).parent / 'binary_classifier_test_results.json'
    else:
        output_file = Path(args.output)

    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_models': len(results),
                'avg_val_accuracy': float(avg_val_acc) if results else 0.0,
                'avg_test_accuracy': float(avg_test_acc) if results else 0.0,
            },
            'results': results
        }, f, indent=2)

    logger.info(f"\n💾 Results saved to: {output_file}")
    logger.info("\n✅ Binary classifier testing completed!")


if __name__ == '__main__':
    main()
