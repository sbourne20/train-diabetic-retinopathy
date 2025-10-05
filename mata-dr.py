#!/usr/bin/env python3
"""
MATA-DR: Medical AI Tool for Diabetic Retinopathy Grading
Single Image Inference using DenseNet121 + MedSigLIP-448 + EfficientNetB2 Ensemble

Usage:
    python mata-dr.py --file ./test_image/40014_left.jpeg
    python mata-dr.py --file ./test_image/40014_left.jpeg --model densenet
    python mata-dr.py --file ./test_image/40014_left.jpeg --model medsiglip
    python mata-dr.py --file ./test_image/40014_left.jpeg --model efficientnetb2
    python mata-dr.py --file ./test_image/40014_left.jpeg --model ensemble
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import argparse

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import model architectures
from torchvision.models import densenet121, efficientnet_b2
from transformers import AutoModel

class DenseNet121_DR(nn.Module):
    """DenseNet121 for DR classification"""
    def __init__(self, num_classes=5, dropout=0.2):
        super().__init__()
        self.backbone = densenet121(weights=None)
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()

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

class EfficientNetB2_DR(nn.Module):
    """EfficientNetB2 for DR classification"""
    def __init__(self, num_classes=5, dropout=0.2):
        super().__init__()
        self.backbone = efficientnet_b2(weights=None)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

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
    """MedSigLIP-448 for DR classification"""
    def __init__(self, num_classes=5, dropout=0.2):
        super().__init__()
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not hf_token:
            raise ValueError("HUGGINGFACE_TOKEN not found in .env file")

        self.backbone = AutoModel.from_pretrained(
            "google/medsiglip-448",
            token=hf_token,
            trust_remote_code=True
        )
        num_features = self.backbone.config.vision_config.hidden_size

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
        if x.size(-1) != 448:
            x = F.interpolate(x, size=(448, 448), mode='bilinear', align_corners=False)
        vision_outputs = self.backbone.vision_model(pixel_values=x)
        pooled_output = vision_outputs.pooler_output
        return self.classifier(pooled_output)

class MATADR:
    """Medical AI Tool for Diabetic Retinopathy Grading"""

    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.class_names = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'PDR']
        self.class_descriptions = {
            0: {
                'name': 'No Diabetic Retinopathy',
                'severity': 'Normal',
                'recommendation': 'Annual screening recommended',
                'clinical': 'No abnormalities detected'
            },
            1: {
                'name': 'Mild Non-Proliferative Diabetic Retinopathy',
                'severity': 'Mild',
                'recommendation': 'Annual follow-up recommended',
                'clinical': 'Microaneurysms only'
            },
            2: {
                'name': 'Moderate Non-Proliferative Diabetic Retinopathy',
                'severity': 'Moderate',
                'recommendation': '6-month follow-up recommended',
                'clinical': 'Multiple hemorrhages, exudates, or cotton wool spots'
            },
            3: {
                'name': 'Severe Non-Proliferative Diabetic Retinopathy',
                'severity': 'Severe',
                'recommendation': '3-month follow-up, urgent referral consideration',
                'clinical': 'Extensive hemorrhages, venous beading, or IRMA'
            },
            4: {
                'name': 'Proliferative Diabetic Retinopathy',
                'severity': 'Sight-Threatening',
                'recommendation': 'IMMEDIATE ophthalmology referral required',
                'clinical': 'Neovascularization, vitreous hemorrhage'
            }
        }

        self.models = {}
        self.model_paths = {
            'densenet': './densenet_eyepacs_results/models/best_densenet121_multiclass.pth',
            'medsiglip': './medsiglip_95percent_results/models/best_medsiglip_448_multiclass.pth',
            'efficientnetb2': './efficientnetb2_eyepacs_results/models/best_efficientnetb2_multiclass.pth'
        }

    def load_model(self, model_type):
        """Load a specific model"""
        if model_type in self.models:
            return self.models[model_type]

        checkpoint_path = self.model_paths[model_type]
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        print(f"Loading {model_type.upper()} model...")

        if model_type == 'densenet':
            model = DenseNet121_DR(num_classes=5)
        elif model_type == 'medsiglip':
            model = MedSigLIP_DR(num_classes=5)
        elif model_type == 'efficientnetb2':
            model = EfficientNetB2_DR(num_classes=5)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        self.models[model_type] = {
            'model': model,
            'val_acc': checkpoint.get('best_val_accuracy', 0.0)
        }

        print(f"✅ {model_type.upper()} loaded (Val Acc: {checkpoint.get('best_val_accuracy', 0.0):.4f})")
        return self.models[model_type]

    def preprocess_image(self, image_path, target_size=299):
        """Load and preprocess image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Transform
        transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor.to(self.device)

    def predict_single_model(self, image_path, model_type, resize_to=None):
        """Get prediction from a single model"""
        # Load model
        model_info = self.load_model(model_type)
        model = model_info['model']

        # Preprocess image
        image_tensor = self.preprocess_image(image_path, target_size=resize_to or 299)

        # Resize if needed for specific model
        if resize_to and image_tensor.size(-1) != resize_to:
            image_tensor = F.interpolate(image_tensor, size=(resize_to, resize_to),
                                        mode='bilinear', align_corners=False)

        # Predict
        with torch.no_grad():
            logits = model(image_tensor)
            probs = F.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()

        return pred_class, confidence, probs[0].cpu().numpy()

    def predict_ensemble(self, image_path):
        """Get ensemble prediction from all three models"""
        # DenseNet prediction
        densenet_class, densenet_conf, densenet_probs = self.predict_single_model(
            image_path, 'densenet', resize_to=299
        )

        # MedSigLIP prediction
        medsiglip_class, medsiglip_conf, medsiglip_probs = self.predict_single_model(
            image_path, 'medsiglip', resize_to=448
        )

        # EfficientNetB2 prediction
        efficientnetb2_class, efficientnetb2_conf, efficientnetb2_probs = self.predict_single_model(
            image_path, 'efficientnetb2', resize_to=224
        )

        # Average probabilities from all three models
        ensemble_probs = (densenet_probs + medsiglip_probs + efficientnetb2_probs) / 3
        ensemble_class = np.argmax(ensemble_probs)
        ensemble_conf = ensemble_probs[ensemble_class]

        return {
            'ensemble': {
                'class': int(ensemble_class),
                'confidence': float(ensemble_conf),
                'probabilities': ensemble_probs.tolist()
            },
            'densenet': {
                'class': int(densenet_class),
                'confidence': float(densenet_conf),
                'probabilities': densenet_probs.tolist()
            },
            'medsiglip': {
                'class': int(medsiglip_class),
                'confidence': float(medsiglip_conf),
                'probabilities': medsiglip_probs.tolist()
            },
            'efficientnetb2': {
                'class': int(efficientnetb2_class),
                'confidence': float(efficientnetb2_conf),
                'probabilities': efficientnetb2_probs.tolist()
            }
        }

    def format_result(self, pred_class, confidence, probabilities, show_all_probs=False):
        """Format prediction result for display"""
        info = self.class_descriptions[pred_class]

        result = f"""
╔══════════════════════════════════════════════════════════════╗
║              DIABETIC RETINOPATHY GRADING RESULT             ║
╠══════════════════════════════════════════════════════════════╣
║ Predicted Grade: {info['name']:<44} ║
║ Severity Level:  {info['severity']:<44} ║
║ Confidence:      {confidence*100:>5.2f}%                                      ║
╠══════════════════════════════════════════════════════════════╣
║ Clinical Finding:                                            ║
║   {info['clinical']:<58} ║
║                                                              ║
║ Recommendation:                                              ║
║   {info['recommendation']:<58} ║
╚══════════════════════════════════════════════════════════════╝
"""

        if show_all_probs:
            result += "\n📊 Class Probabilities:\n"
            for i, (class_name, prob) in enumerate(zip(self.class_names, probabilities)):
                bar_length = int(prob * 40)
                bar = '█' * bar_length + '░' * (40 - bar_length)
                result += f"  {class_name:<18} {bar} {prob*100:>5.2f}%\n"

        return result

def main():
    parser = argparse.ArgumentParser(
        description='MATA-DR: Medical AI Tool for Diabetic Retinopathy Grading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ensemble prediction (recommended)
  python mata-dr.py --file ./test_image/40014_left.jpeg

  # Single model prediction
  python mata-dr.py --file ./test_image/40014_left.jpeg --model densenet
  python mata-dr.py --file ./test_image/40014_left.jpeg --model medsiglip
  python mata-dr.py --file ./test_image/40014_left.jpeg --model efficientnetb2

  # Show detailed probabilities
  python mata-dr.py --file ./test_image/40014_left.jpeg --verbose
        """
    )

    parser.add_argument('--file', type=str, required=True,
                       help='Path to fundus image (JPEG/PNG)')
    parser.add_argument('--model', type=str, default='ensemble',
                       choices=['ensemble', 'densenet', 'medsiglip', 'efficientnetb2'],
                       help='Model to use for prediction (default: ensemble)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed probabilities for all classes')

    args = parser.parse_args()

    # Initialize MATA-DR
    print("🏥 MATA-DR: Medical AI Tool for Diabetic Retinopathy")
    print("="*62)

    mata = MATADR(device=args.device)

    print(f"📁 Image: {args.file}")
    print(f"🤖 Model: {args.model.upper()}")
    print(f"💻 Device: {mata.device}")
    print()

    # Make prediction
    try:
        if args.model == 'ensemble':
            print("🔍 Running ensemble prediction (DenseNet121 + MedSigLIP-448 + EfficientNetB2)...")
            results = mata.predict_ensemble(args.file)

            # Display ensemble result
            print(mata.format_result(
                results['ensemble']['class'],
                results['ensemble']['confidence'],
                results['ensemble']['probabilities'],
                show_all_probs=args.verbose
            ))

            if args.verbose:
                print("\n📋 Individual Model Predictions:")
                print(f"\n  DenseNet121 (299x299):")
                print(f"    Predicted: {mata.class_names[results['densenet']['class']]}")
                print(f"    Confidence: {results['densenet']['confidence']*100:.2f}%")
                print(f"\n  MedSigLIP-448 (448x448):")
                print(f"    Predicted: {mata.class_names[results['medsiglip']['class']]}")
                print(f"    Confidence: {results['medsiglip']['confidence']*100:.2f}%")
                print(f"\n  EfficientNetB2 (224x224):")
                print(f"    Predicted: {mata.class_names[results['efficientnetb2']['class']]}")
                print(f"    Confidence: {results['efficientnetb2']['confidence']*100:.2f}%")
        else:
            print(f"🔍 Running {args.model.upper()} prediction...")
            if args.model == 'densenet':
                resize_to = 299
            elif args.model == 'medsiglip':
                resize_to = 448
            elif args.model == 'efficientnetb2':
                resize_to = 224
            else:
                resize_to = 299

            pred_class, confidence, probabilities = mata.predict_single_model(
                args.file, args.model, resize_to=resize_to
            )

            print(mata.format_result(
                pred_class, confidence, probabilities,
                show_all_probs=args.verbose
            ))

        print("\n✅ Prediction completed successfully!")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
