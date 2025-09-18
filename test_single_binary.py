#!/usr/bin/env python3
"""
Test a single binary classifier to verify training worked
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import sys
from pathlib import Path

class BinaryClassifier(nn.Module):
    """Binary classifier with frozen pre-trained backbone."""

    def __init__(self, model_name='mobilenet_v2', freeze_weights=True, dropout=0.5):
        super().__init__()
        self.model_name = model_name

        # Load pre-trained model
        if model_name == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=True)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif model_name == 'inception_v3':
            self.backbone = models.inception_v3(pretrained=True)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == 'densenet121':
            self.backbone = models.densenet121(pretrained=True)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()

        # Freeze backbone weights if specified
        if freeze_weights:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Forward pass for binary classification."""
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        return self.classifier(features)

def test_binary_classifier():
    """Test a single binary classifier"""

    print("🧪 Testing Single Binary Classifier")
    print("="*50)

    # Load one binary classifier (mobilenet_v2 classes 0 vs 1)
    model_path = "./ovo_ensemble_results/models/best_mobilenet_v2_0_1.pth"

    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return

    # Create model
    model = BinaryClassifier(model_name='mobilenet_v2')

    # Load weights
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"✅ Loaded model: {model_path}")
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return

    # Test forward pass
    model.eval()

    # Create dummy input (batch of 2, 299x299 RGB images)
    dummy_input = torch.randn(2, 3, 299, 299)

    with torch.no_grad():
        output = model(dummy_input)
        print(f"✅ Forward pass successful")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output values: {output.squeeze()}")

        # Check if outputs are reasonable (between 0 and 1 for sigmoid)
        if torch.all(output >= 0) and torch.all(output <= 1):
            print("✅ Output range valid (0-1)")
        else:
            print("❌ Output range invalid")

if __name__ == "__main__":
    test_binary_classifier()