#!/usr/bin/env python3
"""
Quick test to verify transform pipeline is working
"""

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np

def test_transform_pipeline():
    """Test that our transform pipeline works correctly."""

    print("🧪 Testing Transform Pipeline...")

    # Test transforms (same as OVO training)
    test_transforms = [
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    transform = transforms.Compose(test_transforms)

    # Test with base dataset (no transforms)
    try:
        base_dataset = ImageFolder("./dataset6/train", transform=None)
        print(f"✅ Base dataset loaded: {len(base_dataset)} images")

        # Test single image
        image, label = base_dataset[0]
        print(f"✅ Base image type: {type(image)}")
        print(f"✅ Base image size: {image.size if hasattr(image, 'size') else 'N/A'}")

        # Test transform on PIL image
        transformed = transform(image)
        print(f"✅ Transformed image type: {type(transformed)}")
        print(f"✅ Transformed image shape: {transformed.shape}")

        print("✅ Transform pipeline working correctly!")
        return True

    except Exception as e:
        print(f"❌ Transform test failed: {e}")
        return False

if __name__ == "__main__":
    test_transform_pipeline()