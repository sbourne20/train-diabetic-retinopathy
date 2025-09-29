#!/usr/bin/env python3
"""
Test script to verify MedSigLIP model loading fix
"""

import torch
import os
from transformers import AutoModel

def test_medsiglip_loading():
    """Test the fixed MedSigLIP loading to ensure it matches working checkpoint."""

    print("🔬 Testing MedSigLIP model loading fix...")

    # Test environment
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if not hf_token:
        print("❌ HUGGINGFACE_TOKEN not found in environment")
        return False

    try:
        # Load model the same way as fixed code
        backbone = AutoModel.from_pretrained(
            "google/medsiglip-448",
            token=hf_token,
            trust_remote_code=True
        )

        # Check parameter count
        total_params = sum(p.numel() for p in backbone.parameters())
        trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        print(f"✅ Full MedSigLIP model loaded")
        print(f"📊 Total parameters: {total_params:,}")
        print(f"🎯 Trainable parameters: {trainable_params:,}")

        # Apply same freezing logic as working version
        freeze_weights = False  # Same as working config
        if freeze_weights:
            for name, param in backbone.named_parameters():
                if 'vision_model' in name and 'encoder.layers.23' not in name:
                    param.requires_grad = False  # Freeze most vision layers
                else:
                    param.requires_grad = True   # Keep text model and last vision layer trainable

        trainable_after_freeze = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        print(f"🔧 Trainable after freeze logic: {trainable_after_freeze:,}")

        # Test image features extraction
        test_input = torch.randn(1, 3, 448, 448)

        with torch.no_grad():
            features = backbone.get_image_features(test_input)

        print(f"📐 Feature shape: {features.shape}")
        print(f"📊 Feature dimensions: {features.size(-1)}")

        # Check if this matches working checkpoint expectations
        expected_params = 1_308_711_034  # From working checkpoint minus classifier
        classifier_params = 1_845_256  # From working checkpoint
        backbone_expected = expected_params - classifier_params

        print(f"\n🎯 Comparison with working checkpoint:")
        print(f"   Expected backbone params: {backbone_expected:,}")
        print(f"   Actual backbone params: {total_params:,}")
        print(f"   Match: {'✅ YES' if abs(total_params - backbone_expected) < 1000 else '❌ NO'}")

        if abs(total_params - backbone_expected) < 1000:
            print("\n🎉 SUCCESS: Model loading matches working configuration!")
            print("🚀 This should restore 86%+ accuracy")
            return True
        else:
            print("\n⚠️ WARNING: Parameter count doesn't match working checkpoint")
            return False

    except Exception as e:
        print(f"❌ Error loading MedSigLIP: {e}")
        return False

if __name__ == "__main__":
    success = test_medsiglip_loading()
    exit(0 if success else 1)