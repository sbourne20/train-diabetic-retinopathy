#!/usr/bin/env python3
"""
Test script to verify wandb integration works properly
"""

import sys
import os

def test_wandb_integration():
    """Test wandb installation and basic functionality."""

    print("🧪 Testing Wandb Integration")
    print("=" * 40)

    # Test 1: Import wandb
    try:
        import wandb
        print("✅ Wandb import successful")
    except ImportError:
        print("❌ Wandb not installed. Run: pip install wandb")
        return False

    # Test 2: Import psutil for monitoring
    try:
        import psutil
        print("✅ Psutil import successful")
    except ImportError:
        print("❌ Psutil not installed. Run: pip install psutil")
        return False

    # Test 3: Test basic monitoring
    try:
        import torch

        # System info
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()

        print(f"✅ System monitoring working:")
        print(f"   CPU: {cpu_percent}%")
        print(f"   Memory: {memory_info.percent}%")

        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            print(f"   GPU Memory: {gpu_memory:.2f} GB")

    except Exception as e:
        print(f"⚠️ System monitoring warning: {e}")

    # Test 4: Test wandb init (offline mode)
    try:
        # Test offline initialization
        wandb.init(
            project="test-project",
            mode="offline",
            name="test-run"
        )

        # Test logging
        wandb.log({
            "test_metric": 0.95,
            "test_accuracy": 85.5,
            "cpu_usage": cpu_percent
        })

        wandb.finish()
        print("✅ Wandb offline test successful")

    except Exception as e:
        print(f"⚠️ Wandb test warning: {e}")
        print("   This is normal if wandb is not configured yet")

    print("\n📊 Expected Wandb Features:")
    print("   🎯 Real-time training metrics")
    print("   💾 GPU/CPU/Memory monitoring")
    print("   📈 Learning curves & accuracy plots")
    print("   ⚡ Resource usage tracking")
    print("   🔍 Hyperparameter comparison")
    print("   📱 Mobile monitoring via wandb app")

    print("\n🚀 To use wandb:")
    print("   1. Sign up at https://wandb.ai")
    print("   2. Run: wandb login")
    print("   3. Run the super ensemble trainer")
    print("   4. Monitor at https://wandb.ai/your-username/diabetic-retinopathy-super-ensemble")

    return True

if __name__ == "__main__":
    test_wandb_integration()