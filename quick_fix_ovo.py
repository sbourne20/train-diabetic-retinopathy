#!/usr/bin/env python3
"""
Quick Fix: Skip CLAHE for faster OVO training
This will get you to model training immediately
"""

import os
import sys

def create_fast_training_command():
    """Create fast training command without CLAHE for immediate results."""

    cmd = [
        "python", "ensemble_local_trainer.py",
        "--mode", "train",
        "--dataset_path", "./dataset6",
        "--output_dir", "./ovo_fast_results",

        # Fast training parameters
        "--epochs", "20",  # Reduced for quick testing
        "--batch_size", "32",  # Larger batch for better GPU usage
        "--learning_rate", "2e-4",

        # OVO models
        "--base_models", "mobilenet_v2",  # Start with single model first
        "--freeze_weights",

        # DISABLE CLAHE for speed
        # "--enable_clahe",  # Commented out

        # Training optimization
        "--early_stopping_patience", "5",
        "--experiment_name", "ovo_fast_test",
        "--seed", "42"
    ]

    return cmd

def main():
    print("🚀 QUICK FIX: Fast OVO Training (No CLAHE)")
    print("=" * 50)

    cmd = create_fast_training_command()

    print("💡 SOLUTION:")
    print("1. Kill current slow training:")
    print("   pkill -f ensemble_local_trainer.py")
    print()
    print("2. Run fast version:")
    print("   " + " ".join(cmd))
    print()
    print("3. This will:")
    print("   ✅ Skip CLAHE preprocessing (major speedup)")
    print("   ✅ Use single model first (mobilenet_v2)")
    print("   ✅ Start GPU training immediately")
    print("   ✅ Show epoch progress in ~2-3 minutes")

    # Write command to file for easy execution
    with open("run_fast_ovo.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Quick fix for OVO training\n")
        f.write("echo 'Starting fast OVO training...'\n")
        f.write(" ".join(cmd) + "\n")

    os.chmod("run_fast_ovo.sh", 0o755)
    print("\n📝 Command saved to: run_fast_ovo.sh")
    print("   Execute with: ./run_fast_ovo.sh")

if __name__ == "__main__":
    main()