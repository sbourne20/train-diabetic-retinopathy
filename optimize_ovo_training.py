#!/usr/bin/env python3
"""
Optimized OVO Training Script - GPU Performance Enhanced
Fixes CUDA utilization issues and speeds up training
"""

import os
import sys
import torch
import multiprocessing as mp

def optimize_cuda_settings():
    """Optimize CUDA settings for maximum GPU utilization."""

    if torch.cuda.is_available():
        # Enable optimized CUDA operations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Set optimal memory management
        torch.cuda.empty_cache()

        # Enable tensor core operations if available
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True

        print(f"✅ CUDA optimizations enabled")
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

        return True
    return False

def get_optimal_workers():
    """Get optimal number of workers for data loading."""

    # Get CPU count
    cpu_count = mp.cpu_count()

    # Optimal workers: min(8, cpu_count)
    optimal_workers = min(8, cpu_count)

    print(f"🔧 CPU cores: {cpu_count}")
    print(f"🔧 Optimal workers: {optimal_workers}")

    return optimal_workers

def check_dataset_ready():
    """Check if dataset6 is ready and properly structured."""

    import pathlib

    dataset_path = pathlib.Path("./dataset6")

    if not dataset_path.exists():
        print("❌ ERROR: dataset6 not found")
        return False

    # Check structure
    required_dirs = ["train", "val", "test"]
    for split in required_dirs:
        split_path = dataset_path / split
        if not split_path.exists():
            print(f"❌ ERROR: {split} directory missing in dataset6")
            return False

        # Check classes
        classes = list(range(5))
        for cls in classes:
            class_path = split_path / str(cls)
            if not class_path.exists():
                print(f"❌ ERROR: Class {cls} missing in {split}")
                return False

    print("✅ Dataset6 structure verified")
    return True

def create_optimized_training_command():
    """Create optimized training command with better parameters."""

    optimal_workers = get_optimal_workers()

    # Base command
    cmd = [
        "python", "ensemble_local_trainer.py",
        "--mode", "train",
        "--dataset_path", "./dataset6",
        "--output_dir", "./ovo_ensemble_results",

        # Optimized training parameters
        "--epochs", "40",  # Reduced for faster iteration
        "--batch_size", "32",  # Increased for better GPU utilization
        "--learning_rate", "2e-4",  # Slightly higher for faster convergence
        "--weight_decay", "1e-4",

        # OVO specific
        "--base_models", "mobilenet_v2", "inception_v3", "densenet121",
        "--freeze_weights",
        "--ovo_dropout", "0.3",  # Reduced dropout for better training

        # Enhanced preprocessing
        "--enable_clahe",
        "--clahe_clip_limit", "2.5",  # Slightly reduced for speed

        # Training strategy
        "--early_stopping_patience", "8",  # Increased patience
        "--binary_epochs", "25",  # Reduced for faster testing

        # Optimization
        "--enable_class_weights",
        "--target_accuracy", "0.9696",
        "--experiment_name", "ovo_ensemble_optimized",
        "--seed", "42"
    ]

    return cmd

def main():
    """Main optimization function."""

    print("🚀 OVO TRAINING OPTIMIZATION SCRIPT")
    print("=" * 50)

    # Check CUDA
    if not optimize_cuda_settings():
        print("⚠️ WARNING: CUDA not available, training will be slow")

    # Check dataset
    if not check_dataset_ready():
        print("❌ Dataset preparation failed")
        return False

    # Get optimal command
    cmd = create_optimized_training_command()

    print("\n🔧 OPTIMIZED TRAINING COMMAND:")
    print(" ".join(cmd))
    print()

    # Suggest modifications to existing code
    print("📝 SUGGESTED OPTIMIZATIONS:")
    print("1. Increase batch_size to 32 (better GPU utilization)")
    print("2. Reduce epochs to 25-40 for faster iteration")
    print("3. Use 8 data loader workers")
    print("4. Enable CUDA optimizations")
    print("5. Monitor GPU utilization during training")

    return True

if __name__ == "__main__":
    main()