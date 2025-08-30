#!/usr/bin/env python3
"""
Quick pipeline test script to validate all components work before expensive training.

This script runs a 2-epoch debug test to catch any bugs early and save costs.
"""

import os
import sys
import subprocess
import argparse

def run_debug_test(dataset_name: str = "dataset3_augmented_resized", 
                   dataset_type: int = 1,
                   bucket_name: str = "dr-data-2",
                   project_id: str = "curalis-20250522",
                   region: str = "us-central1"):
    """Run a 2-epoch debug test to validate the entire pipeline."""
    
    print("🐛 PIPELINE DEBUG TEST")
    print("=" * 50)
    print("This test validates the entire training pipeline in ~15-20 minutes")
    print("Cost: ~$5-10 (vs $150+ for full training)")
    print("Purpose: Catch bugs early and save money!")
    print()
    
    # Test locally first if possible
    print("🔍 Step 1: Local validation test")
    try:
        # Run a minimal local test
        cmd = [
            "python", "main.py",
            "--mode", "train",
            "--debug_mode",
            "--max_epochs", "1",
            "--batch_size", "2",  # Very small batch for memory
            "--no_wandb"
        ]
        
        if dataset_type == 1:
            cmd.extend(["--dataset_path", f"gs://{bucket_name}/{dataset_name}"])
            cmd.extend(["--num_classes", "5"])
        
        print(f"Command: {' '.join(cmd)}")
        print("⚠️  Note: This may fail due to missing dataset locally, which is expected.")
        print("   The main purpose is to validate imports and basic setup.")
        
        # Don't actually run locally as dataset is on GCS
        print("   Skipping local test (dataset is on GCS)")
        
    except Exception as e:
        print(f"   Local test failed (expected): {e}")
    
    print()
    print("🚀 Step 2: Vertex AI debug test (2 epochs)")
    
    # Run on Vertex AI with debug mode
    cmd = [
        "python", "vertex_ai_trainer.py",
        "--action", "train",
        "--dataset", dataset_name,
        "--dataset-type", str(dataset_type),
        "--bucket_name", bucket_name,
        "--project_id", project_id,
        "--region", region,
        "--debug_mode",
        "--max_epochs", "2",
        "--eval_frequency", "1",
        "--checkpoint_frequency", "1"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    print("⏳ Starting Vertex AI debug test...")
    print("   - This will test: model loading, data loading, training, evaluation, checkpointing")
    print("   - Expected time: 15-20 minutes")
    print("   - Expected cost: $5-10")
    print("   - If successful, you can confidently run full training!")
    print()
    
    confirm = input("🤔 Start Vertex AI debug test? (y/N): ").lower().strip()
    if confirm != 'y':
        print("❌ Debug test cancelled")
        return False
    
    try:
        # Run the command
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        print()
        print("✅ DEBUG TEST SUCCESSFUL!")
        print("=" * 50)
        print("🎉 All pipeline components work correctly!")
        print("💰 You can now run full training with confidence")
        print("📦 Checkpoints will be saved to gs://dr-data-2/checkpoints/")
        print()
        print("🚀 To run full training:")
        full_cmd = cmd.copy()
        # Remove debug parameters and set full training parameters
        full_cmd = [arg for arg in full_cmd if arg not in ["--debug_mode", "--max_epochs", "2", "--eval_frequency", "1", "--checkpoint_frequency", "1"]]
        print(f"   {' '.join(full_cmd)}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print()
        print("❌ DEBUG TEST FAILED!")
        print("=" * 50)
        print("🐛 The pipeline has bugs that need to be fixed before full training")
        print(f"   Error code: {e.returncode}")
        print("💰 Good news: You caught this early and saved $100+!")
        print()
        print("🔧 Check the error logs above and fix the issues")
        print("   Then run this debug test again before full training")
        
        return False

def main():
    parser = argparse.ArgumentParser(description="Run pipeline debug test")
    parser.add_argument("--dataset", default="dataset3_augmented_resized", help="Dataset name")
    parser.add_argument("--dataset-type", type=int, default=1, choices=[0, 1], help="Dataset type")
    parser.add_argument("--bucket_name", default="dr-data-2", help="GCS bucket name")
    parser.add_argument("--project_id", default="curalis-20250522", help="GCP project ID")
    parser.add_argument("--region", default="us-central1", help="GCP region")
    
    args = parser.parse_args()
    
    success = run_debug_test(
        dataset_name=args.dataset,
        dataset_type=args.dataset_type,
        bucket_name=args.bucket_name,
        project_id=args.project_id,
        region=args.region
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()