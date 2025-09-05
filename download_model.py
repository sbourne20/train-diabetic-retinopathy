#!/usr/bin/env python3
"""
Download MedSigLIP-448 model to local cache
Run this once to avoid downloading during training
"""

import os
import sys
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("❌ Error: python-dotenv not found. Install with: pip install python-dotenv")
    sys.exit(1)

# Check HuggingFace token
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    print("❌ Error: HUGGINGFACE_TOKEN not found in .env file")
    print("Add: HUGGINGFACE_TOKEN=hf_your_token_here")
    sys.exit(1)

# Setup cache directory
results_dir = Path("results")
models_dir = results_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)

# Set transformers cache
os.environ["TRANSFORMERS_CACHE"] = str(models_dir)

print("🏥 MEDSIGLIP-448 MODEL DOWNLOADER")
print("="*50)
print(f"📥 Downloading MedSigLIP-448 model...")
print(f"💾 Cache location: {models_dir}")
print(f"🔗 Model: google/medsiglip-448")
print("")

try:
    from transformers import AutoModel
    
    # Download the model
    print("⏬ Starting download...")
    model = AutoModel.from_pretrained(
        "google/medsiglip-448",
        token=hf_token,
        trust_remote_code=True,
        cache_dir=str(models_dir)
    )
    
    print("")
    print("✅ SUCCESS: MedSigLIP-448 model downloaded successfully!")
    print(f"💾 Cached at: {models_dir}")
    print("")
    print("🚀 NEXT STEPS:")
    print("  • Model is now cached locally")
    print("  • Training will start instantly (no download)")
    print("  • Run: ./mlx_train_local.sh")
    print("")
    
except Exception as e:
    print(f"❌ ERROR: Failed to download model: {e}")
    print("")
    print("🔧 TROUBLESHOOTING:")
    print("  • Check your HUGGINGFACE_TOKEN in .env file")
    print("  • Ensure token has access to google/medsiglip-448")
    print("  • Check internet connection")
    sys.exit(1)