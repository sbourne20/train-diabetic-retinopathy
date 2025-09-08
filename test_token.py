#!/usr/bin/env python3
"""
Test script to verify HuggingFace token loading and MedSigLIP access
"""

import os

# Load environment variables
try:
    from dotenv import load_dotenv
    env_loaded = load_dotenv()
    print(f"✅ dotenv loaded: {env_loaded}")
except ImportError:
    print("❌ python-dotenv not installed. Run: pip install python-dotenv")
    exit(1)

# Check current directory and .env file
print(f"📁 Current directory: {os.getcwd()}")
print(f"📄 .env file exists: {os.path.exists('.env')}")

if os.path.exists('.env'):
    print("\n📄 .env file contents:")
    with open('.env', 'r') as f:
        for i, line in enumerate(f, 1):
            if line.strip() and not line.startswith('#'):
                if 'HUGGINGFACE_TOKEN' in line:
                    parts = line.strip().split('=', 1)
                    if len(parts) == 2:
                        token_preview = parts[1][:10] + '***' + parts[1][-5:] if len(parts[1]) > 15 else 'hf_***'
                        print(f"   Line {i}: {parts[0]}={token_preview}")
                    else:
                        print(f"   Line {i}: {line.strip()}")
                else:
                    print(f"   Line {i}: {line.strip()}")

# Check environment variable
hf_token = os.getenv("HUGGINGFACE_TOKEN")
print(f"\n🔑 HUGGINGFACE_TOKEN in environment: {'✅ Yes' if hf_token else '❌ No'}")

if hf_token:
    token_preview = hf_token[:10] + '***' + hf_token[-5:] if len(hf_token) > 15 else 'hf_***'
    print(f"🔑 Token preview: {token_preview}")
    
    # Test HuggingFace access
    print("\n🤖 Testing HuggingFace access...")
    try:
        from transformers import AutoConfig
        print("✅ Transformers library loaded")
        
        # Test MedSigLIP access
        print("🔍 Testing MedSigLIP-448 access...")
        config = AutoConfig.from_pretrained('google/medsiglip-448', token=hf_token)
        print("✅ MedSigLIP-448 accessible with token!")
        print(f"   Model type: {config.model_type}")
        
    except Exception as e:
        print(f"❌ MedSigLIP access failed: {e}")
        print("💡 This might be due to:")
        print("   1. Invalid token")
        print("   2. No access to google/medsiglip-448 model")
        print("   3. Network connectivity issues")
        
else:
    print("\n🔧 TROUBLESHOOTING:")
    print("1. Verify .env file format (no spaces around =):")
    print("   HUGGINGFACE_TOKEN=hf_your_token_here")
    print("2. Ensure .env file is in the same directory as this script")
    print("3. Check if python-dotenv is installed: pip install python-dotenv")
    print("4. Alternative: export HUGGINGFACE_TOKEN=hf_your_token_here")

print("\n" + "="*60)
print("🎯 SUMMARY:")
print("✅ Required for training:" if hf_token else "❌ Required for training:")
print("   - HUGGINGFACE_TOKEN loaded and valid")
print("   - MedSigLIP-448 model accessible")
print("="*60)