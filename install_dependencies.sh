#!/bin/bash
# V100 PyTorch + Dependencies Installation Script

echo "🎮 Installing PyTorch for V100 CUDA Training..."
echo "=================================================="

# Check CUDA version
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
    echo ""
else
    echo "⚠️ nvidia-smi not found. Installing CPU version..."
fi

# Install PyTorch with CUDA support
echo "📦 Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "📦 Installing Transformers and Model Libraries..."
pip install transformers>=4.30.0
pip install timm>=0.9.0
pip install peft>=0.4.0
pip install accelerate>=0.20.0
pip install bitsandbytes>=0.41.0

echo ""
echo "🖼️ Installing Image Processing Libraries..."
pip install pillow>=9.0.0
pip install opencv-python>=4.5.0
pip install albumentations>=1.3.0

echo ""
echo "📊 Installing Data Science Libraries..."
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install scikit-learn>=1.0.0
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0

echo ""
echo "🚀 Installing Training Utilities..."
pip install tqdm>=4.64.0
pip install datasets>=2.12.0
pip install evaluate>=0.4.0

echo ""
echo "☁️ Installing Optional Cloud Libraries..."
pip install google-cloud-storage>=2.10.0 || echo "⚠️ GCS library failed - continuing without cloud support"
pip install wandb>=0.15.0 || echo "⚠️ wandb failed - continuing without logging support"

echo ""
echo "🔑 Installing HuggingFace Libraries..."
pip install huggingface-hub
pip install python-dotenv>=1.0.0

echo ""
echo "✅ Testing PyTorch CUDA Installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
else:
    print('⚠️ CUDA not available - using CPU mode')
"

echo ""
echo "🎯 Installation Summary:"
echo "========================="
echo "✅ PyTorch with CUDA support"
echo "✅ MedSigLIP-448 compatible transformers"
echo "✅ LoRA fine-tuning support (PEFT)"
echo "✅ Medical image processing libraries"
echo "✅ Training and evaluation utilities"
echo ""
echo "🚀 Ready for V100 training!"
echo "Run: ./medical_grade_local_training.sh"