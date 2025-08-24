#!/bin/bash

# Setup script for Diabetic Retinopathy project environment

set -e

echo "🐍 Setting up Python Virtual Environment for Diabetic Retinopathy Project"
echo "======================================================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    echo "Visit: https://www.python.org/downloads/"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
echo "🔍 Found Python version: $PYTHON_VERSION"

# Check if version is 3.8 or higher
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "✅ Python version is compatible"
else
    echo "❌ Python 3.8 or higher is required. Current version: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Removing old one..."
    rm -rf venv
fi

python3 -m venv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing project dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "❌ requirements.txt not found!"
    exit 1
fi

# Install additional development tools
echo "🛠️  Installing development tools..."
pip install jupyter ipykernel black flake8

# Add kernel to Jupyter (if Jupyter is available)
if command -v jupyter &> /dev/null; then
    python -m ipykernel install --user --name=diabetic_retinopathy --display-name="Diabetic Retinopathy"
    echo "📓 Jupyter kernel 'Diabetic Retinopathy' added"
fi

# Create .env template if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📄 Creating .env template..."
    cat > .env << 'EOF'
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_REGION=us-central1
GCS_BUCKET=your-bucket-name

# Training Configuration
WANDB_PROJECT=diabetic-retinopathy
WANDB_ENTITY=your-wandb-username

# Optional: OpenAI API for enhanced medical reasoning
# OPENAI_API_KEY=your-openai-key
EOF
fi

# Verify installation
echo "🔍 Verifying installation..."
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

# Check key packages
PACKAGES=("torch" "transformers" "timm" "google-cloud-aiplatform")
for package in "${PACKAGES[@]}"; do
    if pip show "$package" &> /dev/null; then
        VERSION=$(pip show "$package" | grep Version | cut -d' ' -f2)
        echo "✅ $package: $VERSION"
    else
        echo "❌ $package: Not installed"
    fi
done

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "🚀 To get started:"
echo "1. Activate the environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Configure your project:"
echo "   cp .env.example .env  # Edit with your settings"
echo ""
echo "3. Test the installation:"
echo "   python -c \"import torch; print('PyTorch:', torch.__version__)\""
echo "   python -c \"import transformers; print('Transformers:', transformers.__version__)\""
echo ""
echo "4. Start training locally:"
echo "   python main.py --mode train --epochs 10"
echo ""
echo "5. Or setup for Vertex AI:"
echo "   ./quick_start_vertex_ai.sh YOUR_PROJECT_ID"
echo ""
echo "💡 Remember to activate the virtual environment before working:"
echo "   source venv/bin/activate"
echo ""
echo "📚 Documentation:"
echo "   - Local training: python main.py --help"
echo "   - Vertex AI: see VERTEX_AI_SETUP.md"
echo "   - Inference: python inference.py --help"