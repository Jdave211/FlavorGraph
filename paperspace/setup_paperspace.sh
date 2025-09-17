#!/bin/bash
# FlavorGraph AI Training Setup Script for Paperspace
# Run this script to set up the environment and start training

set -e

echo "🚀 FlavorGraph AI Training Setup for Paperspace"
echo "=============================================="

# Check if we're on Paperspace (optional)
if [ -d "/notebooks" ]; then
    echo "✅ Running on Paperspace environment"
    PAPERSPACE=true
else
    echo "ℹ️  Running on local/other environment"
    PAPERSPACE=false
fi

# Update system packages
echo "📦 Updating system packages..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y git wget curl
fi

# Check Python version
echo "🐍 Checking Python version..."
python3 --version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Install pip if not available
if ! command -v pip3 &> /dev/null; then
    echo "📦 Installing pip..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py
    rm get-pip.py
fi

# Upgrade pip
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip

# Install requirements
echo "📦 Installing Python requirements..."
pip3 install -r requirements.txt

# Verify GPU availability
echo "🔍 Checking GPU availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

# Setup Weights & Biases (optional)
echo "📊 Setting up Weights & Biases..."
echo "Please run 'wandb login' manually if you want to use W&B tracking"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p output/
mkdir -p logs/

# Check if FlavorGraph data exists
if [ ! -f "input/cleaned/nodes_cleaned_basic.csv" ]; then
    echo "⚠️  FlavorGraph data not found. Please ensure you have:"
    echo "   - input/cleaned/nodes_cleaned_basic.csv"
    echo "   - input/edges_191120.csv"
    echo "   - input/compound_flavors/compound_flavor_mappings.csv"
    echo "   - output/*.pickle (embeddings)"
    exit 1
fi

# Generate training data
echo "🧠 Preparing training data..."
python3 prepare_training_data.py

# Check if training data was created successfully
if [ ! -f "training_data/combined_training.jsonl" ]; then
    echo "❌ Training data generation failed"
    exit 1
fi

echo "✅ Setup complete!"
echo ""
echo "🎯 To start training, run:"
echo "   python3 train_flavor_model.py --config configs/llama7b_lora.yaml"
echo ""
echo "📊 Available configurations:"
echo "   - configs/llama7b_lora.yaml     (Llama 7B with LoRA - requires A100 40GB+)"
echo "   - configs/mistral7b_qlora.yaml  (Mistral 7B with QLoRA - works on RTX4000+)"
echo ""
echo "📈 Monitor training with:"
echo "   - Weights & Biases dashboard (if configured)"
echo "   - TensorBoard: tensorboard --logdir output/"
echo ""
echo "🎉 Happy training!"
