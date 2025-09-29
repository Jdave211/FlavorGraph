#!/bin/bash
# RunPod Deployment Script for FlavorGraph Training
# Optimized to prevent overfitting and repetitive outputs

set -e

echo "🚀 Starting FlavorGraph Training on RunPod"
echo "=========================================="

# Update system packages
echo "📦 Updating system packages..."
apt-get update -y
apt-get install -y git wget curl

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets peft accelerate bitsandbytes
pip install wandb scikit-learn numpy pandas pyyaml
pip install sentencepiece protobuf

# Clone repository if not already present
if [ ! -d "FlavorGraph" ]; then
    echo "📁 Cloning FlavorGraph repository..."
    git clone https://github.com/your-username/FlavorGraph.git
fi

cd FlavorGraph/paperspace

# Make scripts executable
chmod +x *.py
chmod +x *.sh

# Clean training data
echo "🧹 Cleaning training data..."
python3 clean_training_data.py --input training_data/combined_training.jsonl --output training_data/cleaned_training.jsonl

# Analyze data quality
echo "📊 Analyzing data quality..."
python3 clean_training_data.py --input training_data/cleaned_training.jsonl --analyze

# Estimate training time
echo "⏱️  Estimating training time..."
python3 estimate_training_time.py --config configs/runpod_long_training.yaml

# Start training with long duration configuration
echo "🎯 Starting long training (3-6 hours)..."
python3 train_runpod_optimized.py --config configs/runpod_long_training.yaml --clean-data

echo "✅ Training completed successfully!"
echo "📁 Check output directory for trained model and logs"
