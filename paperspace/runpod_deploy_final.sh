#!/bin/bash
# Final RunPod Deployment Script for FlavorGraph Training
# Uses expanded high-quality dataset (33K+ examples)

set -e

echo "🚀 Starting FlavorGraph Training on RunPod (Final Version)"
echo "========================================================"

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

# Generate expanded training data (if not already present)
if [ ! -f "training_data/expanded_training.jsonl" ]; then
    echo "📝 Generating expanded training dataset..."
    python3 expand_training_data.py
else
    echo "✅ Expanded training dataset already exists"
fi

# Test dataset quality
echo "🧪 Testing dataset quality..."
python3 test_expanded_training.py

# Estimate training time
echo "⏱️  Estimating training time..."
python3 estimate_training_time.py --config configs/runpod_expanded_dataset.yaml

# Start training with expanded dataset
echo "🎯 Starting training with expanded dataset (2-3 hours)..."
echo "📊 Dataset: 33,355 high-quality examples"
echo "⏰ Expected time: 2-3 hours"
echo ""

python3 train_runpod_optimized.py --config configs/runpod_expanded_dataset.yaml

echo "✅ Training completed successfully!"
echo "📁 Check output directory for trained model and logs"
echo "🎉 Model should now generate meaningful responses instead of 'input!'"
