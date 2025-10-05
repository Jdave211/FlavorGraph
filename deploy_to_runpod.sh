#!/bin/bash
# Deploy and run FlavorGraph training on RunPod

set -e

RUNPOD_HOST="turqcdwlx8to2r-644114e5@ssh.runpod.io"
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "======================================================================"
echo "🚀 FlavorGraph RunPod Deployment"
echo "======================================================================"
echo ""

# Step 1: Upload project to RunPod
echo "📤 Uploading project to RunPod..."
rsync -avz --progress \
  --exclude='.git' \
  --exclude='output' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.vscode' \
  --exclude='.kiro' \
  --exclude='flavorgraph_ai_model.tar.gz' \
  -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
  ./ ${RUNPOD_HOST}:/workspace/FlavorGraph/

echo "✅ Upload complete"
echo ""

# Step 2: Run training on RunPod
echo "🎯 Starting training on RunPod..."
echo ""

ssh -i $SSH_KEY ${RUNPOD_HOST} << 'ENDSSH'
cd /workspace/FlavorGraph

echo "======================================================================"
echo "Setting up environment..."
echo "======================================================================"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r training/requirements.txt

echo "✅ Dependencies installed"
echo ""

# Check setup
echo "🔍 Validating setup..."
python training/check_setup.py
echo ""

# Generate training data
echo "📊 Generating training data..."
python training/generate_llama_training_data.py
echo ""

# Start training
echo "======================================================================"
echo "🚀 Starting LLaMA Training"
echo "======================================================================"
echo ""
echo "This will take approximately 2-3 hours on A100..."
echo ""

# Disable wandb if no token
export WANDB_MODE=disabled

# Run training
python training/train_llama.py --config training/config_llama_training.yaml

echo ""
echo "======================================================================"
echo "✅ Training Complete!"
echo "======================================================================"
echo ""

# Compress model for download
echo "📦 Compressing model..."
cd training/output
tar -czf flavorgraph_llama_trained.tar.gz flavorgraph_llama_v1/

echo "✅ Model compressed: training/output/flavorgraph_llama_trained.tar.gz"
echo ""
echo "Download with:"
echo "scp -i ~/.ssh/id_ed25519 ${RUNPOD_HOST}:/workspace/FlavorGraph/training/output/flavorgraph_llama_trained.tar.gz ./"

ENDSSH

echo ""
echo "======================================================================"
echo "🎉 Deployment Complete!"
echo "======================================================================"
echo ""
echo "Training is running on RunPod. To monitor:"
echo "ssh -i $SSH_KEY ${RUNPOD_HOST}"
echo "cd /workspace/FlavorGraph"
echo "tail -f training/output/flavorgraph_llama_v1/trainer_state.json"
