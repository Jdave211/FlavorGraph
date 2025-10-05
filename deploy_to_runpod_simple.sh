#!/bin/bash
# Simple deployment to RunPod using tar and scp

set -e

RUNPOD_HOST="turqcdwlx8to2r-644114e5@ssh.runpod.io"
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "======================================================================"
echo "🚀 FlavorGraph RunPod Deployment"
echo "======================================================================"
echo ""

# Step 1: Create tar archive (excluding large files)
echo "📦 Creating project archive..."
tar -czf /tmp/flavorgraph_deploy.tar.gz \
  --exclude='.git' \
  --exclude='output' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.vscode' \
  --exclude='.kiro' \
  --exclude='flavorgraph_ai_model.tar.gz' \
  -C /Users/davejaga/Desktop/Startups FlavorGraph

echo "✅ Archive created: $(du -h /tmp/flavorgraph_deploy.tar.gz | cut -f1)"
echo ""

# Step 2: Upload to RunPod
echo "📤 Uploading to RunPod..."
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no \
  /tmp/flavorgraph_deploy.tar.gz \
  ${RUNPOD_HOST}:/workspace/

echo "✅ Upload complete"
echo ""

# Step 3: Extract and run training
echo "🎯 Starting training on RunPod..."
echo ""

ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no ${RUNPOD_HOST} << 'ENDSSH'
cd /workspace

# Extract
echo "📦 Extracting project..."
tar -xzf flavorgraph_deploy.tar.gz
cd FlavorGraph

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
nohup python training/train_llama.py --config training/config_llama_training.yaml > training.log 2>&1 &

echo "✅ Training started in background!"
echo ""
echo "Monitor with: tail -f /workspace/FlavorGraph/training.log"
echo "Or reconnect and check: cd /workspace/FlavorGraph && tail -f training.log"

ENDSSH

echo ""
echo "======================================================================"
echo "✅ Deployment Complete!"
echo "======================================================================"
echo ""
echo "Training is now running on RunPod in the background."
echo ""
echo "To monitor progress:"
echo "  ssh -i $SSH_KEY ${RUNPOD_HOST}"
echo "  tail -f /workspace/FlavorGraph/training.log"
echo ""
echo "Training will take ~2-3 hours. The pod will keep running."
echo ""
echo "To download model when done:"
echo "  scp -i $SSH_KEY ${RUNPOD_HOST}:/workspace/FlavorGraph/training/output/flavorgraph_llama_v1/adapter_model.bin ./"

# Cleanup
rm /tmp/flavorgraph_deploy.tar.gz
