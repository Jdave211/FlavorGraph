# 🚀 FlavorGraph AI Training on RunPod - Step-by-Step

## Step 1: Configure Your Pod

**When launching your RTX 4090 pod:**
```
Template: PyTorch 2.0 (or RunPod PyTorch)
GPU: RTX 4090 (24GB)
Container Disk: 50GB
Volume Storage: 20GB (optional but recommended)
Expose HTTP: 8888 (for Jupyter if needed)
```

## Step 2: Connect to Your Pod

**Once your pod is running:**
1. Click "Connect" in RunPod dashboard
2. Choose "Start Web Terminal" or "Connect via SSH"
3. You'll get a terminal in your pod

## Step 3: Setup FlavorGraph Environment

**Run these commands in order:**

```bash
# Navigate to workspace
cd /workspace

# Update system and install git (if needed)
apt update && apt install -y git

# Clone your FlavorGraph repository
# Replace <your-repo-url> with your actual repository URL
git clone <your-repo-url> FlavorGraph
cd FlavorGraph

# OR upload your local FlavorGraph folder via RunPod interface
# Then: cd FlavorGraph

# Install Python requirements
pip install -r paperspace/requirements.txt

# Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

## Step 4: Generate Training Data

```bash
# Generate the training dataset
python3 paperspace/prepare_training_data.py

# Verify training data was created
ls -la paperspace/training_data/
```

## Step 5: Start Training

**For RTX 4090 (24GB), you can use the full Llama 7B config:**

```bash
# Start training with Llama 7B LoRA
python3 paperspace/train_flavor_model.py --config paperspace/configs/llama7b_lora.yaml

# OR if you want to be extra safe with memory, use the A4000 config:
# python3 paperspace/train_flavor_model.py --config paperspace/configs/llama7b_a4000.yaml
```

## Step 6: Monitor Training

**Open a new terminal tab and monitor progress:**

```bash
# Watch GPU usage
watch -n 5 nvidia-smi

# Monitor training logs (in another terminal)
tail -f output/llama7b_flavorgraph/training_logs.txt

# Check training progress
ls -la output/llama7b_flavorgraph/
```

## Step 7: Training Completion

**When training finishes (8-10 hours):**

```bash
# Check final model
ls -la output/llama7b_flavorgraph/

# Test the model
python3 paperspace/evaluate_model.py \
  --model_path output/llama7b_flavorgraph \
  --interactive

# Compress for download
tar -czf flavorgraph_ai_model.tar.gz output/llama7b_flavorgraph/
```

## Step 8: Download Your Model

**Via RunPod interface:**
1. Go to "Files" tab in RunPod
2. Navigate to `/workspace/FlavorGraph/`
3. Download `flavorgraph_ai_model.tar.gz`

**OR via command line:**
```bash
# If you have RunPod CLI installed locally
runpod download <pod-id> flavorgraph_ai_model.tar.gz
```

## 🔧 Troubleshooting

**If you get CUDA out of memory:**
```bash
# Use the memory-optimized config instead
python3 paperspace/train_flavor_model.py --config paperspace/configs/llama7b_a4000.yaml
```

**If training is too slow:**
```bash
# Check GPU utilization
nvidia-smi
# Should show ~90%+ GPU usage
```

**If you need to resume training:**
```bash
# Training will auto-resume from last checkpoint
python3 paperspace/train_flavor_model.py --config paperspace/configs/llama7b_lora.yaml --resume_from_checkpoint output/llama7b_flavorgraph/checkpoint-XXXX
```

## ⏱️ Expected Timeline

- **Setup**: 10-15 minutes
- **Training**: 8-10 hours
- **Evaluation**: 30 minutes
- **Download**: 10 minutes
- **Total**: ~9-11 hours

## 💰 Cost Estimate

- **RTX 4090**: $0.59/hr × 10 hours = **$5.90**
- **Setup time**: $0.59 × 0.5 hours = **$0.30**
- **Total**: ~**$6.20**

## 🎉 Success!

Once complete, you'll have a trained FlavorGraph AI that can:
- Answer questions about ingredients
- Suggest substitutions
- Analyze flavor profiles
- Understand food pairings

Ready to start? Just follow the steps above! 🚀
