# FlavorGraph AI Training on RunPod

## 🚀 Quick Setup (5 minutes)

### 1. Create RunPod Account
- Go to [runpod.io](https://runpod.io)
- Sign up and add $10-20 credit

### 2. Launch GPU Pod
```
Template: PyTorch 2.0
GPU: RTX 4090 (24GB) - ~$0.50/hr
Storage: 50GB Container + 20GB Volume
```

### 3. Setup FlavorGraph
```bash
# In RunPod terminal:
cd /workspace
git clone <your-repo> FlavorGraph
cd FlavorGraph

# Install requirements
pip install -r paperspace/requirements.txt

# Generate training data
python3 paperspace/prepare_training_data.py

# Start training (RTX 4090 can handle full Llama 7B!)
python3 paperspace/train_flavor_model.py --config paperspace/configs/llama7b_lora.yaml
```

### 4. Monitor Training
```bash
# Watch logs
tail -f output/*/training_logs.txt

# Check GPU usage
nvidia-smi
```

### 5. Download Model
```bash
# Zip trained model
tar -czf flavorgraph_model.tar.gz output/

# Download via RunPod interface
```

## 💰 Cost Estimate
- **RTX 4090**: $0.50/hr × 12 hours = **$6**
- **Total with setup**: ~$8-10

## 🎯 Why RunPod?
- ✅ Instant access (no approval)
- ✅ Cheaper than Paperspace
- ✅ RTX 4090 handles Llama 7B easily
- ✅ Simple Docker-based setup
- ✅ Persistent storage options
