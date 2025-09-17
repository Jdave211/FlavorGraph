# FlavorGraph AI Training on Vast.ai

## 💰 Cheapest Option (2-4x cheaper than Paperspace!)

### 1. Create Vast.ai Account
- Go to [vast.ai](https://vast.ai)
- Sign up and add $5-10 credit

### 2. Find GPU Instance
```
Search filters:
- GPU: RTX 3090 or RTX 4090
- RAM: 32GB+
- Storage: 50GB+
- Price: <$0.50/hr
- Reliability: >95%
```

### 3. Launch Instance
```bash
# Vast.ai provides SSH command like:
ssh -p 12345 root@ssh.vast.ai

# Once connected:
apt update && apt install -y git python3-pip
cd /workspace
```

### 4. Setup FlavorGraph
```bash
# Clone your repo
git clone <your-repo> FlavorGraph
cd FlavorGraph

# Install requirements
pip3 install -r paperspace/requirements.txt

# Generate training data
python3 paperspace/prepare_training_data.py

# Choose config based on GPU:
# RTX 3090/4090 (24GB): Use full Llama config
python3 paperspace/train_flavor_model.py --config paperspace/configs/llama7b_lora.yaml

# RTX 3080 (10GB): Use Mistral config
python3 paperspace/train_flavor_model.py --config paperspace/configs/mistral7b_qlora.yaml
```

### 5. Monitor & Download
```bash
# Monitor training
watch -n 30 nvidia-smi

# When done, compress model
tar -czf model.tar.gz output/

# Download via SCP
scp -P 12345 root@ssh.vast.ai:/workspace/FlavorGraph/model.tar.gz ./
```

## 💰 Cost Estimate
- **RTX 3090**: $0.30/hr × 15 hours = **$4.50**
- **RTX 4090**: $0.45/hr × 12 hours = **$5.40**

## ⚠️ Considerations
- ✅ Cheapest option available
- ✅ Instant access
- ⚠️ Reliability varies by host
- ⚠️ More technical setup required
- ⚠️ Instance can be interrupted
