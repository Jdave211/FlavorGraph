# RunPod Training Instructions

## 🚀 Two Ways to Deploy

### Option 1: Automated Script (Recommended)

From your local machine, run:
```bash
bash deploy_to_runpod.sh
```

This will:
1. Upload your project to RunPod
2. Install dependencies
3. Generate training data
4. Start training
5. Compress the model for download

---

### Option 2: Manual Steps

#### Step 1: Connect to RunPod
```bash
ssh turqcdwlx8to2r-644114e5@ssh.runpod.io -i ~/.ssh/id_ed25519
```

#### Step 2: Upload Project
From your **local terminal** (separate window):
```bash
rsync -avz --progress \
  --exclude='.git' --exclude='output' --exclude='__pycache__' \
  -e "ssh -i ~/.ssh/id_ed25519" \
  /Users/davejaga/Desktop/Startups/FlavorGraph/ \
  turqcdwlx8to2r-644114e5@ssh.runpod.io:/workspace/FlavorGraph/
```

#### Step 3: Setup Environment (on RunPod)
```bash
cd /workspace/FlavorGraph
pip install -r training/requirements.txt
python training/check_setup.py
```

#### Step 4: Generate Training Data (on RunPod)
```bash
python training/generate_llama_training_data.py
```

This creates `training/data/flavorgraph_training_data.jsonl` with ~3,300 examples.

#### Step 5: Start Training (on RunPod)
```bash
# Disable W&B if you don't have account
export WANDB_MODE=disabled

# Start training (2-3 hours on A100)
python training/train_llama.py --config training/config_llama_training.yaml
```

#### Step 6: Monitor Progress
```bash
# In another terminal, watch the training
ssh turqcdwlx8to2r-644114e5@ssh.runpod.io -i ~/.ssh/id_ed25519
cd /workspace/FlavorGraph
watch -n 30 tail -20 training/output/flavorgraph_llama_v1/trainer_state.json
```

Or use `tmux`/`screen` to keep training running:
```bash
# Start tmux session
tmux new -s training

# Run training in tmux
python training/train_llama.py --config training/config_llama_training.yaml

# Detach: Ctrl+B then D
# Reattach later: tmux attach -t training
```

#### Step 7: Download Trained Model
After training completes, from **local terminal**:
```bash
# Compress on RunPod first
ssh turqcdwlx8to2r-644114e5@ssh.runpod.io -i ~/.ssh/id_ed25519 \
  "cd /workspace/FlavorGraph/training/output && tar -czf model.tar.gz flavorgraph_llama_v1/"

# Download to local machine
scp -i ~/.ssh/id_ed25519 \
  turqcdwlx8to2r-644114e5@ssh.runpod.io:/workspace/FlavorGraph/training/output/model.tar.gz \
  ./
```

---

## 🎯 Quick Commands Reference

### Upload project:
```bash
rsync -avz --progress --exclude='.git' --exclude='output' \
  -e "ssh -i ~/.ssh/id_ed25519" \
  /Users/davejaga/Desktop/Startups/FlavorGraph/ \
  turqcdwlx8to2r-644114e5@ssh.runpod.io:/workspace/FlavorGraph/
```

### Connect to RunPod:
```bash
ssh turqcdwlx8to2r-644114e5@ssh.runpod.io -i ~/.ssh/id_ed25519
```

### Run full pipeline on RunPod:
```bash
cd /workspace/FlavorGraph
bash training/run_full_pipeline.sh
```

### Download model:
```bash
scp -i ~/.ssh/id_ed25519 \
  turqcdwlx8to2r-644114e5@ssh.runpod.io:/workspace/FlavorGraph/training/output/model.tar.gz \
  ./
```

---

## ⏱️ Timeline (A100)

- **Upload**: 2-3 minutes (depends on connection)
- **Setup**: 3-5 minutes (pip install)
- **Data generation**: 5-10 minutes
- **Model download**: 5-10 minutes (one-time, 13GB)
- **Training**: 2-3 hours
- **Download model**: 1-2 minutes (200MB adapter)

**Total**: ~2.5-3 hours
**Cost**: ~$3-4

---

## 🔍 Troubleshooting

### Connection timeout
```bash
# Add verbose flag to see what's happening
ssh -v -i ~/.ssh/id_ed25519 turqcdwlx8to2r-644114e5@ssh.runpod.io
```

### Upload too slow
Use RunPod's web terminal and upload via their UI:
1. Go to RunPod dashboard
2. Click "Connect" → "Web Terminal"
3. Use file upload feature

### Training crashes
Check logs:
```bash
cat training/output/flavorgraph_llama_v1/trainer_state.json
```

Resume from checkpoint:
```bash
python training/train_llama.py \
  --config training/config_llama_training.yaml \
  --resume_from_checkpoint training/output/flavorgraph_llama_v1/checkpoint-XXX
```

### Out of memory
Edit config:
```bash
nano training/config_llama_training.yaml
# Change: batch_size: 4 → 2
```

---

## 💡 Pro Tips

1. **Use tmux** to keep training running if you disconnect:
   ```bash
   tmux new -s training
   # Run training
   # Detach: Ctrl+B then D
   ```

2. **Monitor GPU usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Save money**: Generate training data locally first, then upload only the .jsonl file

4. **Track progress**: Enable W&B for real-time monitoring:
   ```bash
   pip install wandb
   wandb login
   # Edit config: use_wandb: true
   ```

---

## 🎉 What You'll Get

After training:
- `adapter_model.bin` (~200MB) - Your trained weights
- `adapter_config.json` - LoRA configuration
- `tokenizer files` - For inference
- `training_config.yaml` - Training settings

You can then use these with the base LLaMA model for inference!
