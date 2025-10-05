# Manual RunPod Deployment Steps

Since automated deployment requires PTY support, follow these manual steps from your **local terminal**:

## Step 1: Create Project Archive

```bash
cd /Users/davejaga/Desktop/Startups/FlavorGraph

tar -czf /tmp/flavorgraph.tar.gz \
  --exclude='.git' \
  --exclude='output' \
  --exclude='__pycache__' \
  --exclude='.vscode' \
  --exclude='.kiro' \
  --exclude='flavorgraph_ai_model.tar.gz' \
  .
```

## Step 2: Upload to RunPod

```bash
scp -i ~/.ssh/id_ed25519 \
  /tmp/flavorgraph.tar.gz \
  turqcdwlx8to2r-644114e5@ssh.runpod.io:/workspace/
```

## Step 3: Connect to RunPod

```bash
ssh -i ~/.ssh/id_ed25519 turqcdwlx8to2r-644114e5@ssh.runpod.io
```

## Step 4: Extract and Setup (on RunPod)

Once connected to RunPod, run these commands:

```bash
cd /workspace
tar -xzf flavorgraph.tar.gz
cd FlavorGraph

# Install dependencies
pip install -r training/requirements.txt

# Verify setup
python training/check_setup.py
```

## Step 5: Generate Training Data (on RunPod)

```bash
python training/generate_llama_training_data.py
```

This will create `training/data/flavorgraph_training_data.jsonl` with ~3,300 examples.

## Step 6: Start Training (on RunPod)

```bash
# Optional: use tmux to keep training running if you disconnect
tmux new -s training

# Disable W&B (unless you have an account)
export WANDB_MODE=disabled

# Start training
python training/train_llama.py --config training/config_llama_training.yaml
```

**If using tmux:**
- Detach: Press `Ctrl+B` then `D`
- Reattach later: `tmux attach -t training`

## Step 7: Monitor Progress

### Option A: Stay connected and watch
```bash
# Just let the training run, you'll see progress bars
```

### Option B: Check logs
```bash
# In another terminal/tmux window
tail -f training/output/flavorgraph_llama_v1/trainer_state.json
```

### Option C: Check GPU usage
```bash
watch -n 1 nvidia-smi
```

## Step 8: Download Trained Model

After training completes (~2-3 hours), from your **local terminal**:

```bash
# Download just the adapter weights (~200MB)
scp -i ~/.ssh/id_ed25519 \
  turqcdwlx8to2r-644114e5@ssh.runpod.io:/workspace/FlavorGraph/training/output/flavorgraph_llama_v1/adapter_model.bin \
  ./

# Or download everything
scp -i ~/.ssh/id_ed25519 -r \
  turqcdwlx8to2r-644114e5@ssh.runpod.io:/workspace/FlavorGraph/training/output/flavorgraph_llama_v1/ \
  ./
```

---

## Quick Command Summary

```bash
# 1. Create archive (local)
cd /Users/davejaga/Desktop/Startups/FlavorGraph
tar -czf /tmp/flavorgraph.tar.gz --exclude='.git' --exclude='output' --exclude='__pycache__' .

# 2. Upload (local)
scp -i ~/.ssh/id_ed25519 /tmp/flavorgraph.tar.gz turqcdwlx8to2r-644114e5@ssh.runpod.io:/workspace/

# 3. Connect (local)
ssh -i ~/.ssh/id_ed25519 turqcdwlx8to2r-644114e5@ssh.runpod.io

# 4. Setup (RunPod)
cd /workspace && tar -xzf flavorgraph.tar.gz && cd FlavorGraph
pip install -r training/requirements.txt
python training/check_setup.py

# 5. Generate data (RunPod)
python training/generate_llama_training_data.py

# 6. Train (RunPod)
export WANDB_MODE=disabled
python training/train_llama.py --config training/config_llama_training.yaml

# 7. Download (local, after training)
scp -i ~/.ssh/id_ed25519 -r turqcdwlx8to2r-644114e5@ssh.runpod.io:/workspace/FlavorGraph/training/output/flavorgraph_llama_v1/ ./
```

---

## Timeline

- **Upload**: 1-2 minutes (11MB)
- **Setup**: 3-5 minutes
- **Data generation**: 5-10 minutes
- **Training**: 2-3 hours on A100
- **Download**: 1-2 minutes (200MB)

**Total**: ~2.5-3 hours
**Cost**: ~$3-4 on A100

---

## Troubleshooting

### Connection issues
- Make sure RunPod pod is running
- Check SSH key: `ls -la ~/.ssh/id_ed25519`

### Upload fails
- Try smaller chunks
- Or use RunPod web UI file upload

### Training fails
- Check: `cat training/output/*/trainer_state.json`
- Try: Reduce `batch_size` in config to 2

### Out of memory
- Edit config: `batch_size: 2`
- Or use 8-bit instead: `load_in_4bit: false, load_in_8bit: true`
