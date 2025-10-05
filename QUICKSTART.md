# FlavorGraph LLaMA Training - Quick Start Guide

## 🚀 One-Command Setup

```bash
# Check if everything is ready
python training/check_setup.py

# Run full pipeline (data generation → training)
bash training/run_full_pipeline.sh
```

## 📋 Step-by-Step

### 1. Check Setup (30 seconds)

```bash
python training/check_setup.py
```

This validates:
- ✅ Python version (3.8+)
- ✅ GPU availability
- ✅ Required packages
- ✅ Data files
- ✅ Disk space

### 2. Install Dependencies (5 minutes)

```bash
pip install -r training/requirements.txt
```

Key packages:
- `transformers` - Hugging Face models
- `peft` - LoRA fine-tuning
- `bitsandbytes` - Quantization
- `datasets` - Data loading

### 3. Generate Training Data (5-10 minutes)

```bash
python training/generate_llama_training_data.py
```

Creates:
- `training/data/flavorgraph_training_data.jsonl` (~3,300 examples)
- Combines ingredients, recipes, chemical data

### 4. Train Model (2-8 hours depending on GPU)

```bash
python training/train_llama.py --config training/config_llama_training.yaml
```

Trains:
- LLaMA 2 7B with LoRA
- 4-bit quantization (~8GB VRAM)
- 3 epochs

### 5. Evaluate (10 minutes)

```bash
python training/evaluate_model.py \
  --model training/output/flavorgraph_llama_v1 \
  --base_model meta-llama/Llama-2-7b-hf
```

## 🎯 Expected Outputs

After training, you'll have:

```
training/output/flavorgraph_llama_v1/
├── adapter_config.json          # LoRA configuration
├── adapter_model.bin            # Trained weights
├── tokenizer files              # For inference
└── README.md                    # Model card
```

## 💡 Quick Test

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, "training/output/flavorgraph_llama_v1")
tokenizer = AutoTokenizer.from_pretrained("training/output/flavorgraph_llama_v1")

# Ask
prompt = "### Instruction:\nWhat pairs well with tomato?\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
```

## ⚙️ Configuration Cheat Sheet

Edit `training/config_llama_training.yaml`:

**Reduce memory usage:**
```yaml
batch_size: 2                    # Down from 4
gradient_accumulation_steps: 8   # Up from 4
```

**Speed up training:**
```yaml
bf16: true                       # Use on A100/H100
num_epochs: 2                    # Down from 3
```

**Improve quality:**
```yaml
num_epochs: 5                    # Up from 3
lora_r: 32                       # Up from 16
learning_rate: 5e-4              # Up from 2e-4
```

## 🔧 Common Issues

### Out of Memory
```bash
# Edit config: reduce batch_size to 1 or 2
# Or use smaller model: meta-llama/Llama-2-7b-hf
```

### Missing Data
```bash
# Check data files exist
python training/preprocess_data.py
```

### Slow Training
```bash
# Enable mixed precision in config
bf16: true  # or fp16: true
```

### No GPU
```bash
# Use cloud services:
# - Google Colab (free T4)
# - Paperspace (A4000+)
# - RunPod (A100/H100)
```

## 📊 Performance Expectations

| Metric | Expected Result |
|--------|----------------|
| Ingredient Pairing | 70-85% accuracy |
| Flavor Profiles | 65-80% accuracy |
| Recipe Analysis | 60-75% accuracy |
| Substitutions | 70-80% accuracy |

## 🎓 What the Model Learns

**Input:** "What ingredients pair well with chocolate?"

**Output:** "Based on FlavorGraph analysis, here are excellent pairings for chocolate:
- coffee (compatibility: 0.89)
- vanilla (compatibility: 0.85)
- strawberry (compatibility: 0.82)
- mint (compatibility: 0.78)

These ingredients share complementary flavor compounds that enhance chocolate's aromatic profile..."

## 📚 More Help

- **Full guide:** `training/README.md`
- **Setup details:** `TRAINING_SETUP.md`
- **Config reference:** `training/config_llama_training.yaml`

## 🚀 TL;DR - Fastest Path

```bash
# 1. Check everything is ready
python training/check_setup.py

# 2. Install (if needed)
pip install -r training/requirements.txt

# 3. Run everything
bash training/run_full_pipeline.sh

# Done! Model will be in training/output/flavorgraph_llama_v1/
```

---

**Questions?** Check `training/README.md` or run `python training/check_setup.py` for diagnostics.
