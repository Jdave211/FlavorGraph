# FlavorGraph LLaMA Training

Fine-tune LLaMA models on FlavorGraph data to create an AI that understands food ingredients, chemical relationships, and recipe compatibility.

## Overview

This training pipeline transforms FlavorGraph's knowledge graph (8K nodes, 147K edges) into instruction-following datasets for LLaMA fine-tuning. The resulting model understands:

- **Ingredient Pairing**: Recommendations based on flavor chemistry
- **Chemical Relationships**: How compounds contribute to taste
- **Recipe Analysis**: Ingredient compatibility scoring
- **Substitutions**: Alternative ingredients with similar profiles

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Requirements:**
- PyTorch 2.0+
- Transformers 4.36+
- PEFT (LoRA)
- BitsAndBytes (quantization)
- Weights & Biases (optional, for monitoring)

### 2. Validate Data

```bash
python training/preprocess_data.py
```

This checks that all required data files exist:
- `input/nodes_191120.csv` - Ingredient and compound nodes
- `input/edges_191120.csv` - Relationship edges
- `input/recipes/extracted_recipes.json` - Recipe data
- `input/compound_flavors/*.json` - Flavor profiles

### 3. Generate Training Data

```bash
python training/generate_llama_training_data.py
```

This creates `training/data/flavorgraph_training_data.jsonl` containing ~3,300 instruction examples across 6 task types:

| Task Type | Count | Description |
|-----------|-------|-------------|
| Ingredient Pairing | 1,000 | "What pairs well with X?" |
| Flavor Profiles | 500 | "Describe the flavor of X" |
| Recipe Analysis | 800 | "Analyze this combination" |
| Substitutions | 500 | "What can substitute X?" |
| Chemical Roles | 300 | "What role does compound X play?" |
| Hub Ingredients | 200 | "Why is X versatile?" |

### 4. Configure Training

Edit `training/config_llama_training.yaml` to customize:

```yaml
model_config:
  base_model: "meta-llama/Llama-2-7b-hf"  # or Llama-3-8b
  use_lora: true
  lora_r: 16
  load_in_4bit: true  # QLoRA for memory efficiency

training_config:
  num_epochs: 3
  batch_size: 4
  learning_rate: 2.0e-4
  gradient_accumulation_steps: 4
```

**Model Options:**
- `meta-llama/Llama-2-7b-hf` - LLaMA 2 7B (recommended)
- `meta-llama/Llama-2-13b-hf` - LLaMA 2 13B (requires more memory)
- `meta-llama/Llama-3-8b` - LLaMA 3 8B (newer architecture)
- `mistralai/Mistral-7B-v0.1` - Mistral alternative

### 5. Train the Model

```bash
python training/train_llama.py --config training/config_llama_training.yaml
```

**Training Time Estimates** (LLaMA 2 7B, 3 epochs):
- A100 (40GB): ~2-3 hours
- RTX 4090 (24GB): ~4-5 hours
- V100 (16GB): ~6-8 hours

**Memory Requirements:**
- 4-bit quantization (QLoRA): ~8-10 GB VRAM
- 8-bit quantization: ~12-15 GB VRAM
- Full precision: ~28+ GB VRAM

### 6. Evaluate the Model

```bash
python training/evaluate_model.py \
  --model training/output/flavorgraph_llama_v1 \
  --base_model meta-llama/Llama-2-7b-hf \
  --output evaluation_results.json
```

## Training Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ FlavorGraph Data Sources                                     │
├─────────────────────────────────────────────────────────────┤
│ • nodes_191120.csv        (8K nodes)                         │
│ • edges_191120.csv        (147K edges)                       │
│ • extracted_recipes.json  (recipes)                          │
│ • compound_flavors/       (chemical profiles)                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ generate_llama_training_data.py                              │
├─────────────────────────────────────────────────────────────┤
│ Converts graph data into instruction-response pairs         │
│ • Ingredient pairing instructions                            │
│ • Flavor profile descriptions                                │
│ • Recipe analysis tasks                                      │
│ • Chemical relationship explanations                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ flavorgraph_training_data.jsonl                              │
├─────────────────────────────────────────────────────────────┤
│ Formatted as instruction-following examples:                 │
│                                                               │
│ {                                                             │
│   "instruction": "What pairs well with tomato?",             │
│   "input": "",                                               │
│   "output": "Basil, mozzarella, garlic...",                  │
│   "metadata": {...}                                          │
│ }                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ train_llama.py                                               │
├─────────────────────────────────────────────────────────────┤
│ • Load LLaMA base model                                      │
│ • Apply 4-bit quantization (QLoRA)                           │
│ • Add LoRA adapters (rank 16)                                │
│ • Train on instruction data                                  │
│ • Save fine-tuned weights                                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Fine-tuned Model Output                                      │
├─────────────────────────────────────────────────────────────┤
│ • LoRA adapter weights                                       │
│ • Tokenizer files                                            │
│ • Training config                                            │
│ • Model card README                                          │
└─────────────────────────────────────────────────────────────┘
```

## Configuration Options

### Model Selection

**LLaMA 2 vs LLaMA 3:**
- **LLaMA 2 7B**: More stable, well-tested, 4K context
- **LLaMA 3 8B**: Better performance, 8K context, newer architecture

**Quantization:**
- **4-bit (QLoRA)**: Best for consumer GPUs, minimal quality loss
- **8-bit**: Middle ground between memory and quality
- **FP16**: Best quality, requires high-end GPUs

### LoRA Parameters

```yaml
lora_r: 16              # Rank (8-64, higher = more capacity)
lora_alpha: 32          # Scaling factor (usually 2x rank)
lora_dropout: 0.1       # Regularization
lora_target_modules:    # Which layers to adapt
  - q_proj              # Query projection
  - v_proj              # Value projection
  - k_proj              # Key projection (optional)
  - o_proj              # Output projection (optional)
```

**Recommendations:**
- Start with `r=16`, increase to 32-64 for complex tasks
- Use `alpha=2*r` as a rule of thumb
- Target at least `q_proj` and `v_proj`

### Training Hyperparameters

```yaml
num_epochs: 3                        # Number of passes through data
batch_size: 4                        # Samples per GPU
gradient_accumulation_steps: 4       # Effective batch = 4 * 4 = 16
learning_rate: 2.0e-4                # LoRA learning rate
lr_scheduler: cosine                 # Learning rate schedule
```

**Tips:**
- Effective batch size of 16-32 works well
- Learning rate 2e-4 to 5e-4 for LoRA
- 3 epochs usually sufficient for fine-tuning

## Using the Fine-tuned Model

### Inference Script

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    "training/output/flavorgraph_llama_v1"
)

tokenizer = AutoTokenizer.from_pretrained(
    "training/output/flavorgraph_llama_v1"
)

# Generate response
def ask_flavorgraph(question: str) -> str:
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_length=300,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:")[-1].strip()

# Examples
print(ask_flavorgraph("What ingredients pair well with tomato?"))
print(ask_flavorgraph("Describe the flavor profile of vanilla."))
print(ask_flavorgraph("What can I substitute for butter?"))
```

### Merge LoRA Weights (Optional)

To create a standalone model without requiring PEFT:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load and merge
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base, "training/output/flavorgraph_llama_v1")
merged = model.merge_and_unload()

# Save merged model
merged.save_pretrained("training/output/flavorgraph_llama_merged")
```

## Monitoring Training

### Weights & Biases

Enable in config:

```yaml
output_config:
  use_wandb: true
  wandb_project: "flavorgraph-llama"
  wandb_entity: "your-username"
```

Then login:

```bash
wandb login
```

### Key Metrics to Watch

- **Training Loss**: Should decrease steadily
- **Validation Loss**: Should decrease without increasing (no overfitting)
- **Learning Rate**: Cosine schedule starts high, decreases
- **GPU Memory**: Should be stable throughout

## Troubleshooting

### Out of Memory (OOM)

**Solutions:**
1. Reduce `batch_size` (e.g., from 4 to 2 or 1)
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Enable `gradient_checkpointing: true`
4. Use 4-bit quantization instead of 8-bit
5. Reduce `max_seq_length` (512 → 256)

### Slow Training

**Solutions:**
1. Use `bf16: true` on A100/H100 GPUs
2. Increase `dataloader_num_workers`
3. Reduce logging frequency
4. Use faster disk I/O (SSD)

### Poor Quality Outputs

**Solutions:**
1. Increase training epochs (3 → 5)
2. Increase LoRA rank (`lora_r: 16` → `32`)
3. Generate more training data
4. Adjust learning rate
5. Check data quality with `preprocess_data.py`

### Model Not Learning

**Symptoms:** Loss not decreasing
**Solutions:**
1. Increase learning rate (2e-4 → 5e-4)
2. Check data format is correct
3. Increase LoRA rank
4. Verify gradient accumulation is working
5. Check base model loaded correctly

## Advanced Topics

### Multi-GPU Training

```yaml
hardware_config:
  num_gpus: 4
  distributed: true
```

### DeepSpeed Integration

Create `ds_config.json`:

```json
{
  "train_batch_size": 16,
  "gradient_accumulation_steps": 4,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 2e-4
    }
  },
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2
  }
}
```

Reference in config:

```yaml
hardware_config:
  deepspeed_config: "ds_config.json"
```

### Custom Test Sets

Create `test_cases.json`:

```json
{
  "pairing": [
    {
      "ingredient": "basil",
      "expected_pairings": ["tomato", "mozzarella", "pine nuts"]
    }
  ],
  "flavor": [
    {
      "ingredient": "lemon",
      "expected_flavors": ["sour", "aromatic"]
    }
  ]
}
```

Use in evaluation:

```bash
python training/evaluate_model.py \
  --model output/flavorgraph_llama_v1 \
  --test_data test_cases.json
```

## File Structure

```
training/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── config_llama_training.yaml          # Training configuration
├── generate_llama_training_data.py     # Data generator
├── train_llama.py                      # Training script
├── evaluate_model.py                   # Evaluation script
├── preprocess_data.py                  # Data validation
├── data/                               # Generated training data
│   ├── flavorgraph_training_data.jsonl
│   └── training_metadata.json
└── output/                             # Trained models
    └── flavorgraph_llama_v1/
        ├── adapter_config.json
        ├── adapter_model.bin
        ├── tokenizer files
        ├── training_config.yaml
        └── README.md
```

## Citation

If you use this training pipeline, please cite:

```bibtex
@article{park2021flavorgraph,
  title={FlavorGraph: a large-scale food-chemical graph for generating food representations and recommending food pairings},
  author={Park, Donghyeon and Kim, Keonwoo and Kim, Seoyoon and Spranger, Michael and Kang, Jaewoo},
  journal={Scientific reports},
  volume={11},
  number={1},
  pages={1--13},
  year={2021},
  publisher={Nature Publishing Group}
}
```

## Support

- Issues: File on GitHub
- Questions: Check evaluation results for model performance
- Improvements: PRs welcome!

## License

Apache License 2.0
