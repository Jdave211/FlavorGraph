# FlavorGraph AI Training on Paperspace

This directory contains everything needed to train an AI model on Paperspace that understands FlavorGraph ingredient relationships, flavor profiles, and food pairing recommendations.

## 🎯 Overview

**FlavorGraph AI** is a fine-tuned language model that understands:
- **Ingredient Properties**: Categories, flavor profiles, chemical compositions
- **Food Pairing**: Which ingredients work well together and why
- **Substitutions**: Smart ingredient replacements based on flavor similarity
- **Recipe Analysis**: Understanding ingredient combinations and relationships

## 📊 Training Dataset

**Generated 8,830 training examples** across 4 categories:

| Category | Examples | Description |
|----------|----------|-------------|
| **Ingredient Knowledge** | 3,387 | Detailed descriptions of ingredients, categories, and properties |
| **Flavor Analysis** | 4,953 | Chemical compound flavor profiles and taste characteristics |
| **Substitution Pairs** | 290 | Category-aware ingredient substitution recommendations |
| **Recipe Analysis** | 200 | Ingredient combination patterns and co-occurrence insights |

### Data Sources
- **8,312 ingredients/compounds** from FlavorGraph nodes
- **1,509 trained embeddings** (300-dimensional vectors)
- **623 ingredient categories** (Fruit, Dairy, Spice, etc.)
- **1,651 compound flavor profiles** (30 flavor dimensions)
- **147,179 co-occurrence edges** between ingredients

## 🚀 Quick Start on Paperspace

### 1. Setup Environment
```bash
# Clone your FlavorGraph repository
git clone <your-repo-url>
cd FlavorGraph/paperspace

# Run setup script
./setup_paperspace.sh

# Login to Weights & Biases (optional)
wandb login
```

### 2. Choose Your Configuration

**Option A: Llama 7B with LoRA** (Recommended)
- **Hardware**: A100 40GB or 80GB
- **Memory**: ~20GB VRAM
- **Training Time**: 6-12 hours
```bash
python3 train_flavor_model.py --config configs/llama7b_lora.yaml
```

**Option B: Mistral 7B with QLoRA** (Memory Efficient)
- **Hardware**: RTX4000, RTX5000, or A100
- **Memory**: ~12GB VRAM
- **Training Time**: 8-16 hours
```bash
python3 train_flavor_model.py --config configs/mistral7b_qlora.yaml
```

### 3. Monitor Training
- **Weights & Biases**: Real-time metrics and loss curves
- **TensorBoard**: `tensorboard --logdir output/`
- **Console Logs**: Training progress and validation metrics

## 📁 File Structure

```
paperspace/
├── README.md                    # This file
├── setup_paperspace.sh          # Environment setup script
├── requirements.txt             # Python dependencies
│
├── prepare_training_data.py     # Generate training dataset
├── train_flavor_model.py        # Main training script
├── evaluate_model.py            # Model evaluation and testing
│
├── configs/
│   ├── llama7b_lora.yaml       # Llama 7B LoRA configuration
│   └── mistral7b_qlora.yaml    # Mistral 7B QLoRA configuration
│
└── training_data/               # Generated training data
    ├── combined_training.jsonl  # All training examples
    ├── ingredient_knowledge.jsonl
    ├── flavor_analysis.jsonl
    ├── substitution_pairs.jsonl
    ├── recipe_analysis.jsonl
    ├── embeddings_reference.pkl # FlavorGraph embeddings
    └── dataset_metadata.json   # Dataset statistics
```

## 🎛️ Configuration Options

### Model Architecture
- **Base Models**: Llama-2-7b, Mistral-7B-Instruct
- **Fine-tuning**: LoRA (Low-Rank Adaptation) or QLoRA (4-bit quantized)
- **Custom Layers**: Flavor projection layers for chemical compound understanding

### Training Parameters
- **Batch Size**: 4-16 (with gradient accumulation)
- **Learning Rate**: 1e-4 to 2e-4
- **Epochs**: 3-4
- **Sequence Length**: 512-1024 tokens
- **Mixed Precision**: FP16 or BF16

### Hardware Requirements

| Configuration | GPU | VRAM | Training Time |
|---------------|-----|------|---------------|
| Llama 7B LoRA | A100 40GB | 20GB | 6-12 hours |
| Llama 7B LoRA | A100 80GB | 20GB | 4-8 hours |
| Mistral QLoRA | RTX4000 | 12GB | 8-16 hours |
| Mistral QLoRA | RTX5000 | 12GB | 6-12 hours |

## 🧪 Model Evaluation

After training, evaluate your model:

```bash
# Run comprehensive evaluation
python3 evaluate_model.py --model_path output/llama7b_flavorgraph --output evaluation_results.json

# Interactive demo
python3 evaluate_model.py --model_path output/llama7b_flavorgraph --interactive
```

### Evaluation Metrics
- **Ingredient Knowledge**: Category mention rate, flavor understanding
- **Substitution Quality**: Category accuracy, explanation coherence
- **Flavor Understanding**: Chemical compound flavor profile accuracy

## 💡 Example Capabilities

### Ingredient Knowledge
```
Q: What can you tell me about garlic?
A: Garlic (node_id: 2819) is a Plant/Vegetable with strong sulfur_allium flavor characteristics. It has high umami (0.8) and aromatic (0.7) profiles. Common substitutes include shallot, onion powder, and garlic powder. It frequently appears with olive oil, onions, and herbs in Mediterranean cuisine.
```

### Substitution Recommendations
```
Q: What can I substitute for butter?
A: Based on category (Dairy) and functional similarity, coconut oil can substitute for butter. Both ingredients share similar fat content and cooking properties within the Essential Oil/Fat category (similarity: 0.742).
```

### Flavor Analysis
```
Q: Analyze the flavor profile of capsaicin
A: Capsaicin (node_id: 8704) is a chemical compound with primary flavor characteristics: heat=0.95, pungent_spicy=0.9. This capsaicinoid creates the burning sensation in spicy foods and activates TRPV1 receptors for the perception of heat.
```

## 🔧 Troubleshooting

### Common Issues

**Out of Memory Error**
- Reduce `batch_size` in config
- Enable `gradient_checkpointing`
- Use QLoRA instead of LoRA

**Slow Training**
- Increase `gradient_accumulation_steps`
- Use multiple GPUs with `device_map="auto"`
- Enable mixed precision (`fp16` or `bf16`)

**Poor Results**
- Increase training epochs
- Adjust learning rate
- Check data quality in `training_data/`

### Performance Optimization
- **Gradient Checkpointing**: Saves memory at cost of speed
- **Mixed Precision**: FP16/BF16 for faster training
- **Batch Size Tuning**: Balance memory usage and convergence

## 📈 Expected Results

After successful training, your model should achieve:
- **70%+ category accuracy** for ingredient substitutions
- **80%+ flavor understanding** for chemical compounds
- **Coherent explanations** for ingredient relationships
- **Recipe-appropriate suggestions** based on FlavorGraph data

## 🎉 Next Steps

1. **Deploy Model**: Create API endpoint or chat interface
2. **Integration**: Connect with existing FlavorGraph similarity tools
3. **Expansion**: Add more cuisines and ingredients to training data
4. **Fine-tuning**: Specialize for specific cooking styles or dietary restrictions

## 📚 Additional Resources

- [FlavorGraph Paper](https://link-to-paper)
- [Weights & Biases Dashboard](https://wandb.ai/your-project)
- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers)
- [PEFT (LoRA) Documentation](https://huggingface.co/docs/peft)

---

**Happy Training! 🚀**

For questions or issues, check the troubleshooting section above or refer to the FlavorGraph documentation.
