#!/usr/bin/env python3
"""
FlavorGraph LLaMA Training Script
Fine-tunes LLaMA models on FlavorGraph data for food and chemical understanding
"""

import os
import sys
import yaml
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    set_seed
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Optional W&B
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  Weights & Biases not available. Install with: pip install wandb")


class FlavorGraphLLaMATrainer:
    """Main trainer class for FlavorGraph LLaMA fine-tuning"""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.tokenizer = None
        self.model = None

        # Setup output directory
        self.output_dir = Path(self.config['output_config']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("🚀 FlavorGraph LLaMA Training")
        print("=" * 70)
        print(f"📋 Config: {config_path}")
        print(f"📁 Output: {self.output_dir}")
        print("=" * 70 + "\n")

    def load_config(self) -> Dict[str, Any]:
        """Load training configuration from YAML"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def setup_wandb(self):
        """Initialize Weights & Biases tracking"""
        if not WANDB_AVAILABLE:
            print("⚠️  W&B not available, skipping...\n")
            return

        if self.config['output_config'].get('use_wandb', False):
            wandb.init(
                project=self.config['output_config']['wandb_project'],
                entity=self.config['output_config'].get('wandb_entity'),
                name=self.config['output_config']['run_name'],
                config=self.config
            )
            print("✅ Weights & Biases initialized\n")

    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with LoRA/QLoRA"""
        model_config = self.config['model_config']
        base_model = model_config['base_model']

        print(f"🤖 Loading base model: {base_model}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True
        )

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(f"   ✓ Tokenizer loaded (vocab size: {len(self.tokenizer)})")

        # Setup quantization config
        quantization_config = None
        if model_config.get('load_in_4bit', False):
            print("   ⚡ Configuring 4-bit quantization (QLoRA)...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type=model_config.get('bnb_4bit_quant_type', 'nf4'),
                bnb_4bit_use_double_quant=model_config.get('bnb_4bit_use_double_quant', True)
            )
        elif model_config.get('load_in_8bit', False):
            print("   ⚡ Configuring 8-bit quantization...")
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load model
        print("   📥 Loading model weights...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16 if not model_config.get('load_in_4bit', False) else None,
            trust_remote_code=True
        )

        # Prepare for k-bit training if quantized
        if quantization_config is not None:
            print("   🔧 Preparing model for k-bit training...")
            self.model = prepare_model_for_kbit_training(self.model)

        # Setup LoRA
        if model_config.get('use_lora', True):
            print("   🎯 Applying LoRA configuration...")

            lora_config = LoraConfig(
                r=model_config.get('lora_r', 16),
                lora_alpha=model_config.get('lora_alpha', 32),
                target_modules=model_config.get('lora_target_modules', ["q_proj", "v_proj"]),
                lora_dropout=model_config.get('lora_dropout', 0.1),
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )

            self.model = get_peft_model(self.model, lora_config)

            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_percent = 100 * trainable_params / total_params

            print(f"   📊 Trainable params: {trainable_params:,} / {total_params:,} ({trainable_percent:.2f}%)")

        print("✅ Model and tokenizer setup complete\n")

    def load_and_prepare_dataset(self):
        """Load and tokenize the training dataset"""
        train_config = self.config['training_config']
        train_file = train_config['train_file']

        print(f"📊 Loading dataset: {train_file}")

        # Check if file exists
        train_path = Path(train_file)
        if not train_path.exists():
            raise FileNotFoundError(
                f"Training file not found: {train_file}\n"
                f"Please run: python training/generate_llama_training_data.py"
            )

        # Load JSONL dataset
        dataset = load_dataset('json', data_files=str(train_path), split='train')
        print(f"   ✓ Loaded {len(dataset)} training examples")

        # Format dataset for instruction following
        def format_example(example):
            """Format as instruction-response pairs"""
            instruction = example['instruction']
            input_text = example.get('input', '')
            output_text = example['output']

            # Alpaca-style formatting
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

            full_text = prompt + output_text + self.tokenizer.eos_token
            return {"text": full_text}

        print("   🔄 Formatting examples...")
        dataset = dataset.map(format_example, remove_columns=dataset.column_names)

        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=train_config.get('max_seq_length', 512),
                return_overflowing_tokens=False,
            )

        print("   🔤 Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing"
        )

        # Split into train/validation
        validation_split = train_config.get('validation_split', 0.1)
        if validation_split > 0:
            print(f"   📊 Splitting dataset (validation: {validation_split * 100}%)...")
            split_dataset = tokenized_dataset.train_test_split(test_size=validation_split, seed=42)
            train_dataset = split_dataset['train']
            eval_dataset = split_dataset['test']
            print(f"   ✓ Train: {len(train_dataset)}, Validation: {len(eval_dataset)}")
        else:
            train_dataset = tokenized_dataset
            eval_dataset = None
            print(f"   ✓ Train: {len(train_dataset)} (no validation)")

        print("✅ Dataset preparation complete\n")
        return train_dataset, eval_dataset

    def setup_training_arguments(self):
        """Setup Hugging Face training arguments"""
        train_config = self.config['training_config']
        output_config = self.config['output_config']

        print("⚙️  Configuring training arguments...")

        # Calculate effective batch size
        per_device_batch = train_config['batch_size']
        grad_accum = train_config['gradient_accumulation_steps']
        num_gpus = self.config['hardware_config'].get('num_gpus', 1)
        effective_batch = per_device_batch * grad_accum * num_gpus

        print(f"   Batch size per device: {per_device_batch}")
        print(f"   Gradient accumulation: {grad_accum}")
        print(f"   Effective batch size: {effective_batch}")

        training_args = TrainingArguments(
            # Output and logging
            output_dir=str(self.output_dir),
            run_name=output_config['run_name'],

            # Training parameters
            num_train_epochs=train_config['num_epochs'],
            per_device_train_batch_size=per_device_batch,
            per_device_eval_batch_size=per_device_batch,
            gradient_accumulation_steps=grad_accum,

            # Optimization
            learning_rate=train_config['learning_rate'],
            weight_decay=train_config['weight_decay'],
            warmup_steps=train_config['warmup_steps'],
            lr_scheduler_type=train_config.get('lr_scheduler', 'cosine'),
            max_grad_norm=1.0,

            # Memory and performance
            gradient_checkpointing=train_config.get('gradient_checkpointing', True),
            dataloader_num_workers=train_config.get('dataloader_num_workers', 4),

            # Mixed precision
            fp16=train_config.get('fp16', False),
            bf16=train_config.get('bf16', True),

            # Logging and saving
            logging_steps=train_config['logging_steps'],
            save_steps=train_config['save_steps'],
            eval_steps=train_config.get('eval_steps', 500),
            evaluation_strategy="steps" if train_config.get('eval_steps') else "no",
            save_total_limit=train_config.get('save_total_limit', 3),
            load_best_model_at_end=True if train_config.get('eval_steps') else False,

            # Misc
            remove_unused_columns=False,
            report_to="wandb" if (output_config.get('use_wandb') and WANDB_AVAILABLE) else "none",
            push_to_hub=output_config.get('push_to_hub', False),
            hub_model_id=output_config.get('hub_model_id'),
        )

        print("✅ Training arguments configured\n")
        return training_args

    def train(self):
        """Main training loop"""
        print("🎯 Starting training pipeline...\n")

        # Set random seed for reproducibility
        set_seed(42)

        # Setup all components
        self.setup_wandb()
        self.setup_model_and_tokenizer()

        # Load and prepare data
        train_dataset, eval_dataset = self.load_and_prepare_dataset()

        # Setup training arguments
        training_args = self.setup_training_arguments()

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )

        # Initialize trainer
        print("🏋️  Initializing Trainer...")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        print("✅ Trainer initialized\n")

        # Start training
        print("=" * 70)
        print("🚀 BEGINNING TRAINING")
        print("=" * 70 + "\n")

        trainer.train()

        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE")
        print("=" * 70 + "\n")

        # Save final model
        print("💾 Saving final model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"   ✓ Model saved to {self.output_dir}")

        # Save training config
        config_output = self.output_dir / "training_config.yaml"
        with open(config_output, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        print(f"   ✓ Config saved to {config_output}")

        # Save model card
        model_card = self._generate_model_card()
        card_path = self.output_dir / "README.md"
        with open(card_path, 'w') as f:
            f.write(model_card)
        print(f"   ✓ Model card saved to {card_path}")

        print("\n✅ All artifacts saved!")

        # Finish wandb run
        if WANDB_AVAILABLE and self.config['output_config'].get('use_wandb'):
            wandb.finish()

        print("\n" + "=" * 70)
        print("🎉 TRAINING PIPELINE COMPLETE!")
        print("=" * 70)

    def _generate_model_card(self) -> str:
        """Generate model card documentation"""
        model_name = self.config['model_config']['base_model']
        output_name = self.config['output_config']['run_name']

        card = f"""# {output_name}

This model is a fine-tuned version of [{model_name}](https://huggingface.co/{model_name}) on FlavorGraph data.

## Model Description

FlavorGraph LLaMA is trained to understand:
- Food ingredient relationships and pairings
- Chemical compound flavor profiles
- Recipe compatibility analysis
- Ingredient substitutions based on flavor chemistry

## Training Data

The model was trained on FlavorGraph, a knowledge graph containing:
- 8K+ food ingredients and chemical compounds
- 147K+ relationship edges
- 1M+ recipe combinations
- Chemical flavor profile mappings

## Intended Use

This model is designed for:
- Food pairing recommendations
- Recipe development assistance
- Ingredient substitution suggestions
- Culinary education and exploration

## Training Details

- **Base Model**: {model_name}
- **Training Method**: LoRA fine-tuning
- **Hardware**: {self.config['hardware_config'].get('num_gpus', 1)} GPU(s)
- **Training Epochs**: {self.config['training_config']['num_epochs']}

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load model
base_model = AutoModelForCausalLM.from_pretrained("{model_name}")
model = PeftModel.from_pretrained(base_model, "{self.output_dir}")
tokenizer = AutoTokenizer.from_pretrained("{self.output_dir}")

# Generate response
prompt = "### Instruction:\\nWhat ingredients pair well with tomato?\\n\\n### Response:\\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
```

## Citation

If you use this model, please cite:

```bibtex
@article{{park2021flavorgraph,
  title={{FlavorGraph: a large-scale food-chemical graph for generating food representations and recommending food pairings}},
  author={{Park, Donghyeon and Kim, Keonwoo and Kim, Seoyoon and Spranger, Michael and Kang, Jaewoo}},
  journal={{Scientific reports}},
  volume={{11}},
  number={{1}},
  pages={{1--13}},
  year={{2021}},
  publisher={{Nature Publishing Group}}
}}
```
"""
        return card


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train FlavorGraph LLaMA Model")
    parser.add_argument(
        "--config",
        type=str,
        default="training/config_llama_training.yaml",
        help="Path to training config YAML"
    )

    args = parser.parse_args()

    # Check if config exists
    if not Path(args.config).exists():
        print(f"❌ Config file not found: {args.config}")
        print("   Please create a config file or specify a valid path.")
        sys.exit(1)

    # Initialize trainer
    trainer = FlavorGraphLLaMATrainer(args.config)

    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
