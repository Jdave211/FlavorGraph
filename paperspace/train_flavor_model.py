#!/usr/bin/env python3
"""
FlavorGraph AI Model Training Script for Paperspace
Fine-tunes language models on FlavorGraph data for ingredient/flavor understanding
"""

import os
import sys
import yaml
import json
import pickle
import torch
import wandb
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class FlavorGraphTrainingArguments:
    """Custom training arguments for FlavorGraph"""
    config_path: str = field(metadata={"help": "Path to YAML config file"})
    data_dir: str = field(default="training_data", metadata={"help": "Training data directory"})
    resume_from_checkpoint: Optional[str] = field(default=None, metadata={"help": "Resume training from checkpoint"})

class FlavorGraphTrainer:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        self.tokenizer = None
        self.model = None
        self.embeddings_reference = None
        
        # Setup directories
        self.output_dir = Path(self.config['output_config']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("🚀 Initializing FlavorGraph AI Trainer")
        print(f"📋 Config: {config_path}")
        print(f"📁 Output: {self.output_dir}")
        
    def load_config(self) -> Dict[str, Any]:
        """Load training configuration from YAML"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_wandb(self):
        """Initialize Weights & Biases tracking"""
        if self.config['output_config']['use_wandb']:
            wandb.init(
                project=self.config['output_config']['wandb_project'],
                entity=self.config['output_config'].get('wandb_entity'),
                name=self.config['output_config']['run_name'],
                config=self.config
            )
            print("✅ Weights & Biases initialized")
    
    def load_embeddings_reference(self):
        """Load FlavorGraph embeddings for evaluation"""
        embeddings_path = Path(self.config['training_config']['train_file']).parent / "embeddings_reference.pkl"
        if embeddings_path.exists():
            with open(embeddings_path, 'rb') as f:
                self.embeddings_reference = pickle.load(f)
            print(f"✅ Loaded {len(self.embeddings_reference)} reference embeddings")
        else:
            print("⚠️  No embeddings reference found - skipping embedding-based evaluation")
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with LoRA/QLoRA configuration"""
        model_config = self.config['model_config']
        base_model = model_config['base_model']
        
        print(f"🤖 Loading model: {base_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup quantization if specified
        quantization_config = None
        if model_config.get('load_in_4bit', False):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type=model_config.get('bnb_4bit_quant_type', 'nf4'),
                bnb_4bit_use_double_quant=model_config.get('bnb_4bit_use_double_quant', True)
            )
            print("⚡ Using 4-bit quantization (QLoRA)")
        elif model_config.get('load_in_8bit', False):
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            print("⚡ Using 8-bit quantization")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16 if not model_config.get('load_in_4bit', False) else None,
            trust_remote_code=True
        )
        
        # Prepare for k-bit training if quantized
        if quantization_config is not None:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Setup LoRA
        if model_config.get('use_lora', True):
            lora_config = LoraConfig(
                r=model_config.get('lora_r', 16),
                lora_alpha=model_config.get('lora_alpha', 32),
                target_modules=model_config.get('lora_target_modules', ["q_proj", "v_proj"]),
                lora_dropout=model_config.get('lora_dropout', 0.1),
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            print("✅ LoRA configuration applied")
        
        print(f"✅ Model and tokenizer loaded")
    
    def load_and_prepare_dataset(self):
        """Load and tokenize the training dataset"""
        train_config = self.config['training_config']
        train_file = train_config['train_file']
        
        print(f"📊 Loading dataset: {train_file}")
        
        # Load JSONL dataset
        dataset = load_dataset('json', data_files=train_file, split='train')
        print(f"📈 Loaded {len(dataset)} training examples")
        
        # Format dataset for instruction following
        def format_example(example):
            instruction = example['instruction']
            input_text = example.get('input', '')
            output_text = example['output']
            
            # Format as instruction-following conversation
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            
            full_text = prompt + output_text + self.tokenizer.eos_token
            return {"text": full_text}
        
        # Apply formatting
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
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing dataset"
        )
        
        # Split into train/validation
        validation_split = train_config.get('validation_split', 0.1)
        if validation_split > 0:
            split_dataset = tokenized_dataset.train_test_split(test_size=validation_split, seed=42)
            train_dataset = split_dataset['train']
            eval_dataset = split_dataset['test']
            print(f"📊 Train: {len(train_dataset)}, Validation: {len(eval_dataset)}")
        else:
            train_dataset = tokenized_dataset
            eval_dataset = None
            print(f"📊 Train: {len(train_dataset)} (no validation split)")
        
        return train_dataset, eval_dataset
    
    def setup_training_arguments(self):
        """Setup Hugging Face training arguments"""
        train_config = self.config['training_config']
        output_config = self.config['output_config']
        
        training_args = TrainingArguments(
            # Output and logging
            output_dir=str(self.output_dir),
            run_name=output_config['run_name'],
            
            # Training parameters
            num_train_epochs=train_config['num_epochs'],
            per_device_train_batch_size=train_config['batch_size'],
            per_device_eval_batch_size=train_config['batch_size'],
            gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
            
            # Optimization
            learning_rate=train_config['learning_rate'],
            weight_decay=train_config['weight_decay'],
            warmup_steps=train_config['warmup_steps'],
            lr_scheduler_type=train_config.get('lr_scheduler', 'cosine'),
            
            # Memory and performance
            gradient_checkpointing=train_config.get('gradient_checkpointing', True),
            dataloader_num_workers=train_config.get('dataloader_num_workers', 4),
            dataloader_pin_memory=self.config['hardware_config'].get('dataloader_pin_memory', True),
            
            # Mixed precision
            fp16=train_config.get('fp16', False),
            bf16=train_config.get('bf16', False),
            
            # Logging and saving
            logging_steps=train_config['logging_steps'],
            save_steps=train_config['save_steps'],
            eval_steps=train_config.get('eval_steps', 500),
            evaluation_strategy="steps" if train_config.get('eval_steps') else "no",
            save_total_limit=train_config.get('save_total_limit', 3),
            
            # Misc
            remove_unused_columns=False,
            report_to="wandb" if output_config['use_wandb'] else None,
            push_to_hub=output_config.get('push_to_hub', False),
            hub_model_id=output_config.get('hub_model_id'),
        )
        
        return training_args
    
    def compute_metrics(self, eval_pred):
        """Custom metrics computation for FlavorGraph evaluation"""
        # Basic perplexity computation
        predictions, labels = eval_pred
        
        # Shift predictions and labels for causal LM
        shift_predictions = predictions[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute loss
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        shift_predictions = shift_predictions.view(-1, shift_predictions.size(-1))
        shift_labels = shift_labels.view(-1)
        loss = loss_fct(torch.tensor(shift_predictions), torch.tensor(shift_labels))
        
        perplexity = torch.exp(loss)
        
        return {"perplexity": perplexity.item()}
    
    def train(self):
        """Main training loop"""
        print("🎯 Starting FlavorGraph AI Training")
        print("=" * 50)
        
        # Setup all components
        self.setup_wandb()
        self.load_embeddings_reference()
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
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics if eval_dataset else None,
        )
        
        # Start training
        print("🚀 Beginning training...")
        trainer.train()
        
        # Save final model
        print("💾 Saving final model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save training config
        config_path = self.output_dir / "training_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        print(f"✅ Training complete! Model saved to {self.output_dir}")
        
        # Finish wandb run
        if self.config['output_config']['use_wandb']:
            wandb.finish()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train FlavorGraph AI Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = FlavorGraphTrainer(args.config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
