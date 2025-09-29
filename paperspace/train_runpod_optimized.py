#!/usr/bin/env python3
"""
Optimized FlavorGraph Training Script for RunPod
Includes early stopping, better monitoring, and overfitting prevention
"""

import os
import sys
import yaml
import json
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
    EarlyStoppingCallback
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class RunPodTrainingArguments:
    """Custom training arguments for RunPod"""
    config_path: str = field(metadata={"help": "Path to YAML config file"})
    data_dir: str = field(default="training_data", metadata={"help": "Training data directory"})
    resume_from_checkpoint: Optional[str] = field(default=None, metadata={"help": "Resume training from checkpoint"})
    clean_data: bool = field(default=True, metadata={"help": "Clean training data before training"})

class RunPodFlavorGraphTrainer:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        self.output_dir = Path(self.config['output_config']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.embeddings_reference = None
        
    def load_config(self) -> Dict[str, Any]:
        """Load training configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_wandb(self):
        """Setup Weights & Biases logging"""
        if self.config['output_config']['use_wandb']:
            wandb.init(
                project=self.config['output_config']['wandb_project'],
                entity=self.config['output_config']['wandb_entity'],
                name=self.config['output_config']['run_name'],
                config=self.config
            )
    
    def clean_training_data(self):
        """Clean training data to prevent overfitting"""
        print("🧹 Cleaning training data...")
        
        input_file = self.config['training_config']['train_file']
        cleaned_file = input_file.replace('.jsonl', '_cleaned.jsonl')
        
        # Run data cleaning
        os.system(f"python3 clean_training_data.py --input {input_file} --output {cleaned_file}")
        
        # Update config to use cleaned data
        self.config['training_config']['train_file'] = cleaned_file
        print(f"✅ Using cleaned data: {cleaned_file}")
    
    def load_embeddings_reference(self):
        """Load FlavorGraph embeddings for reference"""
        try:
            embeddings_path = "../output/FlavorGraph+CSL-embedding_M11-metapath_300-dim_0.0025-initial_lr_3-window_size_1-iterations_5-min_count-_False-isCSP_0.0001-CSPcoef.pickle"
            if os.path.exists(embeddings_path):
                import pickle
                with open(embeddings_path, 'rb') as f:
                    self.embeddings_reference = pickle.load(f)
                print(f"✅ Loaded {len(self.embeddings_reference)} FlavorGraph embeddings")
            else:
                print("⚠️  FlavorGraph embeddings not found, continuing without reference")
        except Exception as e:
            print(f"⚠️  Could not load embeddings: {e}")
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with LoRA"""
        model_config = self.config['model_config']
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config['base_model'],
            padding_side="left"
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        if model_config.get('load_in_4bit', False):
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_config['base_model'],
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_config['base_model'],
                torch_dtype=torch.float16 if self.config['training_config'].get('fp16', False) else torch.float32,
                device_map="auto"
            )
        
        # Prepare model for k-bit training if needed
        if model_config.get('load_in_4bit', False):
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Setup LoRA
        if model_config.get('use_lora', False):
            lora_config = LoraConfig(
                r=model_config['lora_r'],
                lora_alpha=model_config['lora_alpha'],
                target_modules=model_config['lora_target_modules'],
                lora_dropout=model_config['lora_dropout'],
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        print(f"✅ Model loaded: {model_config['base_model']}")
    
    def load_and_prepare_dataset(self):
        """Load and prepare training dataset"""
        train_config = self.config['training_config']
        
        # Load dataset
        dataset = load_dataset('json', data_files=train_config['train_file'], split='train')
        
        # Create instruction format
        def format_instruction(example):
            instruction = example['instruction']
            input_text = example['input']
            output = example['output']
            
            # Format as conversation
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            return {"text": text}
        
        dataset = dataset.map(format_instruction)
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=train_config['max_seq_length'],
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Split into train/eval
        if train_config.get('validation_split', 0) > 0:
            split_dataset = tokenized_dataset.train_test_split(
                test_size=train_config['validation_split'],
                seed=42
            )
            train_dataset = split_dataset['train']
            eval_dataset = split_dataset['test']
            print(f"📊 Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
        else:
            train_dataset = tokenized_dataset
            eval_dataset = None
            print(f"📊 Train: {len(train_dataset)} (no validation split)")
        
        return train_dataset, eval_dataset
    
    def setup_training_arguments(self):
        """Setup Hugging Face training arguments with early stopping"""
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
            lr_scheduler_type=train_config.get('lr_scheduler', 'linear'),
            
            # Memory and performance
            gradient_checkpointing=train_config.get('gradient_checkpointing', True),
            dataloader_num_workers=train_config.get('dataloader_num_workers', 2),
            dataloader_pin_memory=self.config['hardware_config'].get('dataloader_pin_memory', True),
            
            # Mixed precision
            fp16=train_config.get('fp16', False),
            bf16=train_config.get('bf16', False),
            
            # Logging and saving
            logging_steps=train_config['logging_steps'],
            save_steps=train_config['save_steps'],
            eval_steps=train_config.get('eval_steps', 100),
            evaluation_strategy="steps" if train_config.get('eval_steps') else "no",
            save_total_limit=train_config.get('save_total_limit', 2),
            
            # Early stopping
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Misc
            remove_unused_columns=False,
            report_to="wandb" if output_config['use_wandb'] else None,
            push_to_hub=output_config.get('push_to_hub', False),
            hub_model_id=output_config.get('hub_model_id'),
        )
        
        return training_args
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Calculate perplexity
        shift_logits = predictions[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        perplexity = torch.exp(loss)
        
        return {"perplexity": perplexity.item()}
    
    def train(self):
        """Main training loop with early stopping"""
        print("🎯 Starting Optimized FlavorGraph Training")
        print("=" * 50)
        
        # Setup all components
        self.setup_wandb()
        self.load_embeddings_reference()
        
        # Clean data if requested
        if self.config.get('clean_data', True):
            self.clean_training_data()
        
        self.setup_model_and_tokenizer()
        
        # Load and prepare data
        train_dataset, eval_dataset = self.load_and_prepare_dataset()
        
        # Setup training arguments
        training_args = self.setup_training_arguments()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Early stopping callback
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=self.config['training_config'].get('early_stopping_patience', 3),
            early_stopping_threshold=self.config['training_config'].get('early_stopping_threshold', 0.01)
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics if eval_dataset else None,
            callbacks=[early_stopping] if eval_dataset else None,
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
    
    parser = argparse.ArgumentParser(description='Optimized FlavorGraph training for RunPod')
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--clean-data", action="store_true", default=True, help="Clean training data")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = RunPodFlavorGraphTrainer(args.config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
