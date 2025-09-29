#!/usr/bin/env python3
"""
Estimate training time for FlavorGraph model
"""

import yaml
import json
import math

def estimate_training_time(config_path: str):
    """Estimate training time based on configuration"""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load training data to get dataset size
    train_file = config['training_config']['train_file']
    with open(train_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    dataset_size = len(data)
    validation_split = config['training_config'].get('validation_split', 0.1)
    train_size = int(dataset_size * (1 - validation_split))
    
    # Training parameters
    num_epochs = config['training_config']['num_epochs']
    batch_size = config['training_config']['batch_size']
    gradient_accumulation_steps = config['training_config']['gradient_accumulation_steps']
    effective_batch_size = batch_size * gradient_accumulation_steps
    
    # Calculate steps
    steps_per_epoch = math.ceil(train_size / effective_batch_size)
    total_steps = steps_per_epoch * num_epochs
    
    # Estimate time per step (varies by hardware)
    hardware = config['hardware_config']['machine_type']
    if 'A100' in hardware:
        time_per_step = 0.5  # seconds
    elif 'RTX4090' in hardware:
        time_per_step = 0.8  # seconds
    elif 'RTX4000' in hardware:
        time_per_step = 1.2  # seconds
    else:
        time_per_step = 1.0  # seconds
    
    # Calculate total time
    total_seconds = total_steps * time_per_step
    total_hours = total_seconds / 3600
    total_minutes = total_seconds / 60
    
    print(f"📊 Training Time Estimation")
    print(f"=" * 40)
    print(f"Dataset size: {dataset_size:,} examples")
    print(f"Training size: {train_size:,} examples")
    print(f"Validation size: {dataset_size - train_size:,} examples")
    print(f"")
    print(f"Training parameters:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"")
    print(f"Steps calculation:")
    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Total steps: {total_steps:,}")
    print(f"")
    print(f"Time estimation ({hardware}):")
    print(f"  Time per step: {time_per_step:.1f}s")
    print(f"  Total time: {total_hours:.1f} hours ({total_minutes:.0f} minutes)")
    print(f"")
    
    # Add overhead estimates
    overhead_hours = 0.5  # Data loading, model saving, etc.
    total_with_overhead = total_hours + overhead_hours
    
    print(f"With overhead (data loading, saving, etc.):")
    print(f"  Estimated total time: {total_with_overhead:.1f} hours")
    
    # Recommendations
    if total_hours < 1:
        print(f"\n⚠️  WARNING: Training time is very short ({total_hours:.1f}h)")
        print(f"   This may indicate insufficient training for good results")
    elif total_hours > 12:
        print(f"\n⚠️  WARNING: Training time is very long ({total_hours:.1f}h)")
        print(f"   Consider reducing epochs or increasing batch size")
    else:
        print(f"\n✅ Training duration looks good for thorough learning")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Estimate FlavorGraph training time')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    
    args = parser.parse_args()
    estimate_training_time(args.config)
