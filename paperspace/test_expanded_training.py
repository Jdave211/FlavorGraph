#!/usr/bin/env python3
"""
Test script to verify the expanded training dataset works
"""

import json
import random

def test_expanded_dataset():
    """Test the expanded training dataset"""
    
    print("🧪 Testing Expanded Training Dataset")
    print("=" * 40)
    
    # Load the dataset
    with open('training_data/expanded_training.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    
    print(f"Total examples: {len(data):,}")
    
    # Test a few random examples
    print("\n📝 Sample Training Examples:")
    print("-" * 40)
    
    for i, example in enumerate(random.sample(data, 5)):
        print(f"\nExample {i+1}:")
        print(f"Instruction: {example['instruction']}")
        print(f"Input: {example['input']}")
        print(f"Output: {example['output']}")
        print(f"Output length: {len(example['output'].split())} words")
    
    # Check for quality issues
    print(f"\n🔍 Quality Check:")
    print("-" * 40)
    
    # Check for repetitive patterns
    problematic_patterns = ['is a chemical compound', 'is a ingredient', 'node_id:']
    issues = 0
    
    for example in data:
        output = example['output']
        if any(pattern in output for pattern in problematic_patterns):
            issues += 1
    
    print(f"Examples with problematic patterns: {issues} ({issues/len(data)*100:.1f}%)")
    
    # Check output lengths
    output_lengths = [len(example['output'].split()) for example in data]
    avg_length = sum(output_lengths) / len(output_lengths)
    short_outputs = sum(1 for length in output_lengths if length < 10)
    
    print(f"Average output length: {avg_length:.1f} words")
    print(f"Short outputs (<10 words): {short_outputs} ({short_outputs/len(data)*100:.1f}%)")
    
    # Check instruction diversity
    instructions = [example['instruction'] for example in data]
    unique_instructions = len(set(instructions))
    print(f"Unique instructions: {unique_instructions:,}")
    
    # Overall assessment
    print(f"\n✅ Overall Assessment:")
    print(f"  Dataset size: {'GOOD' if len(data) > 10000 else 'POOR'}")
    print(f"  Quality: {'EXCELLENT' if issues < len(data) * 0.05 else 'POOR'}")
    print(f"  Diversity: {'EXCELLENT' if unique_instructions > len(data) * 0.8 else 'POOR'}")
    
    if issues < len(data) * 0.05 and unique_instructions > len(data) * 0.8:
        print(f"\n🎉 Dataset is ready for training!")
        return True
    else:
        print(f"\n⚠️  Dataset needs improvement before training")
        return False

if __name__ == "__main__":
    test_expanded_dataset()
