#!/usr/bin/env python3
"""
Clean and improve FlavorGraph training data to prevent overfitting
Fixes repetitive patterns and improves data quality
"""

import json
import random
from typing import List, Dict, Any
from collections import Counter

def clean_training_data(input_file: str, output_file: str):
    """Clean training data to prevent overfitting and repetitive outputs"""
    
    print(f"Loading training data from {input_file}")
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    print(f"Original data size: {len(data)}")
    
    # Remove problematic patterns
    cleaned_data = []
    problematic_patterns = [
        "is a chemical compound.",
        "is a ingredient.",
        "classified as",
        "node_id:",
        "frequently appear together"
    ]
    
    for item in data:
        output = item.get('output', '')
        
        # Skip if output is too short or repetitive
        if len(output.split()) < 3:
            continue
            
        # Skip if output contains problematic patterns
        if any(pattern in output for pattern in problematic_patterns):
            continue
            
        # Skip if output is just the input + exclamation
        if output.strip().endswith('!') and len(output.split()) < 5:
            continue
            
        cleaned_data.append(item)
    
    print(f"After removing problematic patterns: {len(cleaned_data)}")
    
    # Add variety to prevent overfitting
    enhanced_data = []
    
    for item in cleaned_data:
        # Add original item
        enhanced_data.append(item)
        
        # Create variations with different phrasings
        if random.random() < 0.3:  # 30% chance to add variation
            variation = create_variation(item)
            if variation:
                enhanced_data.append(variation)
    
    # Shuffle to prevent order bias
    random.shuffle(enhanced_data)
    
    # Limit dataset size to prevent overfitting
    max_size = min(5000, len(enhanced_data))
    enhanced_data = enhanced_data[:max_size]
    
    print(f"Final dataset size: {len(enhanced_data)}")
    
    # Save cleaned data
    with open(output_file, 'w') as f:
        for item in enhanced_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Cleaned data saved to {output_file}")

def create_variation(item: Dict[str, Any]) -> Dict[str, Any]:
    """Create a variation of a training item to add diversity"""
    
    instruction = item.get('instruction', '')
    input_text = item.get('input', '')
    output = item.get('output', '')
    
    # Different instruction phrasings
    instruction_variations = {
        "Explain the taste properties of": [
            "Describe the flavor characteristics of",
            "What are the taste notes of",
            "Analyze the flavor profile of",
            "What does taste like"
        ],
        "What are the chemical flavor characteristics of this compound?": [
            "Describe the flavor compounds in",
            "What flavor molecules are in",
            "Analyze the taste chemistry of",
            "What gives its flavor"
        ],
        "What type of ingredient is": [
            "What category does belong to",
            "Classify the ingredient",
            "What kind of food is",
            "Describe the ingredient type of"
        ],
        "Analyze this ingredient combination": [
            "How do these ingredients work together:",
            "What happens when you combine:",
            "Evaluate this food pairing:",
            "Why do these ingredients pair well:"
        ]
    }
    
    # Find matching instruction pattern
    for pattern, variations in instruction_variations.items():
        if pattern in instruction:
            new_instruction = random.choice(variations)
            return {
                "instruction": new_instruction,
                "input": input_text,
                "output": output
            }
    
    return None

def analyze_data_quality(file_path: str):
    """Analyze the quality of training data"""
    
    print(f"\nAnalyzing data quality in {file_path}")
    
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Count output patterns
    output_patterns = Counter()
    for item in data:
        output = item.get('output', '')
        if output.endswith('.'):
            output_patterns['ends_with_period'] += 1
        if output.endswith('!'):
            output_patterns['ends_with_exclamation'] += 1
        if 'node_id:' in output:
            output_patterns['contains_node_id'] += 1
        if len(output.split()) < 5:
            output_patterns['very_short'] += 1
    
    print("Output pattern analysis:")
    for pattern, count in output_patterns.most_common():
        percentage = (count / len(data)) * 100
        print(f"  {pattern}: {count} ({percentage:.1f}%)")
    
    # Check for repetitive outputs
    output_texts = [item.get('output', '') for item in data]
    unique_outputs = len(set(output_texts))
    repetition_rate = (len(data) - unique_outputs) / len(data) * 100
    
    print(f"\nRepetition analysis:")
    print(f"  Total outputs: {len(data)}")
    print(f"  Unique outputs: {unique_outputs}")
    print(f"  Repetition rate: {repetition_rate:.1f}%")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean FlavorGraph training data')
    parser.add_argument('--input', default='training_data/combined_training.jsonl', 
                       help='Input training file')
    parser.add_argument('--output', default='training_data/cleaned_training.jsonl',
                       help='Output cleaned file')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze data quality')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_data_quality(args.input)
    else:
        clean_training_data(args.input, args.output)
        analyze_data_quality(args.output)
