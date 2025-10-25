#!/usr/bin/env python3
"""
Expand and improve FlavorGraph training data
Creates a larger, higher-quality dataset for proper model training
"""

import json
import random
import os
from typing import List, Dict, Any
from collections import defaultdict

def load_flavorgraph_data():
    """Load FlavorGraph embeddings and metadata"""
    try:
        # Load embeddings
        import pickle
        with open('../output/FlavorGraph+CSL-embedding_M11-metapath_300-dim_0.0025-initial_lr_3-window_size_1-iterations_5-min_count-_False-isCSP_0.0001-CSPcoef.pickle', 'rb') as f:
            embeddings = pickle.load(f)
        
        # Load nodes data
        import pandas as pd
        nodes_df = pd.read_csv('../input/cleaned/nodes_cleaned_basic.csv')
        
        # Load category mapping
        category_df = pd.read_csv('../input/dict_ingr2cate - Top300+FDB400+HyperFoods104=616.csv')
        
        # Merge category data with nodes
        nodes_df = nodes_df.merge(category_df, left_on='cleaned_name', right_on='ingredient', how='left')
        
        # Load compound flavor data
        compound_df = pd.read_csv('../input/compound_flavors/compound_flavor_mappings.csv')
        
        return embeddings, nodes_df, compound_df
    except Exception as e:
        print(f"Warning: Could not load FlavorGraph data: {e}")
        return {}, None, None

def create_ingredient_knowledge_examples(nodes_df, embeddings, num_examples=5000):
    """Create detailed ingredient knowledge examples"""
    import pandas as pd
    examples = []

    if nodes_df is None:
        return examples

    # Filter ingredients - exclude malformed names
    valid_ingredients = nodes_df[
        (nodes_df['node_type'] == 'ingredient') &
        (~nodes_df['cleaned_name'].str.endswith('_with', na=False)) &
        (~nodes_df['cleaned_name'].str.contains('__', na=False))
    ].copy()

    # Sample ingredients
    ingredients = valid_ingredients.sample(min(num_examples, len(valid_ingredients)), random_state=42)

    # Category descriptions for more specific outputs
    category_descriptions = {
        'Spice': 'aromatic spice that adds warmth and complexity',
        'Plant/Vegetable': 'fresh vegetable that provides texture and nutrients',
        'Fruit': 'fruit that brings natural sweetness and acidity',
        'Meat/Animal Product': 'protein source that provides savory richness',
        'Seafood': 'seafood ingredient that offers delicate, umami-rich flavors',
        'Dairy': 'dairy product that adds creaminess and richness',
        'Cereal/Crop/Bean': 'staple ingredient that provides substance and nutrition',
        'Nut/Seed': 'nut or seed that contributes crunch and healthy fats',
        'Sauce/Powder/Dressing': 'seasoning that enhances and ties together flavors',
        'Beverage': 'beverage ingredient used in cooking and drinking',
        'Essential Oil/Fat': 'fat or oil that carries flavors and adds richness',
        'Bakery/Dessert/Snack': 'baked good or treat',
        'Fungus': 'mushroom or fungus that provides earthy, umami notes',
        'Flower': 'edible flower that adds delicate flavors and visual appeal'
    }

    for _, row in ingredients.iterrows():
        name = row['cleaned_name']

        # Skip if name is NaN or invalid
        if pd.isna(name) or not isinstance(name, str):
            continue

        category = row.get('category', 'Unknown')
        if pd.isna(category):
            category = 'Unknown'
        node_id = row['node_id']

        # Get category description
        category_desc = category_descriptions.get(category, 'ingredient')
        display_name = name.replace('_', ' ')

        # Create multiple instruction variations with specific outputs
        examples_data = [
            (f"Describe the culinary properties of {display_name}",
             f"{display_name.title()} is a {category_desc}. It's commonly used in various cuisines and adds depth to dishes through its unique flavor characteristics."),

            (f"What makes {display_name} unique in cooking?",
             f"As a {category}, {display_name} enhances dishes with its distinctive properties. Its versatility makes it valuable in many cooking traditions."),

            (f"How is {display_name} typically used in recipes?",
             f"{display_name.title()} is a {category_desc} prized for its ability to complement other ingredients and contribute both flavor and nutritional value."),

            (f"What are the key characteristics of {display_name}?",
             f"This {category_desc} is valued for adding complexity and depth to dishes, making it an essential component in many culinary applications."),

            (f"Explain the role of {display_name} in food preparation",
             f"{display_name.title()} functions as a {category_desc}, bringing distinctive qualities that can elevate both simple and complex dishes.")
        ]

        for instruction, output in examples_data:
            examples.append({
                "instruction": instruction,
                "input": name,
                "output": output
            })

    return examples

def create_flavor_analysis_examples(compound_df, num_examples=3000):
    """Create detailed flavor analysis examples"""
    examples = []
    
    if compound_df is None:
        return examples
    
    # Sample compounds with flavor data
    compounds_with_flavor = compound_df[compound_df['primary_flavor'].notna()].sample(
        min(num_examples, len(compound_df)), random_state=42
    )
    
    for _, row in compounds_with_flavor.iterrows():
        compound = row['compound']
        primary_flavor = row['primary_flavor']
        
        # Create flavor descriptions
        flavor_descriptions = {
            'sweet': f"{compound} contributes a sweet, sugary note that enhances desserts and balances acidity in savory dishes.",
            'bitter': f"{compound} provides a bitter complexity that adds depth to beverages and helps balance overly sweet flavors.",
            'aromatic': f"{compound} delivers aromatic compounds that create the characteristic scent and flavor profile of many herbs and spices.",
            'acid': f"{compound} adds bright, acidic notes that can cut through richness and add freshness to dishes.",
            'umami': f"{compound} contributes savory, umami depth that enhances the overall flavor profile and creates satisfying taste experiences.",
            'heat': f"{compound} provides spicy heat that can range from mild warmth to intense pungency, depending on concentration."
        }
        
        instructions = [
            f"Analyze the flavor profile of {compound}",
            f"What taste characteristics does {compound} contribute?",
            f"Describe the flavor impact of {compound}",
            f"How does {compound} affect taste perception?",
            f"What makes {compound} unique in flavor chemistry?"
        ]
        
        output = flavor_descriptions.get(primary_flavor.lower(), 
            f"{compound} is a flavor compound that contributes distinctive taste characteristics to food and beverages.")
        
        for instruction in instructions:
            examples.append({
                "instruction": instruction,
                "input": compound,
                "output": output
            })
    
    return examples

def create_substitution_examples(nodes_df, embeddings, num_examples=2000):
    """Create ingredient substitution examples"""
    examples = []
    
    if nodes_df is None or not embeddings:
        return examples
    
    # Group ingredients by category
    categories = defaultdict(list)
    for _, row in nodes_df.iterrows():
        if row['node_type'] == 'ingredient':
            categories[row.get('category', 'Unknown')].append(row['cleaned_name'])
    
    # Create substitution pairs
    substitution_pairs = [
        ('garlic', 'onion', 'Both provide aromatic base flavors, though garlic is more pungent while onion is sweeter'),
        ('butter', 'olive oil', 'Both add richness, but butter provides creaminess while olive oil adds fruitiness'),
        ('lemon', 'lime', 'Both add citrus acidity, with lemon being more tart and lime more floral'),
        ('basil', 'oregano', 'Both are Mediterranean herbs, with basil being sweeter and oregano more earthy'),
        ('tomato', 'red bell pepper', 'Both add sweetness and color, with tomato being more acidic'),
        ('chicken', 'turkey', 'Both are lean poultry with similar texture, though turkey is slightly gamier'),
        ('rice', 'quinoa', 'Both are grains, with quinoa having more protein and a nuttier flavor'),
        ('milk', 'coconut milk', 'Both provide creaminess, with coconut milk adding tropical sweetness'),
        ('sugar', 'honey', 'Both add sweetness, with honey providing floral notes and natural complexity'),
        ('salt', 'soy sauce', 'Both add saltiness, with soy sauce providing umami depth')
    ]
    
    for ingredient1, ingredient2, explanation in substitution_pairs:
        instructions = [
            f"What can I substitute for {ingredient1}?",
            f"How can I replace {ingredient1} in a recipe?",
            f"What ingredient works like {ingredient1}?",
            f"Find a substitute for {ingredient1}",
            f"What's a good alternative to {ingredient1}?"
        ]
        
        output = f"{ingredient2} can substitute for {ingredient1}. {explanation}"
        
        for instruction in instructions:
            examples.append({
                "instruction": instruction,
                "input": ingredient1,
                "output": output
            })
    
    return examples

def create_recipe_analysis_examples(nodes_df, num_examples=1500):
    """Create recipe analysis and pairing examples"""
    examples = []
    
    if nodes_df is None:
        return examples
    
    # Common ingredient combinations
    combinations = [
        ('tomato', 'basil', 'Classic Italian pairing that balances acidity with herbal sweetness'),
        ('chocolate', 'vanilla', 'Timeless combination where vanilla enhances chocolate\'s richness'),
        ('lemon', 'garlic', 'Mediterranean duo that creates bright, aromatic base flavors'),
        ('apple', 'cinnamon', 'Warm spice pairing that brings out apple\'s natural sweetness'),
        ('beef', 'red wine', 'Sophisticated pairing where wine tenderizes and adds complexity'),
        ('fish', 'dill', 'Light herb that complements fish without overwhelming delicate flavors'),
        ('pork', 'sage', 'Earthy herb that cuts through pork\'s richness'),
        ('carrot', 'ginger', 'Asian-inspired pairing that balances sweetness with spice'),
        ('strawberry', 'balsamic', 'Unexpected combination that enhances strawberry\'s natural sweetness'),
        ('lamb', 'rosemary', 'Robust herb that stands up to lamb\'s strong flavor')
    ]
    
    for ingredient1, ingredient2, explanation in combinations:
        instructions = [
            f"Why do {ingredient1} and {ingredient2} work well together?",
            f"Analyze the pairing of {ingredient1} and {ingredient2}",
            f"What makes {ingredient1} and {ingredient2} a good combination?",
            f"Explain the flavor relationship between {ingredient1} and {ingredient2}",
            f"How do {ingredient1} and {ingredient2} complement each other?"
        ]
        
        output = f"{ingredient1} and {ingredient2} create a harmonious pairing. {explanation}"
        
        for instruction in instructions:
            examples.append({
                "instruction": instruction,
                "input": f"{ingredient1} + {ingredient2}",
                "output": output
            })
    
    return examples

def create_expanded_dataset():
    """Create expanded, high-quality training dataset"""
    print("🚀 Creating expanded FlavorGraph training dataset...")
    
    # Load FlavorGraph data
    embeddings, nodes_df, compound_df = load_flavorgraph_data()
    
    all_examples = []
    
    # Create different types of examples
    print("📝 Creating ingredient knowledge examples...")
    ingredient_examples = create_ingredient_knowledge_examples(nodes_df, embeddings, 5000)
    all_examples.extend(ingredient_examples)
    print(f"  Created {len(ingredient_examples)} examples")
    
    print("🧪 Creating flavor analysis examples...")
    flavor_examples = create_flavor_analysis_examples(compound_df, 3000)
    all_examples.extend(flavor_examples)
    print(f"  Created {len(flavor_examples)} examples")
    
    print("🔄 Creating substitution examples...")
    substitution_examples = create_substitution_examples(nodes_df, embeddings, 2000)
    all_examples.extend(substitution_examples)
    print(f"  Created {len(substitution_examples)} examples")
    
    print("🍽️ Creating recipe analysis examples...")
    recipe_examples = create_recipe_analysis_examples(nodes_df, 1500)
    all_examples.extend(recipe_examples)
    print(f"  Created {len(recipe_examples)} examples")
    
    # Shuffle examples
    random.shuffle(all_examples)
    
    # Save expanded dataset
    output_file = "training_data/expanded_training.jsonl"
    with open(output_file, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"\n✅ Expanded dataset created: {len(all_examples)} examples")
    print(f"📁 Saved to: {output_file}")
    
    # Analyze quality
    analyze_dataset_quality(all_examples)
    
    return all_examples

def analyze_dataset_quality(examples):
    """Analyze the quality of the expanded dataset"""
    print("\n📊 Dataset Quality Analysis:")
    print("=" * 40)
    
    # Check output lengths
    output_lengths = [len(example['output'].split()) for example in examples]
    avg_length = sum(output_lengths) / len(output_lengths)
    
    print(f"Average output length: {avg_length:.1f} words")
    print(f"Min output length: {min(output_lengths)} words")
    print(f"Max output length: {max(output_lengths)} words")
    
    # Check for repetitive patterns
    repetitive_patterns = ['is a chemical compound', 'is a ingredient', 'node_id:']
    pattern_counts = {}
    
    for pattern in repetitive_patterns:
        count = sum(1 for example in examples if pattern in example['output'])
        pattern_counts[pattern] = count
    
    print(f"\nRepetitive pattern analysis:")
    for pattern, count in pattern_counts.items():
        percentage = (count / len(examples)) * 100
        print(f"  {pattern}: {count} ({percentage:.1f}%)")
    
    # Check instruction diversity
    instructions = [example['instruction'] for example in examples]
    unique_instructions = len(set(instructions))
    print(f"\nInstruction diversity: {unique_instructions} unique instructions")
    
    # Estimate quality
    good_examples = sum(1 for example in examples 
                       if len(example['output'].split()) >= 10 and 
                       not any(pattern in example['output'] for pattern in repetitive_patterns))
    
    print(f"\nQuality assessment:")
    print(f"  Good examples: {good_examples} ({good_examples/len(examples)*100:.1f}%)")
    print(f"  Poor examples: {len(examples) - good_examples} ({(len(examples) - good_examples)/len(examples)*100:.1f}%)")

if __name__ == "__main__":
    create_expanded_dataset()
