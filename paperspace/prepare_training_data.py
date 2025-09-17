#!/usr/bin/env python3
"""
FlavorGraph AI Training Data Preparation Script
Converts FlavorGraph embeddings and relationships into instruction-following training data
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import random
from sklearn.metrics.pairwise import cosine_similarity

class FlavorGraphDataPreparator:
    def __init__(self, base_path: str = "/Users/davejaga/Desktop/Startups/FlavorGraph"):
        self.base_path = Path(base_path)
        self.output_dir = self.base_path / "paperspace" / "training_data"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all data sources
        self.nodes_df = None
        self.categories_df = None
        self.compound_flavors_df = None
        self.embeddings = None
        self.flavor_analysis = None
        
        print("🚀 Initializing FlavorGraph Data Preparator...")
        
    def load_data(self):
        """Load all FlavorGraph data sources"""
        print("📊 Loading FlavorGraph datasets...")
        
        # Load nodes
        nodes_path = self.base_path / "input" / "cleaned" / "nodes_cleaned_basic.csv"
        self.nodes_df = pd.read_csv(nodes_path)
        print(f"   ✅ Loaded {len(self.nodes_df)} nodes")
        
        # Load categories
        categories_path = self.base_path / "input" / "dict_ingr2cate - Top300+FDB400+HyperFoods104=616.csv"
        self.categories_df = pd.read_csv(categories_path)
        print(f"   ✅ Loaded {len(self.categories_df)} ingredient categories")
        
        # Load compound flavors
        compound_path = self.base_path / "input" / "compound_flavors" / "compound_flavor_mappings.csv"
        self.compound_flavors_df = pd.read_csv(compound_path)
        print(f"   ✅ Loaded {len(self.compound_flavors_df)} compound flavor profiles")
        
        # Load embeddings (find the latest one)
        embedding_files = list((self.base_path / "output").glob("*embedding*.pickle"))
        if embedding_files:
            latest_embedding = max(embedding_files, key=os.path.getmtime)
            with open(latest_embedding, 'rb') as f:
                self.embeddings = pickle.load(f)
            print(f"   ✅ Loaded embeddings from {latest_embedding.name}")
            print(f"   📈 Embedding dimensions: {len(self.embeddings)} nodes x {len(next(iter(self.embeddings.values())))} dims")
        
        # Load flavor analysis if available
        flavor_files = list((self.base_path / "output").glob("*flavor_analysis.json"))
        if flavor_files:
            latest_flavor = max(flavor_files, key=os.path.getmtime)
            with open(latest_flavor, 'r') as f:
                self.flavor_analysis = json.load(f)
            print(f"   ✅ Loaded flavor analysis from {latest_flavor.name}")
    
    def create_ingredient_knowledge_base(self) -> List[Dict]:
        """Create ingredient knowledge base training examples"""
        print("🧠 Creating ingredient knowledge base...")
        
        examples = []
        
        # Create node_id to name mapping
        id_to_name = dict(zip(self.nodes_df['node_id'], self.nodes_df['cleaned_name'].fillna(self.nodes_df['name'])))
        
        # Create name to category mapping
        name_to_category = dict(zip(self.categories_df['ingredient'], self.categories_df['category']))
        
        # Focus on ingredients (not compounds) that have embeddings or are in categories
        ingredient_nodes = self.nodes_df[self.nodes_df['node_type'] == 'ingredient'].copy()
        
        for _, row in ingredient_nodes.iterrows():
            node_id = row['node_id']
            name = row['cleaned_name'] if pd.notna(row['cleaned_name']) else row['name']
            node_type = row['node_type']
            is_hub = row['is_hub']
            
            # Include if it has embeddings OR is in our category mapping
            has_embedding = node_id in self.embeddings
            in_categories = name in name_to_category
            
            if not (has_embedding or in_categories):
                continue
                
            # Build comprehensive description
            description_parts = []
            
            # Basic info
            description_parts.append(f"{name.replace('_', ' ').title()} (node_id: {node_id}) is a {node_type}")
            
            # Category info
            if name in name_to_category:
                category = name_to_category[name]
                description_parts.append(f"classified as {category}")
            
            # Hub status
            if is_hub == 'hub':
                description_parts.append("and is a hub ingredient (commonly used across many recipes)")
            
            # Flavor profile if available
            if self.flavor_analysis and str(node_id) in self.flavor_analysis:
                flavor_data = self.flavor_analysis[str(node_id)]
                if 'flavor_profile' in flavor_data:
                    top_flavors = []
                    for flavor, score in flavor_data['flavor_profile'].items():
                        if score > 0.3:  # Only significant flavors
                            top_flavors.append(f"{flavor}={score:.2f}")
                    if top_flavors:
                        description_parts.append(f"Flavor profile: {', '.join(top_flavors)}")
            
            # Find similar ingredients (only if has embeddings)
            if has_embedding and node_id in self.embeddings:
                similar_ingredients = self.find_most_similar(node_id, top_k=3)
                if similar_ingredients:
                    similar_names = [id_to_name.get(sim_id, f"node_{sim_id}") for sim_id, _ in similar_ingredients]
                    description_parts.append(f"Similar ingredients include: {', '.join(similar_names)}")
            
            description = ". ".join(description_parts) + "."
            
            # Create training examples
            examples.extend([
                {
                    "instruction": f"What can you tell me about {name.replace('_', ' ')}?",
                    "input": name,
                    "output": description
                },
                {
                    "instruction": "Describe this ingredient and its properties",
                    "input": name.replace('_', ' '),
                    "output": description
                },
                {
                    "instruction": f"What type of ingredient is {name.replace('_', ' ')}?",
                    "input": name,
                    "output": f"{name.replace('_', ' ').title()} is {description}"
                }
            ])
        
        print(f"   ✅ Created {len(examples)} ingredient knowledge examples")
        return examples
    
    def create_flavor_analysis_examples(self) -> List[Dict]:
        """Create flavor analysis training examples"""
        print("🌶️ Creating flavor analysis examples...")
        
        examples = []
        
        for _, row in self.compound_flavors_df.iterrows():
            compound = row['compound']
            node_id = row['node_id']
            
            # Build flavor profile description
            flavor_scores = {}
            for col in ['salt', 'fat', 'acid', 'heat', 'umami', 'sweet', 'bitter', 'aromatic']:
                if col in row and pd.notna(row[col]):
                    flavor_scores[col] = float(row[col])
            
            # Find dominant flavors
            dominant_flavors = [(k, v) for k, v in flavor_scores.items() if v > 0.5]
            dominant_flavors.sort(key=lambda x: x[1], reverse=True)
            
            # Create description
            description_parts = [f"{compound.replace('_', ' ').title()} (node_id: {node_id}) is a chemical compound"]
            
            if dominant_flavors:
                flavor_desc = []
                for flavor, score in dominant_flavors[:3]:  # Top 3 flavors
                    flavor_desc.append(f"{flavor}={score:.2f}")
                description_parts.append(f"with primary flavor characteristics: {', '.join(flavor_desc)}")
            
            # Add chemical context
            if 'acid' in compound.lower():
                description_parts.append("This acid compound contributes sourness and preservation properties")
            elif 'alcohol' in compound.lower() or 'ethanol' in compound.lower():
                description_parts.append("This alcohol compound affects texture and carries other flavors")
            elif 'capsaicin' in compound.lower():
                description_parts.append("This capsaicinoid creates the burning sensation in spicy foods")
            
            description = ". ".join(description_parts) + "."
            
            examples.extend([
                {
                    "instruction": f"Analyze the flavor profile of {compound}",
                    "input": compound,
                    "output": description
                },
                {
                    "instruction": "What are the chemical flavor characteristics of this compound?",
                    "input": compound.replace('_', ' '),
                    "output": description
                },
                {
                    "instruction": f"Explain the taste properties of {compound}",
                    "input": compound,
                    "output": description
                }
            ])
        
        print(f"   ✅ Created {len(examples)} flavor analysis examples")
        return examples
    
    def create_substitution_examples(self) -> List[Dict]:
        """Create ingredient substitution training examples"""
        print("🔄 Creating substitution examples...")
        
        examples = []
        
        # Create category groups
        category_groups = {}
        for _, row in self.categories_df.iterrows():
            category = row['category']
            ingredient = row['ingredient']
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(ingredient)
        
        # Create node_id to name mapping
        name_to_id = {}
        id_to_name = {}
        for _, row in self.nodes_df.iterrows():
            name = row['cleaned_name'] if pd.notna(row['cleaned_name']) else row['name']
            node_id = row['node_id']
            name_to_id[name] = node_id
            id_to_name[node_id] = name
        
        # Create name to category mapping (for reference in the loop)
        name_to_category = dict(zip(self.categories_df['ingredient'], self.categories_df['category']))
        
        # Generate substitution pairs within categories
        for category, ingredients in category_groups.items():
            if len(ingredients) < 2:
                continue
                
            # Find ingredients with embeddings OR at least in our dataset
            available_ingredients = []
            for ing in ingredients:
                if ing in name_to_id:
                    node_id = name_to_id[ing]
                    # Include if has embeddings OR is a known ingredient
                    if node_id in self.embeddings or ing in name_to_category:
                        available_ingredients.append(ing)
            
            if len(available_ingredients) < 2:
                continue
            
            # Create substitution examples
            for i, base_ingredient in enumerate(available_ingredients[:10]):  # Limit to prevent explosion
                base_id = name_to_id[base_ingredient]
                
                # Find most similar in same category
                similar_in_category = []
                for other_ing in available_ingredients:
                    if other_ing != base_ingredient:
                        other_id = name_to_id[other_ing]
                        
                        # Calculate similarity if both have embeddings
                        if base_id in self.embeddings and other_id in self.embeddings:
                            similarity = cosine_similarity(
                                [self.embeddings[base_id]], 
                                [self.embeddings[other_id]]
                            )[0][0]
                        else:
                            # Use a default similarity for same-category items
                            similarity = 0.6  # Moderate similarity within category
                        
                        similar_in_category.append((other_ing, similarity))
                
                similar_in_category.sort(key=lambda x: x[1], reverse=True)
                
                if similar_in_category:
                    best_substitute = similar_in_category[0]
                    substitute_name, similarity_score = best_substitute
                    
                    if similarity_score > 0.5:  # Only if reasonably similar
                        explanation = f"Based on category ({category}) and functional similarity, "
                        explanation += f"{substitute_name.replace('_', ' ')} can substitute for {base_ingredient.replace('_', ' ')}. "
                        explanation += f"Both ingredients share similar properties within the {category} category"
                        
                        if similarity_score < 1.0:  # Add similarity score if calculated
                            explanation += f" (similarity: {similarity_score:.3f})"
                        explanation += "."
                    
                        examples.extend([
                            {
                                "instruction": f"What can I substitute for {base_ingredient.replace('_', ' ')}?",
                                "input": base_ingredient,
                                "output": explanation
                            },
                            {
                                "instruction": f"Find a replacement for {base_ingredient.replace('_', ' ')} in cooking",
                                "input": base_ingredient,
                                "output": explanation
                            }
                        ])
        
        print(f"   ✅ Created {len(examples)} substitution examples")
        return examples
    
    def create_recipe_analysis_examples(self) -> List[Dict]:
        """Create recipe analysis training examples"""
        print("📝 Creating recipe analysis examples...")
        
        examples = []
        
        # Load edges to understand co-occurrence
        edges_path = self.base_path / "input" / "edges_191120.csv"
        if edges_path.exists():
            edges_df = pd.read_csv(edges_path)
            
            # Create node_id to name mapping
            id_to_name = dict(zip(self.nodes_df['node_id'], self.nodes_df['cleaned_name'].fillna(self.nodes_df['name'])))
            name_to_category = dict(zip(self.categories_df['ingredient'], self.categories_df['category']))
            
            # Group high-scoring ingredient pairs
            high_score_pairs = edges_df[
                (edges_df['edge_type'] == 'ingr-ingr') & 
                (edges_df['score'] > 0.3)
            ].head(100)  # Limit for performance
            
            for _, row in high_score_pairs.iterrows():
                id1, id2, score = row['id_1'], row['id_2'], row['score']
                
                if id1 in id_to_name and id2 in id_to_name:
                    name1 = id_to_name[id1]
                    name2 = id_to_name[id2]
                    
                    category1 = name_to_category.get(name1, "Unknown")
                    category2 = name_to_category.get(name2, "Unknown")
                    
                    analysis = f"{name1.replace('_', ' ').title()} and {name2.replace('_', ' ').title()} "
                    analysis += f"frequently appear together in recipes (co-occurrence score: {score:.3f}). "
                    analysis += f"This pairing combines {category1} with {category2}, "
                    analysis += f"creating complementary flavors and textures that work well together."
                    
                    examples.extend([
                        {
                            "instruction": f"Why do {name1.replace('_', ' ')} and {name2.replace('_', ' ')} work well together?",
                            "input": f"{name1}, {name2}",
                            "output": analysis
                        },
                        {
                            "instruction": "Analyze this ingredient combination",
                            "input": f"{name1.replace('_', ' ')} + {name2.replace('_', ' ')}",
                            "output": analysis
                        }
                    ])
        
        print(f"   ✅ Created {len(examples)} recipe analysis examples")
        return examples
    
    def find_most_similar(self, node_id: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """Find most similar ingredients to a given node_id"""
        if node_id not in self.embeddings:
            return []
        
        target_embedding = self.embeddings[node_id]
        similarities = []
        
        for other_id, other_embedding in self.embeddings.items():
            if other_id != node_id:
                similarity = cosine_similarity([target_embedding], [other_embedding])[0][0]
                similarities.append((other_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def save_training_data(self):
        """Save all training data in JSONL format"""
        print("💾 Generating and saving training datasets...")
        
        # Generate all example types
        ingredient_examples = self.create_ingredient_knowledge_base()
        flavor_examples = self.create_flavor_analysis_examples()
        substitution_examples = self.create_substitution_examples()
        recipe_examples = self.create_recipe_analysis_examples()
        
        # Combine all examples
        all_examples = ingredient_examples + flavor_examples + substitution_examples + recipe_examples
        
        # Shuffle for better training
        random.shuffle(all_examples)
        
        # Save individual datasets
        datasets = {
            'ingredient_knowledge.jsonl': ingredient_examples,
            'flavor_analysis.jsonl': flavor_examples,
            'substitution_pairs.jsonl': substitution_examples,
            'recipe_analysis.jsonl': recipe_examples,
            'combined_training.jsonl': all_examples
        }
        
        for filename, examples in datasets.items():
            filepath = self.output_dir / filename
            with open(filepath, 'w') as f:
                for example in examples:
                    f.write(json.dumps(example) + '\n')
            print(f"   ✅ Saved {len(examples)} examples to {filename}")
        
        # Save embeddings reference
        embeddings_path = self.output_dir / "embeddings_reference.pkl"
        with open(embeddings_path, 'wb') as f:
            pickle.dump(self.embeddings, f)
        print(f"   ✅ Saved embeddings reference to embeddings_reference.pkl")
        
        # Save metadata
        metadata = {
            'total_examples': len(all_examples),
            'ingredient_examples': len(ingredient_examples),
            'flavor_examples': len(flavor_examples),
            'substitution_examples': len(substitution_examples),
            'recipe_examples': len(recipe_examples),
            'total_nodes': len(self.nodes_df),
            'total_embeddings': len(self.embeddings),
            'embedding_dimensions': len(next(iter(self.embeddings.values()))),
            'categories': list(self.categories_df['category'].unique()),
            'node_types': list(self.nodes_df['node_type'].unique())
        }
        
        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✅ Saved dataset metadata to dataset_metadata.json")
        
        return metadata

def main():
    print("🎯 FlavorGraph AI Training Data Preparation")
    print("=" * 50)
    
    # Initialize preparator
    preparator = FlavorGraphDataPreparator()
    
    # Load all data
    preparator.load_data()
    
    # Generate and save training data
    metadata = preparator.save_training_data()
    
    print("\n🎉 Training data preparation complete!")
    print(f"📊 Generated {metadata['total_examples']} total training examples")
    print(f"📁 Data saved to: {preparator.output_dir}")
    print("\nDataset breakdown:")
    for key, value in metadata.items():
        if key.endswith('_examples'):
            print(f"   {key.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    main()
