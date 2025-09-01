#!/usr/bin/env python3
"""
Recipe Data Extractor for FlavorGraph
Extracts recipe-like patterns from ingredient co-occurrence graph
"""

import pandas as pd
import numpy as np
import pickle
import networkx as nx
from collections import defaultdict, Counter
import json
import os

class RecipeDataExtractor:
    def __init__(self, nodes_file, edges_file):
        """
        Initialize with graph data files.
        
        Args:
            nodes_file: Path to nodes CSV
            edges_file: Path to edges CSV
        """
        self.nodes_file = nodes_file
        self.edges_file = edges_file
        self.graph = None
        self.recipes = []
        self.ingredient_names = {}
        
    def load_graph_data(self):
        """Load and build the ingredient graph."""
        print("Loading graph data...")
        
        # Load nodes
        nodes_df = pd.read_csv(self.nodes_file)
        # Use cleaned_name if available, fallback to name
        name_col = 'cleaned_name' if 'cleaned_name' in nodes_df.columns else 'name'
        self.ingredient_names = dict(zip(nodes_df['node_id'], nodes_df[name_col]))
        
        # Load edges
        edges_df = pd.read_csv(self.edges_file)
        
        # Build graph
        self.graph = nx.Graph()
        
        # Add nodes
        for _, row in nodes_df.iterrows():
            if row['node_type'] == 'ingredient':
                node_name = row[name_col] if pd.notnull(row.get(name_col)) else row['name']
                self.graph.add_node(row['node_id'], 
                                  name=node_name,
                                  original_name=row['name'],
                                  is_hub=row['is_hub'])
        
        # Add edges (only ingredient-ingredient relationships)
        recipe_edges = edges_df[edges_df['edge_type'] == 'ingr-ingr']
        for _, edge in recipe_edges.iterrows():
            if edge['score'] > 0.05:  # Threshold for meaningful co-occurrence
                self.graph.add_edge(edge['id_1'], edge['id_2'], 
                                  weight=edge['score'],
                                  cooccurrence_score=edge['score'])
        
        print(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
    def extract_recipe_patterns(self, min_ingredients=2, max_ingredients=10, min_score=0.1):
        """
        Extract recipe-like patterns from the graph.
        
        Args:
            min_ingredients: Minimum ingredients per recipe
            max_ingredients: Maximum ingredients per recipe  
            min_score: Minimum co-occurrence score
        """
        print("Extracting recipe patterns...")
        
        recipes = []
        processed_combinations = set()
        
        # Find strongly connected ingredient groups
        for node in self.graph.nodes():
            if self.graph.degree(node) < 2:  # Skip isolated ingredients
                continue
                
            # Get neighbors with high co-occurrence scores
            neighbors = []
            for neighbor in self.graph.neighbors(node):
                edge_data = self.graph[node][neighbor]
                if edge_data['cooccurrence_score'] >= min_score:
                    neighbors.append((neighbor, edge_data['cooccurrence_score']))
            
            if len(neighbors) < min_ingredients - 1:
                continue
            
            # Sort neighbors by co-occurrence score
            neighbors.sort(key=lambda x: x[1], reverse=True)
            
            # Create recipe combinations
            for i in range(min_ingredients - 1, min(len(neighbors), max_ingredients - 1)):
                recipe_ingredients = [node] + [n[0] for n in neighbors[:i+1]]
                recipe_ingredients.sort()  # Sort for consistent hashing
                
                # Create unique identifier
                recipe_id = tuple(recipe_ingredients)
                if recipe_id in processed_combinations:
                    continue
                processed_combinations.add(recipe_id)
                
                # Calculate recipe score (average co-occurrence)
                total_score = 0
                pair_count = 0
                for j in range(len(recipe_ingredients)):
                    for k in range(j + 1, len(recipe_ingredients)):
                        if self.graph.has_edge(recipe_ingredients[j], recipe_ingredients[k]):
                            total_score += self.graph[recipe_ingredients[j]][recipe_ingredients[k]]['cooccurrence_score']
                            pair_count += 1
                
                if pair_count > 0:
                    avg_score = total_score / pair_count
                    
                    recipe = {
                        'recipe_id': f"recipe_{len(recipes)}",
                        'ingredients': recipe_ingredients,
                        'ingredient_names': [self.ingredient_names.get(i, f"unknown_{i}") for i in recipe_ingredients],
                        'cooccurrence_score': avg_score,
                        'ingredient_count': len(recipe_ingredients),
                        'hub_ingredients': [i for i in recipe_ingredients if self.graph.nodes[i]['is_hub'] == 'hub'],
                        'non_hub_ingredients': [i for i in recipe_ingredients if self.graph.nodes[i]['is_hub'] == 'no_hub']
                    }
                    recipes.append(recipe)
        
        # Sort by co-occurrence score
        self.recipes = sorted(recipes, key=lambda x: x['cooccurrence_score'], reverse=True)
        
        print(f"Extracted {len(self.recipes)} recipe patterns")
        return self.recipes
    
    def analyze_recipe_statistics(self):
        """Analyze the extracted recipe patterns."""
        if not self.recipes:
            print("No recipes extracted yet. Run extract_recipe_patterns() first.")
            return
        
        print("\n" + "="*50)
        print("RECIPE EXTRACTION STATISTICS")
        print("="*50)
        
        # Basic stats
        print(f"Total recipes: {len(self.recipes)}")
        print(f"Average ingredients per recipe: {np.mean([r['ingredient_count'] for r in self.recipes]):.1f}")
        print(f"Average co-occurrence score: {np.mean([r['cooccurrence_score'] for r in self.recipes]):.3f}")
        
        # Ingredient frequency analysis
        all_ingredients = []
        for recipe in self.recipes:
            all_ingredients.extend(recipe['ingredient_names'])
        
        ingredient_counts = Counter(all_ingredients)
        print(f"\nMost common ingredients:")
        for ingredient, count in ingredient_counts.most_common(10):
            print(f"  {ingredient}: {count} recipes")
        
        # Recipe size distribution
        size_distribution = Counter([r['ingredient_count'] for r in self.recipes])
        print(f"\nRecipe size distribution:")
        for size in sorted(size_distribution.keys()):
            print(f"  {size} ingredients: {size_distribution[size]} recipes")
        
        # Show sample recipes
        print(f"\nSample high-scoring recipes:")
        for i, recipe in enumerate(self.recipes[:5]):
            print(f"\n{i+1}. Recipe {recipe['recipe_id']} (score: {recipe['cooccurrence_score']:.3f})")
            print(f"   Ingredients: {', '.join(recipe['ingredient_names'])}")
            print(f"   Hub ingredients: {len(recipe['hub_ingredients'])}")
    
    def save_recipe_data(self, output_dir='./input/recipes/'):
        """Save extracted recipe data."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        recipes_json = os.path.join(output_dir, 'extracted_recipes.json')
        with open(recipes_json, 'w') as f:
            json.dump(self.recipes, f, indent=2)
        
        # Save as CSV for easy analysis
        recipes_data = []
        for recipe in self.recipes:
            recipes_data.append({
                'recipe_id': recipe['recipe_id'],
                'ingredients': '|'.join([str(name) for name in recipe['ingredient_names']]),
                'ingredient_count': recipe['ingredient_count'],
                'cooccurrence_score': recipe['cooccurrence_score'],
                'hub_count': len(recipe['hub_ingredients']),
                'non_hub_count': len(recipe['non_hub_ingredients'])
            })
        
        recipes_df = pd.DataFrame(recipes_data)
        recipes_csv = os.path.join(output_dir, 'extracted_recipes.csv')
        recipes_df.to_csv(recipes_csv, index=False)
        
        # Save ingredient mapping
        ingredient_mapping = os.path.join(output_dir, 'ingredient_mapping.pickle')
        with open(ingredient_mapping, 'wb') as f:
            pickle.dump(self.ingredient_names, f)
        
        print(f"\nRecipe data saved to {output_dir}:")
        print(f"- Recipes JSON: {recipes_json}")
        print(f"- Recipes CSV: {recipes_csv}")
        print(f"- Ingredient mapping: {ingredient_mapping}")
        
        return output_dir
    
    def run_extraction(self, min_ingredients=2, max_ingredients=8, min_score=0.1):
        """Run the complete recipe extraction pipeline."""
        print("Starting recipe data extraction...")
        print("="*60)
        
        # Load graph data
        self.load_graph_data()
        
        # Extract recipe patterns
        self.extract_recipe_patterns(min_ingredients, max_ingredients, min_score)
        
        # Analyze results
        self.analyze_recipe_statistics()
        
        # Save data
        output_dir = self.save_recipe_data()
        
        print("\n" + "="*60)
        print("RECIPE EXTRACTION COMPLETED!")
        print("="*60)
        
        return output_dir

def main():
    """Main function to run recipe extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract recipe patterns from FlavorGraph')
    parser.add_argument('--nodes', default='./input/cleaned/nodes_cleaned_basic.csv', 
                       help='Path to nodes CSV file')
    parser.add_argument('--edges', default='./input/edges_191120.csv', 
                       help='Path to edges CSV file')
    parser.add_argument('--min_ingredients', type=int, default=2, 
                       help='Minimum ingredients per recipe')
    parser.add_argument('--max_ingredients', type=int, default=8, 
                       help='Maximum ingredients per recipe')
    parser.add_argument('--min_score', type=float, default=0.1, 
                       help='Minimum co-occurrence score')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = RecipeDataExtractor(args.nodes, args.edges)
    
    # Run extraction
    extractor.run_extraction(
        min_ingredients=args.min_ingredients,
        max_ingredients=args.max_ingredients,
        min_score=args.min_score
    )

if __name__ == "__main__":
    main()
