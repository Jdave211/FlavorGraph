#!/usr/bin/env python3
"""
Flavor Profile Integrator for FlavorGraph
Maps chemical compounds to taste profiles and enhances embeddings
"""

import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
import os
import glob

class FlavorProfileIntegrator:
    def __init__(self, nodes_file, embeddings_file=None):
        """
        Initialize with nodes file and optional embeddings file.
        
        Args:
            nodes_file: Path to nodes CSV
            embeddings_file: Path to embeddings pickle (optional - will auto-detect)
        """
        self.nodes_file = nodes_file
        self.embeddings_file = embeddings_file
        self.flavor_profiles = {}
        self.enhanced_embeddings = {}
        
        # Auto-detect embeddings file if not provided
        if self.embeddings_file is None:
            embedding_files = glob.glob('./output/*embedding*.pickle')
            if embedding_files:
                self.embeddings_file = max(embedding_files, key=os.path.getmtime)
                print(f"Auto-detected embeddings file: {self.embeddings_file}")
            else:
                print("Warning: No embeddings file found. Only flavor profiles will be created.")
        
    def create_flavor_mappings(self):
        """Create flavor profile mappings for ingredients."""
        print("Creating flavor profile mappings...")
        
        # Load nodes
        nodes_df = pd.read_csv(self.nodes_file)
        ingredients = nodes_df[nodes_df['node_type'] == 'ingredient']
        
        # Use cleaned_name if available, fallback to name
        name_col = 'cleaned_name' if 'cleaned_name' in ingredients.columns else 'name'
        
        # Define flavor categories with keywords
        flavor_categories = {
            'sweet': ['sugar', 'honey', 'maple', 'vanilla', 'chocolate', 'caramel', 'fruit', 
                     'berry', 'apple', 'orange', 'grape', 'sweet', 'syrup', 'molasses'],
            'salty': ['salt', 'soy', 'anchovy', 'olive', 'cheese', 'bacon', 'ham', 
                     'prosciutto', 'salted', 'brine', 'pickle'],
            'sour': ['lemon', 'lime', 'vinegar', 'yogurt', 'buttermilk', 'tamarind', 
                    'sour', 'citrus', 'cranberry', 'pomegranate'],
            'bitter': ['coffee', 'dark_chocolate', 'grapefruit', 'arugula', 'endive', 
                      'bitter', 'cocoa', 'kale', 'dandelion'],
            'umami': ['mushroom', 'tomato', 'parmesan', 'miso', 'soy_sauce', 'anchovy', 
                     'beef', 'pork', 'chicken', 'fish', 'seaweed', 'truffle'],
            'spicy': ['pepper', 'chili', 'ginger', 'garlic', 'onion', 'mustard', 
                     'hot', 'spicy', 'cayenne', 'paprika', 'jalapeno'],
            'herbal': ['basil', 'oregano', 'thyme', 'rosemary', 'sage', 'parsley', 
                      'mint', 'cilantro', 'dill', 'tarragon'],
            'earthy': ['beet', 'mushroom', 'truffle', 'walnut', 'coffee', 'carrot', 
                      'potato', 'turnip', 'radish', 'soil']
        }
        
        # Map ingredients to flavor profiles
        for _, row in ingredients.iterrows():
            ingredient_name = str(row[name_col]).lower()
            flavor_scores = {}
            
            for flavor, keywords in flavor_categories.items():
                score = 0
                for keyword in keywords:
                    if keyword in ingredient_name:
                        score += 1
                # Add partial matches for compound ingredients
                if any(keyword in ingredient_name for keyword in keywords):
                    score += 0.5
                flavor_scores[flavor] = score
            
            # Normalize scores
            total_score = sum(flavor_scores.values())
            if total_score > 0:
                for flavor in flavor_scores:
                    flavor_scores[flavor] = flavor_scores[flavor] / total_score
            else:
                # Default neutral profile if no matches
                for flavor in flavor_scores:
                    flavor_scores[flavor] = 0.125  # Equal distribution across 8 flavors
            
            self.flavor_profiles[row[name_col]] = flavor_scores
        
        print(f"Created flavor profiles for {len(self.flavor_profiles)} ingredients")
        return self.flavor_profiles
    
    def enhance_embeddings_with_flavor(self):
        """Enhance embeddings with flavor profile information."""
        if self.embeddings_file is None:
            print("No embeddings file available. Skipping embedding enhancement.")
            return None
            
        print("Enhancing embeddings with flavor profiles...")
        
        # Load original embeddings
        try:
            with open(self.embeddings_file, 'rb') as f:
                original_embeddings = pickle.load(f)
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return None
        
        # Create enhanced embeddings
        for ingredient, embedding in original_embeddings.items():
            if ingredient in self.flavor_profiles:
                # Get flavor profile
                flavor_profile = self.flavor_profiles[ingredient]
                
                # Convert flavor profile to vector
                flavor_vector = np.array([
                    flavor_profile.get('sweet', 0),
                    flavor_profile.get('salty', 0),
                    flavor_profile.get('sour', 0),
                    flavor_profile.get('bitter', 0),
                    flavor_profile.get('umami', 0),
                    flavor_profile.get('spicy', 0),
                    flavor_profile.get('herbal', 0),
                    flavor_profile.get('earthy', 0)
                ])
                
                # Concatenate original embedding with flavor vector
                enhanced_embedding = np.concatenate([embedding, flavor_vector])
                self.enhanced_embeddings[ingredient] = enhanced_embedding
            else:
                # Keep original embedding if no flavor profile
                zero_flavor = np.zeros(8)  # 8 flavor dimensions
                enhanced_embedding = np.concatenate([embedding, zero_flavor])
                self.enhanced_embeddings[ingredient] = enhanced_embedding
        
        print(f"Enhanced {len(self.enhanced_embeddings)} embeddings")
        print(f"New embedding dimension: {len(next(iter(self.enhanced_embeddings.values())))}")
        return self.enhanced_embeddings
    
    def analyze_flavor_profiles(self):
        """Analyze the created flavor profiles."""
        if not self.flavor_profiles:
            print("No flavor profiles created yet. Run create_flavor_mappings() first.")
            return
        
        print("\n" + "="*50)
        print("FLAVOR PROFILE ANALYSIS")
        print("="*50)
        
        # Flavor distribution analysis
        flavor_stats = defaultdict(list)
        for ingredient, profile in self.flavor_profiles.items():
            for flavor, score in profile.items():
                if score > 0:
                    flavor_stats[flavor].append((ingredient, score))
        
        print(f"Flavor category distribution:")
        for flavor in sorted(flavor_stats.keys()):
            ingredients_with_flavor = len(flavor_stats[flavor])
            avg_score = np.mean([score for _, score in flavor_stats[flavor]])
            print(f"  {flavor}: {ingredients_with_flavor} ingredients (avg score: {avg_score:.3f})")
        
        # Show top ingredients for each flavor
        print(f"\nTop ingredients by flavor category:")
        for flavor in sorted(flavor_stats.keys()):
            top_ingredients = sorted(flavor_stats[flavor], key=lambda x: x[1], reverse=True)[:3]
            ingredient_list = [f"{ing} ({score:.2f})" for ing, score in top_ingredients]
            print(f"  {flavor}: {', '.join(ingredient_list)}")
    
    def save_flavor_data(self, output_dir='./input/flavor_profiles/'):
        """Save flavor profile data."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save flavor profiles
        flavor_profiles_file = os.path.join(output_dir, 'flavor_profiles.pickle')
        with open(flavor_profiles_file, 'wb') as f:
            pickle.dump(self.flavor_profiles, f)
        
        # Save as CSV for easy analysis
        flavor_data = []
        for ingredient, profile in self.flavor_profiles.items():
            row = {'ingredient': ingredient}
            row.update(profile)
            flavor_data.append(row)
        
        flavor_df = pd.DataFrame(flavor_data)
        flavor_csv = os.path.join(output_dir, 'flavor_profiles.csv')
        flavor_df.to_csv(flavor_csv, index=False)
        
        # Save enhanced embeddings if available
        if self.enhanced_embeddings:
            enhanced_embeddings_file = os.path.join(output_dir, 'enhanced_embeddings.pickle')
            with open(enhanced_embeddings_file, 'wb') as f:
                pickle.dump(self.enhanced_embeddings, f)
            print(f"Enhanced embeddings saved: {enhanced_embeddings_file}")
        
        print(f"\nFlavor data saved to {output_dir}:")
        print(f"- Flavor profiles: {flavor_profiles_file}")
        print(f"- Flavor CSV: {flavor_csv}")
        
        return output_dir
    
    def run_integration(self):
        """Run the complete flavor profile integration pipeline."""
        print("Starting flavor profile integration...")
        print("="*60)
        
        # Create flavor mappings
        self.create_flavor_mappings()
        
        # Enhance embeddings with flavor
        self.enhance_embeddings_with_flavor()
        
        # Analyze results
        self.analyze_flavor_profiles()
        
        # Save data
        output_dir = self.save_flavor_data()
        
        print("\n" + "="*60)
        print("FLAVOR PROFILE INTEGRATION COMPLETED!")
        print("="*60)
        
        return output_dir

def main():
    """Main function to run flavor profile integration."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrate flavor profiles into FlavorGraph')
    parser.add_argument('--nodes', default='./input/cleaned/nodes_cleaned_basic.csv', 
                       help='Path to nodes CSV file')
    parser.add_argument('--embeddings', default=None, 
                       help='Path to embeddings pickle (auto-detect if not specified)')
    
    args = parser.parse_args()
    
    # Initialize integrator
    integrator = FlavorProfileIntegrator(args.nodes, args.embeddings)
    
    # Run integration
    integrator.run_integration()

if __name__ == "__main__":
    main()
