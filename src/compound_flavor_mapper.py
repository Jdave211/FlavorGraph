#!/usr/bin/env python3
"""
Chemical Compound to Flavor Mapper for FlavorGraph
Maps chemical compounds to fundamental flavor elements inspired by culinary science
"""

import pandas as pd
import numpy as np
import pickle
import networkx as nx
from collections import defaultdict
import json
import os

class CompoundFlavorMapper:
    def __init__(self, nodes_file, edges_file):
        """
        Initialize with graph data files.
        
        Args:
            nodes_file: Path to nodes CSV
            edges_file: Path to edges CSV
        """
        self.nodes_file = nodes_file
        self.edges_file = edges_file
        self.compound_flavor_map = {}
        self.fundamental_flavors = {}
        
        # Define fundamental flavor chemistry based on culinary science principles
        self.flavor_chemistry = {
            'salt': {
                'description': 'Enhances other flavors, provides minerality',
                'compounds': ['sodium', 'chloride', 'nacl', 'salt', 'brine', 'mineral'],
                'molecular_indicators': ['na+', 'cl-', 'sodium_chloride', 'potassium', 'magnesium']
            },
            'fat': {
                'description': 'Carries flavors, provides richness and mouthfeel',
                'compounds': ['lipid', 'fatty_acid', 'triglyceride', 'oil', 'butter', 'cream'],
                'molecular_indicators': ['oleic', 'palmitic', 'stearic', 'linoleic', 'saturated', 'unsaturated']
            },
            'acid': {
                'description': 'Provides brightness, balances richness',
                'compounds': ['citric', 'acetic', 'malic', 'tartaric', 'lactic', 'ascorbic'],
                'molecular_indicators': ['acid', 'vinegar', 'citrus', 'fermented', 'sour', 'ph']
            },
            'heat': {
                'description': 'Provides warmth, pungency, and sensation',
                'compounds': ['capsaicin', 'piperine', 'allicin', 'gingerol', 'eugenol'],
                'molecular_indicators': ['spicy', 'hot', 'pungent', 'warming', 'burning']
            },
            'umami': {
                'description': 'Savory depth, meaty richness',
                'compounds': ['glutamate', 'inosinate', 'guanylate', 'nucleotide'],
                'molecular_indicators': ['msg', 'amino_acid', 'protein', 'savory', 'meaty']
            },
            'sweet': {
                'description': 'Provides pleasure, balances bitterness',
                'compounds': ['sucrose', 'fructose', 'glucose', 'maltose', 'lactose'],
                'molecular_indicators': ['sugar', 'sweet', 'honey', 'syrup', 'caramel']
            },
            'bitter': {
                'description': 'Adds complexity, signals alkaloids',
                'compounds': ['caffeine', 'theobromine', 'quinine', 'tannin', 'alkaloid'],
                'molecular_indicators': ['bitter', 'alkaloid', 'phenolic', 'astringent']
            },
            'aromatic': {
                'description': 'Volatile compounds that create aroma and flavor',
                'compounds': ['terpene', 'ester', 'aldehyde', 'ketone', 'alcohol'],
                'molecular_indicators': ['volatile', 'aromatic', 'fragrant', 'perfume', 'essential']
            }
        }
    
    def load_compound_data(self):
        """Load compound data from the graph."""
        print("Loading compound data from graph...")
        
        # Load nodes
        nodes_df = pd.read_csv(self.nodes_file)
        
        # Filter compounds
        compounds = nodes_df[nodes_df['node_type'] == 'compound'].copy()
        
        # Use cleaned_name if available, fallback to name
        name_col = 'cleaned_name' if 'cleaned_name' in compounds.columns else 'name'
        
        print(f"Found {len(compounds)} chemical compounds")
        return compounds, name_col
    
    def map_compounds_to_flavors(self):
        """Map chemical compounds to fundamental flavor categories."""
        print("Mapping compounds to fundamental flavors...")
        
        compounds, name_col = self.load_compound_data()
        
        # Initialize compound flavor mappings
        for _, compound in compounds.iterrows():
            compound_name = str(compound[name_col]).lower()
            compound_id = compound['node_id']
            
            # Score compound against each flavor category
            flavor_scores = {}
            for flavor_type, flavor_data in self.flavor_chemistry.items():
                score = 0
                
                # Check compound keywords
                for keyword in flavor_data['compounds']:
                    if keyword in compound_name:
                        score += 2  # Strong match
                
                # Check molecular indicators
                for indicator in flavor_data['molecular_indicators']:
                    if indicator in compound_name:
                        score += 1  # Moderate match
                
                flavor_scores[flavor_type] = score
            
            # Normalize scores
            total_score = sum(flavor_scores.values())
            if total_score > 0:
                for flavor in flavor_scores:
                    flavor_scores[flavor] = flavor_scores[flavor] / total_score
            else:
                # Default neutral if no matches
                for flavor in flavor_scores:
                    flavor_scores[flavor] = 1.0 / len(flavor_scores)
            
            self.compound_flavor_map[compound_name] = {
                'node_id': compound_id,
                'original_name': compound['name'],
                'flavor_profile': flavor_scores,
                'primary_flavor': max(flavor_scores, key=flavor_scores.get),
                'flavor_strength': max(flavor_scores.values())
            }
        
        print(f"Mapped {len(self.compound_flavor_map)} compounds to flavor profiles")
        return self.compound_flavor_map
    
    def analyze_compound_flavors(self):
        """Analyze the compound-flavor mappings."""
        if not self.compound_flavor_map:
            print("No compound mappings created yet. Run map_compounds_to_flavors() first.")
            return
        
        print("\n" + "="*60)
        print("COMPOUND FLAVOR MAPPING ANALYSIS")
        print("="*60)
        
        # Flavor distribution
        flavor_distribution = defaultdict(list)
        for compound, data in self.compound_flavor_map.items():
            primary_flavor = data['primary_flavor']
            strength = data['flavor_strength']
            flavor_distribution[primary_flavor].append((compound, strength))
        
        print("Flavor category distribution:")
        for flavor in sorted(flavor_distribution.keys()):
            count = len(flavor_distribution[flavor])
            avg_strength = np.mean([strength for _, strength in flavor_distribution[flavor]])
            print(f"  {flavor}: {count} compounds (avg strength: {avg_strength:.3f})")
        
        # Show top compounds for each flavor
        print("\nTop compounds by flavor category:")
        for flavor in sorted(flavor_distribution.keys()):
            if flavor_distribution[flavor]:
                top_compounds = sorted(flavor_distribution[flavor], 
                                     key=lambda x: x[1], reverse=True)[:3]
                compound_list = [f"{comp} ({strength:.2f})" 
                               for comp, strength in top_compounds]
                print(f"  {flavor}: {', '.join(compound_list)}")
        
        # Flavor chemistry insights
        print(f"\nFlavor chemistry insights:")
        for flavor_type, flavor_data in self.flavor_chemistry.items():
            compound_count = len(flavor_distribution[flavor_type])
            print(f"  {flavor_type}: {flavor_data['description']} ({compound_count} compounds)")
    
    def create_ingredient_flavor_network(self):
        """Create network connections between ingredients and their flavor compounds."""
        print("Creating ingredient-flavor compound network...")
        
        # Load edges to find ingredient-compound connections
        edges_df = pd.read_csv(self.edges_file)
        
        # Filter for ingredient-compound edges (assuming edge_type exists)
        if 'edge_type' in edges_df.columns:
            ingredient_compound_edges = edges_df[
                (edges_df['edge_type'] != 'ingr-ingr')
            ]
        else:
            # If no edge_type, use all edges
            ingredient_compound_edges = edges_df
        
        # Load nodes for mapping
        nodes_df = pd.read_csv(self.nodes_file)
        node_types = dict(zip(nodes_df['node_id'], nodes_df['node_type']))
        name_col = 'cleaned_name' if 'cleaned_name' in nodes_df.columns else 'name'
        node_names = dict(zip(nodes_df['node_id'], nodes_df[name_col]))
        
        # Build ingredient-flavor profiles
        ingredient_flavors = defaultdict(lambda: defaultdict(float))
        
        for _, edge in ingredient_compound_edges.iterrows():
            node1, node2 = edge['id_1'], edge['id_2']
            weight = edge.get('score', 1.0)
            
            # Determine which is ingredient and which is compound
            type1 = node_types.get(node1, 'unknown')
            type2 = node_types.get(node2, 'unknown')
            
            ingredient_id = None
            compound_id = None
            
            if type1 == 'ingredient' and type2 == 'compound':
                ingredient_id, compound_id = node1, node2
            elif type1 == 'compound' and type2 == 'ingredient':
                ingredient_id, compound_id = node2, node1
            
            if ingredient_id and compound_id:
                ingredient_name = node_names.get(ingredient_id, f"ingredient_{ingredient_id}")
                compound_name = str(node_names.get(compound_id, f"compound_{compound_id}")).lower()
                
                # If compound has flavor mapping, add to ingredient profile
                if compound_name in self.compound_flavor_map:
                    compound_flavors = self.compound_flavor_map[compound_name]['flavor_profile']
                    
                    for flavor, score in compound_flavors.items():
                        ingredient_flavors[ingredient_name][flavor] += score * weight
        
        # Normalize ingredient flavor profiles
        for ingredient in ingredient_flavors:
            total_score = sum(ingredient_flavors[ingredient].values())
            if total_score > 0:
                for flavor in ingredient_flavors[ingredient]:
                    ingredient_flavors[ingredient][flavor] /= total_score
        
        self.ingredient_flavors = dict(ingredient_flavors)
        print(f"Created flavor profiles for {len(self.ingredient_flavors)} ingredients")
        return self.ingredient_flavors
    
    def save_flavor_mappings(self, output_dir='./input/compound_flavors/'):
        """Save compound-flavor mappings."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save compound flavor mappings
        compound_mappings_file = os.path.join(output_dir, 'compound_flavor_mappings.json')
        with open(compound_mappings_file, 'w') as f:
            json.dump(self.compound_flavor_map, f, indent=2)
        
        # Save as CSV
        compound_data = []
        for compound, data in self.compound_flavor_map.items():
            row = {
                'compound': compound,
                'node_id': data['node_id'],
                'original_name': data['original_name'],
                'primary_flavor': data['primary_flavor'],
                'flavor_strength': data['flavor_strength']
            }
            # Add individual flavor scores
            row.update(data['flavor_profile'])
            compound_data.append(row)
        
        compound_df = pd.DataFrame(compound_data)
        compound_csv = os.path.join(output_dir, 'compound_flavor_mappings.csv')
        compound_df.to_csv(compound_csv, index=False)
        
        # Save ingredient flavor profiles if created
        if hasattr(self, 'ingredient_flavors'):
            ingredient_flavors_file = os.path.join(output_dir, 'ingredient_flavor_profiles.json')
            with open(ingredient_flavors_file, 'w') as f:
                json.dump(self.ingredient_flavors, f, indent=2)
            
            # Save ingredient flavors as CSV
            ingredient_data = []
            for ingredient, flavors in self.ingredient_flavors.items():
                row = {'ingredient': ingredient}
                row.update(flavors)
                ingredient_data.append(row)
            
            ingredient_df = pd.DataFrame(ingredient_data)
            ingredient_csv = os.path.join(output_dir, 'ingredient_flavor_profiles.csv')
            ingredient_df.to_csv(ingredient_csv, index=False)
            
            print(f"Ingredient flavor profiles saved: {ingredient_csv}")
        
        # Save flavor chemistry reference
        chemistry_file = os.path.join(output_dir, 'flavor_chemistry_reference.json')
        with open(chemistry_file, 'w') as f:
            json.dump(self.flavor_chemistry, f, indent=2)
        
        print(f"\nCompound flavor data saved to {output_dir}:")
        print(f"- Compound mappings: {compound_csv}")
        print(f"- Flavor chemistry reference: {chemistry_file}")
        
        return output_dir
    
    def run_mapping(self):
        """Run the complete compound-flavor mapping pipeline."""
        print("Starting compound-flavor mapping...")
        print("="*60)
        
        # Map compounds to flavors
        self.map_compounds_to_flavors()
        
        # Analyze mappings
        self.analyze_compound_flavors()
        
        # Create ingredient-flavor network
        self.create_ingredient_flavor_network()
        
        # Save results
        output_dir = self.save_flavor_mappings()
        
        print("\n" + "="*60)
        print("COMPOUND-FLAVOR MAPPING COMPLETED!")
        print("="*60)
        
        return output_dir

def main():
    """Main function to run compound-flavor mapping."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Map chemical compounds to fundamental flavors')
    parser.add_argument('--nodes', default='./input/cleaned/nodes_cleaned_basic.csv', 
                       help='Path to nodes CSV file')
    parser.add_argument('--edges', default='./input/edges_191120.csv', 
                       help='Path to edges CSV file')
    
    args = parser.parse_args()
    
    # Initialize mapper
    mapper = CompoundFlavorMapper(args.nodes, args.edges)
    
    # Run mapping
    mapper.run_mapping()

if __name__ == "__main__":
    main()
