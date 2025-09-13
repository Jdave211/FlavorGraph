#!/usr/bin/env python3
"""
Ingredient Substitution Search

This tool focuses on finding realistic ingredient SUBSTITUTES rather than co-occurring ingredients.
It uses ingredient categories and flavor profiles to suggest functionally equivalent alternatives.

Key differences from similarity search:
- Filters by ingredient category (e.g., only suggest other fruits for apple)  
- Prioritizes flavor compatibility over co-occurrence patterns
- Considers functional roles in recipes

Usage:
  python3 substitution_search.py --ingredient apple --topn 8
  python3 substitution_search.py --ingredient garlic --category "Spice" --topn 5
"""

import argparse
import os
import pickle
import sys
import csv
import numpy as np
from typing import Dict, List, Tuple, Optional


def load_ingredient_categories(category_file: str) -> Dict[str, str]:
    """Load ingredient to category mapping"""
    ingredient_to_category = {}
    with open(category_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ingredient = row['ingredient'].strip().lower()
            category = row['category'].strip()
            ingredient_to_category[ingredient] = category
    return ingredient_to_category


def load_node_mapping(nodes_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load mapping between ingredient names and node IDs"""
    name_to_id = {}
    id_to_name = {}
    
    with open(nodes_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned_name = row['cleaned_name'].strip().lower()
            original_name = row['name'].strip().lower()
            node_id = row['node_id'].strip()
            
            # Store both cleaned and original names
            if cleaned_name:
                name_to_id[cleaned_name] = node_id
                id_to_name[node_id] = cleaned_name
            if original_name and original_name != cleaned_name:
                name_to_id[original_name] = node_id
                # Prefer cleaned name for display
                if node_id not in id_to_name:
                    id_to_name[node_id] = original_name
                    
    return name_to_id, id_to_name


def load_embeddings(embeddings_path: str) -> Dict[str, np.ndarray]:
    """Load embeddings and convert to consistent format"""
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
    
    result = {}
    for k, v in data.items():
        key = str(k)
        if not isinstance(v, np.ndarray):
            result[key] = np.array(v, dtype=np.float32)
        else:
            result[key] = v.astype(np.float32)
    return result


def find_latest_embeddings(output_dir: str) -> str:
    """Find the most recent embeddings file"""
    candidates = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith('.pickle') and 'FlavorGraph+CSL-embedding' in f
    ]
    if not candidates:
        raise FileNotFoundError(f"No embeddings found in {output_dir}")
    return max(candidates, key=os.path.getmtime)


def find_ingredient_id(query: str, name_to_id: Dict[str, str]) -> Tuple[str, str]:
    """Find node_id for ingredient with fuzzy matching"""
    query_lower = query.lower()
    
    # Direct match
    if query_lower in name_to_id:
        return name_to_id[query_lower], query_lower
    
    # Substring matching
    matches = [(k, v) for k, v in name_to_id.items() if query_lower in k]
    if matches:
        if len(matches) > 1:
            print(f"Multiple matches for '{query}':")
            for i, (name, _) in enumerate(matches[:5], 1):
                print(f"  {i}. {name}")
            print(f"Using: {matches[0][0]}")
        return matches[0][1], matches[0][0]
    
    raise KeyError(f"Ingredient '{query}' not found")


def get_ingredient_category(ingredient_name: str, ingredient_to_category: Dict[str, str]) -> Optional[str]:
    """Get category for an ingredient with fuzzy matching"""
    ingredient_lower = ingredient_name.lower()
    
    # Direct match
    if ingredient_lower in ingredient_to_category:
        return ingredient_to_category[ingredient_lower]
    
    # Try partial matches
    for ing_key, category in ingredient_to_category.items():
        if ingredient_lower in ing_key or ing_key in ingredient_lower:
            return category
    
    return None


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    norm1 = np.linalg.norm(vec1) + 1e-12
    norm2 = np.linalg.norm(vec2) + 1e-12
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def find_substitutes(
    query_ingredient: str,
    name_to_id: Dict[str, str],
    id_to_name: Dict[str, str], 
    id_to_vec: Dict[str, np.ndarray],
    ingredient_to_category: Dict[str, str],
    target_category: Optional[str] = None,
    topn: int = 10
) -> List[Tuple[str, str, float]]:
    """Find ingredient substitutes within the same category"""
    
    # Get query ingredient info
    query_id, matched_name = find_ingredient_id(query_ingredient, name_to_id)
    if query_id not in id_to_vec:
        raise KeyError(f"No embeddings found for {matched_name}")
    
    query_vec = id_to_vec[query_id]
    
    # Determine target category
    if target_category is None:
        target_category = get_ingredient_category(matched_name, ingredient_to_category)
        if target_category is None:
            print(f"Warning: No category found for '{matched_name}', showing all similarities")
    
    print(f"Looking for {target_category or 'any'} substitutes for: {matched_name}")
    
    # Find candidates in the same category
    candidates = []
    for node_id, embedding in id_to_vec.items():
        if node_id == query_id:  # Skip self
            continue
            
        ingredient_name = id_to_name.get(node_id, f"ID_{node_id}")
        ingredient_category = get_ingredient_category(ingredient_name, ingredient_to_category)
        
        # Filter by category if specified
        if target_category and ingredient_category != target_category:
            continue
        
        similarity = cosine_similarity(query_vec, embedding)
        candidates.append((ingredient_name, ingredient_category or "Unknown", similarity))
    
    # Sort by similarity and return top N
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[:topn]


def main():
    parser = argparse.ArgumentParser(description='Find ingredient substitutes (not just similar items)')
    parser.add_argument('--ingredient', required=True, help='Ingredient to find substitutes for')
    parser.add_argument('--category', help='Force specific category (e.g., "Fruit", "Spice")')
    parser.add_argument('--topn', type=int, default=10, help='Number of substitutes to show')
    parser.add_argument('--embeddings', default='', help='Path to embeddings file')
    parser.add_argument('--nodes', default='./input/cleaned/nodes_cleaned_basic.csv')
    parser.add_argument('--categories', default='./input/dict_ingr2cate - Top300+FDB400+HyperFoods104=616.csv')
    parser.add_argument('--output_dir', default='./output/')
    
    args = parser.parse_args()
    
    try:
        # Load all data
        print("Loading ingredient categories...")
        ingredient_to_category = load_ingredient_categories(args.categories)
        
        print("Loading node mappings...")
        name_to_id, id_to_name = load_node_mapping(args.nodes)
        
        embeddings_path = args.embeddings or find_latest_embeddings(args.output_dir)
        print(f"Loading embeddings: {os.path.basename(embeddings_path)}")
        id_to_vec = load_embeddings(embeddings_path)
        
        # Find substitutes
        results = find_substitutes(
            args.ingredient,
            name_to_id,
            id_to_name,
            id_to_vec,
            ingredient_to_category,
            args.category,
            args.topn
        )
        
        print(f"\nTop {len(results)} substitutes:")
        print("-" * 70)
        for i, (name, category, sim) in enumerate(results, 1):
            print(f"{i:2d}. {name:<30} [{category:<20}] sim={sim:.4f}")
        
        if not results:
            print("No substitutes found. Try a different category or ingredient.")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
