#!/usr/bin/env python3
"""
Standalone Ingredient Similarity Search

This script is independent of the training codebase. It:
- Loads the latest FlavorGraph embeddings from ./output/
- Computes cosine similarity in pure NumPy (no sklearn dependency)
- Finds the top-N most similar ingredients/compounds to a query

Usage examples:
  python3 run.py --ingredient garlic --topn 10
  python3 run.py --ingredient lemon --embeddings ./output/FlavorGraph+CSL-embedding_...pickle
"""

import argparse
import os
import pickle
import sys
import csv
import numpy as np
from typing import Dict, List, Tuple


def find_latest_embeddings(output_dir: str) -> str:
    candidates = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith('.pickle') and 'FlavorGraph+CSL-embedding' in f
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No embeddings found in {output_dir}. Run training first or pass --embeddings explicitly."
        )
    return max(candidates, key=os.path.getmtime)


def load_node_mapping(nodes_path: str) -> Dict[str, str]:
    """Load mapping from ingredient name to node_id"""
    name_to_id = {}
    with open(nodes_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Use cleaned_name as the searchable name, map to node_id
            cleaned_name = row['cleaned_name'].strip()
            original_name = row['name'].strip()
            node_id = row['node_id'].strip()
            
            # Store both cleaned and original names
            if cleaned_name:
                name_to_id[cleaned_name.lower()] = node_id
            if original_name:
                name_to_id[original_name.lower()] = node_id
    return name_to_id


def load_embeddings(embeddings_path: str) -> Dict[str, np.ndarray]:
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
    # Ensure vectors are numpy arrays and convert keys to strings
    result = {}
    for k, v in data.items():
        key = str(k)  # Convert node_id to string
        if not isinstance(v, np.ndarray):
            result[key] = np.array(v, dtype=np.float32)
        else:
            result[key] = v.astype(np.float32)
    return result


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return matrix / norms


def cosine_topn(
    query_vec: np.ndarray,
    matrix: np.ndarray,
    names: List[str],
    exclude_name: str,
    topn: int,
) -> List[Tuple[str, float]]:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    # Cosine similarity via dot product (rows already normalized optional)
    sims = matrix @ q
    # Get top indices excluding the query itself if present
    sorted_idx = np.argsort(-sims)
    results = []
    for idx in sorted_idx:
        name = names[idx]
        if name == exclude_name:
            continue
        results.append((name, float(sims[idx])))
        if len(results) >= topn:
            break
    return results


def find_ingredient_id(query: str, name_to_id: Dict[str, str]) -> Tuple[str, str]:
    """Find node_id for ingredient name, with fuzzy matching"""
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
            print(f"Using first match: {matches[0][0]}")
        return matches[0][1], matches[0][0]
    
    # Reverse substring (query contains ingredient name)
    reverse_matches = [(k, v) for k, v in name_to_id.items() if k in query_lower]
    if reverse_matches:
        return reverse_matches[0][1], reverse_matches[0][0]
    
    raise KeyError(f"Ingredient '{query}' not found. Try: garlic, lemon, onion, etc.")


def main():
    parser = argparse.ArgumentParser(description='Ingredient similarity search (standalone).')
    parser.add_argument('--ingredient', required=True, help='Query ingredient/compound name (e.g., garlic)')
    parser.add_argument('--topn', type=int, default=10, help='Number of similar items to show')
    parser.add_argument('--embeddings', default='', help='Path to embeddings .pickle (optional)')
    parser.add_argument('--nodes', default='./input/cleaned/nodes_cleaned_basic.csv', help='Path to nodes CSV file')
    parser.add_argument('--output_dir', default='./output/', help='Directory to search for embeddings if not provided')
    args = parser.parse_args()

    try:
        # Load node mapping
        print(f"Loading node mapping: {args.nodes}")
        name_to_id = load_node_mapping(args.nodes)
        
        # Load embeddings
        embeddings_path = args.embeddings or find_latest_embeddings(args.output_dir)
        print(f"Loading embeddings: {embeddings_path}")
        id_to_vec = load_embeddings(embeddings_path)

        # Find the ingredient
        node_id, matched_name = find_ingredient_id(args.ingredient, name_to_id)
        if node_id not in id_to_vec:
            raise KeyError(f"Node ID {node_id} for '{matched_name}' not found in embeddings")
        
        # Prepare data for similarity search
        node_ids = list(id_to_vec.keys())
        matrix = np.stack([id_to_vec[nid] for nid in node_ids]).astype(np.float32)
        matrix = normalize_rows(matrix)
        
        query_vec = id_to_vec[node_id].astype(np.float32)

        # Find similar items
        results = cosine_topn(query_vec, matrix, node_ids, node_id, args.topn)

        # Create reverse mapping for display names
        id_to_name = {v: k for k, v in name_to_id.items()}

        print(f"\nTop similar to: {matched_name} (ID: {node_id})")
        print("-" * 60)
        for i, (nid, sim) in enumerate(results, 1):
            display_name = id_to_name.get(nid, f"ID_{nid}")
            print(f"{i:2d}. {display_name:<35}  cosine={sim:.4f}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


