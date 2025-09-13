#!/usr/bin/env python3
"""
Build ingredient co-occurrence edges from a recipes dataset.

Inputs:
- recipes JSON (list of objects with `ingredients` or `ingredient_names`)

Output CSV columns:
- id_1,id_2,score,edge_type

Score options:
- simple: 1.0 per co-occurrence
- normalized: 1/(n-1) per pair in an n-ingredient recipe
- weighted: uses cooccurrence_score if present

Usage:
  python3 scripts/data_expansion/build_edges_from_recipes.py \
    --recipes ./input/recipes/extracted_recipes.json \
    --output  ./input/edges_from_recipes.csv \
    --scoring normalized
"""

import argparse
import json
import csv
import itertools
from collections import defaultdict
from typing import List, Dict, Any, Tuple


def get_pairs(ingredients: List[int]) -> List[Tuple[int, int]]:
    # Generate undirected unique pairs sorted
    return [tuple(sorted(p)) for p in itertools.combinations(ingredients, 2)]


def main():
    parser = argparse.ArgumentParser(description='Build co-occurrence edges from recipes')
    parser.add_argument('--recipes', default='./input/recipes/extracted_recipes.json')
    parser.add_argument('--output', default='./input/edges_from_recipes.csv')
    parser.add_argument('--scoring', choices=['simple', 'normalized', 'weighted'], default='normalized')
    args = parser.parse_args()

    with open(args.recipes, 'r') as f:
        recipes = json.load(f)

    edge_scores: Dict[Tuple[int, int], float] = defaultdict(float)

    for r in recipes:
        ingredients: List[int]
        if 'ingredients' in r and isinstance(r['ingredients'], list) and r['ingredients'] and isinstance(r['ingredients'][0], int):
            ingredients = r['ingredients']
        elif 'ingredient_names' in r and isinstance(r['ingredient_names'], list):
            # If only names exist, skip since IDs are required to match nodes
            continue
        else:
            continue

        if len(ingredients) < 2:
            continue

        pairs = get_pairs(ingredients)

        if args.scoring == 'simple':
            increment = 1.0
            for a, b in pairs:
                edge_scores[(a, b)] += increment
        elif args.scoring == 'normalized':
            increment = 1.0 / (len(ingredients) - 1)
            for a, b in pairs:
                edge_scores[(a, b)] += increment
        else:  # weighted
            base = float(r.get('cooccurrence_score', 1.0))
            increment = base / max(1, len(pairs))
            for a, b in pairs:
                edge_scores[(a, b)] += increment

    # Write edges
    with open(args.output, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id_1', 'id_2', 'score', 'edge_type'])
        for (a, b), score in edge_scores.items():
            writer.writerow([a, b, score, 'ingr-ingr'])

    print(f'Wrote {len(edge_scores)} edges to {args.output}')


if __name__ == '__main__':
    main()


