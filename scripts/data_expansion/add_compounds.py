#!/usr/bin/env python3
"""
Add new flavor compounds to FlavorGraph.

Inputs (seeds CSV):
- compound,node_id(optional),original_name(optional),primary_flavor(optional),flavor_strength(optional),salt,fat,acid,heat,umami,sweet,bitter,aromatic

Updates:
- input/cleaned/nodes_cleaned_basic.csv (append new compound nodes)
- input/compound_flavors/compound_flavor_mappings.csv (append flavor rows)

Node schema: node_id,name,id,node_type,is_hub,cleaned_name

Usage:
  python3 scripts/data_expansion/add_compounds.py --seeds data/expansion/compounds_new.csv
"""

import argparse
import csv
from typing import List, Dict, Tuple, Set


def read_nodes(nodes_path: str) -> Tuple[int, Set[str]]:
    max_node_id = -1
    existing_cleaned: Set[str] = set()
    with open(nodes_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # header
        for r in reader:
            try:
                nid = int(r[0])
                if nid > max_node_id:
                    max_node_id = nid
            except Exception:
                pass
            if len(r) >= 6:
                existing_cleaned.add(r[5].strip().lower())
    return max_node_id, existing_cleaned


def append_nodes(nodes_path: str, additions: List[List[str]]):
    with open(nodes_path, 'a') as f:
        writer = csv.writer(f)
        for r in additions:
            writer.writerow(r)


def append_compound_flavors(mapping_path: str, rows: List[List[str]]):
    with open(mapping_path, 'a') as f:
        writer = csv.writer(f)
        for r in rows:
            writer.writerow(r)


def clean_name(name: str) -> str:
    return name.strip().lower().replace(' ', '_').replace('-', '_')


def main():
    parser = argparse.ArgumentParser(description='Add flavor compounds')
    parser.add_argument('--seeds', required=True, help='CSV with compound flavor rows')
    parser.add_argument('--nodes', default='./input/cleaned/nodes_cleaned_basic.csv')
    parser.add_argument('--mappings', default='./input/compound_flavors/compound_flavor_mappings.csv')
    args = parser.parse_args()

    max_node_id, existing_cleaned = read_nodes(args.nodes)

    # Read seeds header for validation
    with open(args.seeds, 'r') as f:
        reader = csv.DictReader(f)
        required = ['compound', 'salt', 'fat', 'acid', 'heat', 'umami', 'sweet', 'bitter', 'aromatic']
        for field in required:
            if field not in reader.fieldnames:
                raise ValueError('Seeds CSV missing column: ' + field)
        seed_rows = list(reader)

    node_additions: List[List[str]] = []
    mapping_additions: List[List[str]] = []

    next_id = max_node_id + 1
    for row in seed_rows:
        compound = row['compound'].strip()
        original_name = row.get('original_name', compound).strip()
        node_id = row.get('node_id', '').strip()
        cleaned = clean_name(compound)
        
        # Ensure node_id assigned and node exists
        if not node_id:
            node_id = str(next_id)
            next_id += 1
        if cleaned not in existing_cleaned:
            node_additions.append([node_id, original_name, '', 'compound', 'food', cleaned])
            existing_cleaned.add(cleaned)

        primary_flavor = row.get('primary_flavor', '').strip() or 'aromatic'
        flavor_strength = row.get('flavor_strength', '').strip() or '0.5'

        mapping_row = [
            cleaned,
            node_id,
            original_name,
            primary_flavor,
            flavor_strength,
            row['salt'], row['fat'], row['acid'], row['heat'], row['umami'], row['sweet'], row['bitter'], row['aromatic']
        ]
        mapping_additions.append(mapping_row)

    if node_additions:
        append_nodes(args.nodes, node_additions)
        print(f'Appended {len(node_additions)} compound nodes')
    if mapping_additions:
        append_compound_flavors(args.mappings, mapping_additions)
        print(f'Appended {len(mapping_additions)} flavor mapping rows')


if __name__ == '__main__':
    main()


