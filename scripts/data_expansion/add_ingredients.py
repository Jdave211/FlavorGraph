#!/usr/bin/env python3
"""
Add new ingredients to FlavorGraph with categories and cleaned names.

Inputs:
- seeds CSV: ingredient,category

Updates:
- input/cleaned/nodes_cleaned_basic.csv (append new nodes)
- input/dict_ingr2cate - Top300+FDB400+HyperFoods104=616.csv (append categories)

Usage:
  python3 scripts/data_expansion/add_ingredients.py \
    --seeds data/expansion/seeds_asian.csv \
    --nodes ./input/cleaned/nodes_cleaned_basic.csv \
    --categories "./input/dict_ingr2cate - Top300+FDB400+HyperFoods104=616.csv"
"""

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple, Set


def read_nodes(nodes_path: str) -> Tuple[List[List[str]], Set[str], int]:
    rows: List[List[str]] = []
    max_node_id = -1
    existing_cleaned: Set[str] = set()
    
    with open(nodes_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        if header[:6] != ['node_id', 'name', 'id', 'node_type', 'is_hub', 'cleaned_name']:
            raise ValueError('Unexpected nodes header format')
        for r in reader:
            rows.append(r)
            try:
                nid = int(r[0])
                if nid > max_node_id:
                    max_node_id = nid
            except Exception:
                pass
            if len(r) >= 6:
                existing_cleaned.add(r[5].strip().lower())
    return rows, existing_cleaned, max_node_id


def append_nodes(nodes_path: str, new_nodes: List[List[str]]):
    with open(nodes_path, 'a') as f:
        writer = csv.writer(f)
        for r in new_nodes:
            writer.writerow(r)


def read_categories(cat_path: str) -> Set[str]:
    existing: Set[str] = set()
    with open(cat_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing.add(row['ingredient'].strip().lower())
    return existing


def append_categories(cat_path: str, additions: List[Tuple[str, str]]):
    # Ensure header exists by reading once
    with open(cat_path, 'r') as f:
        header = f.readline()
        if not header:
            raise ValueError('Category file missing header')
    with open(cat_path, 'a') as f:
        writer = csv.writer(f)
        for name, category in additions:
            writer.writerow([name, category])


def clean_name(name: str) -> str:
    return name.strip().lower().replace(' ', '_').replace('-', '_')


def main():
    parser = argparse.ArgumentParser(description='Add new ingredients with categories')
    parser.add_argument('--seeds', required=True, help='CSV with ingredient,category')
    parser.add_argument('--nodes', default='./input/cleaned/nodes_cleaned_basic.csv')
    parser.add_argument('--categories', default='./input/dict_ingr2cate - Top300+FDB400+HyperFoods104=616.csv')
    args = parser.parse_args()

    # Load existing data
    nodes_rows, existing_cleaned, max_node_id = read_nodes(args.nodes)
    existing_cat = read_categories(args.categories)

    # Read seeds
    seeds: List[Tuple[str, str]] = []
    with open(args.seeds, 'r') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['ingredient', 'category']:
            raise ValueError('Seeds CSV must have header: ingredient,category')
        for row in reader:
            seeds.append((row['ingredient'].strip(), row['category'].strip()))

    new_nodes: List[List[str]] = []
    new_cats: List[Tuple[str, str]] = []
    
    next_id = max_node_id + 1
    for ingredient, category in seeds:
        cleaned = clean_name(ingredient)
        if cleaned in existing_cleaned:
            continue
        # node format: node_id,name,id,node_type,is_hub,cleaned_name
        new_nodes.append([str(next_id), ingredient, '', 'ingredient', 'no_hub', cleaned])
        if cleaned not in existing_cat:
            new_cats.append((cleaned, category))
        existing_cleaned.add(cleaned)
        next_id += 1

    if not new_nodes and not new_cats:
        print('No additions needed.')
        return

    if new_nodes:
        append_nodes(args.nodes, new_nodes)
        print(f'Appended {len(new_nodes)} new ingredients to {args.nodes}')
    if new_cats:
        append_categories(args.categories, new_cats)
        print(f'Appended {len(new_cats)} new category rows to {args.categories}')


if __name__ == '__main__':
    main()


