#!/usr/bin/env python3
import argparse
import os
import re
import unicodedata
import pandas as pd

COMMON_PREFIXES = [
    'fresh_', 'dried_', 'canned_', 'frozen_', 'organic_', 'raw_', 'cooked_',
    'smoked_', 'pickled_', 'salted_', 'unsalted_', 'sweetened_', 'unsweetened_',
    'reduced_fat_', 'low_fat_', 'nonfat_', 'no_fat_', 'fat_free_',
    'boneless_', 'skinless_', 'shelled_', 'peeled_', 'seeded_', 'deveined_',
    'phil_', 'kraft_', 'generic_', 'store_brand_', 'brand_',
]

COMMON_SUFFIXES = [
    '_fruit', '_vegetable', '_meat', '_dairy', '_grain', '_spice', '_herb',
    '_sauce', '_dressing', '_seasoning', '_mix', '_blend', '_powder', '_extract',
    '_oil', '_juice', '_puree', '_paste'
]

UNIT_PREFIX_PATTERN = re.compile(r'^(\d+(?:\.\d+)?)_(inch|in|oz|ounce|ounces|g|gram|grams|kg|lb|pound|pounds|cm|mm)_')
PERCENT_PREFIX_PATTERN = re.compile(r'^(\d+(?:\.\d+)?)%_')
WHITESPACE_UNDERSCORE_PATTERN = re.compile(r'_+')
NON_ALNUM_PATTERN = re.compile(r'[^a-z0-9_]+')


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = unicodedata.normalize('NFKD', text)
    text = text.replace(' ', '_')
    return text


def strip_prefixes(name: str) -> str:
    # Remove leading percentage like 90%_
    name = PERCENT_PREFIX_PATTERN.sub('', name)

    # Remove leading numeric unit like 10_inch_
    name = UNIT_PREFIX_PATTERN.sub('', name)

    # Remove one common prefix if present (iterate to remove multiple sequential prefixes)
    changed = True
    while changed:
        changed = False
        for pref in COMMON_PREFIXES:
            if name.startswith(pref):
                name = name[len(pref):]
                changed = True
                break
    return name


def strip_suffixes(name: str) -> str:
    # Remove one common suffix if present
    for suff in COMMON_SUFFIXES:
        if name.endswith(suff):
            name = name[: -len(suff)]
            break
    return name


def clean_ingredient_name(raw_name: str) -> str:
    name = normalize_text(str(raw_name))

    # Early exit if empty
    if not name:
        return name

    name = strip_prefixes(name)
    name = strip_suffixes(name)

    # Remove any lingering non-alnum except underscore
    name = NON_ALNUM_PATTERN.sub('_', name)

    # Collapse multiple underscores and trim
    name = WHITESPACE_UNDERSCORE_PATTERN.sub('_', name).strip('_')

    return name


def process_nodes(input_csv: str, output_csv: str, overwrite: bool = True) -> None:
    df = pd.read_csv(input_csv)
    if 'name' not in df.columns or 'node_type' not in df.columns:
        raise ValueError('Input CSV must contain columns: name, node_type')

    original_names = df['name'].astype(str).tolist()

    cleaned_names = []
    num_changed = 0

    for i, row in df.iterrows():
        name = str(row['name'])
        node_type = row.get('node_type', '')
        if str(node_type) == 'ingredient':
            cleaned = clean_ingredient_name(name)
        else:
            cleaned = normalize_text(name)
        cleaned_names.append(cleaned)
        if cleaned != name:
            num_changed += 1

    df['cleaned_name'] = cleaned_names

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Wrote cleaned nodes to: {output_csv}")
    print(f"Total rows: {len(df)} | Changed names: {num_changed}")

    # Show a few examples of changes
    examples = []
    for orig, cleaned in zip(original_names, cleaned_names):
        if orig != cleaned:
            examples.append((orig, cleaned))
        if len(examples) >= 10:
            break
    if examples:
        print('\nExamples:')
        for o, c in examples:
            print(f" - {o} -> {c}")
    else:
        print('No changes were applied. Check prefixes/suffixes patterns.')


def main():
    parser = argparse.ArgumentParser(description='Basic ingredient name cleaner')
    parser.add_argument('--input', default='./input/nodes_191120.csv', type=str)
    parser.add_argument('--output', default='./input/cleaned/nodes_cleaned_basic.csv', type=str)
    args = parser.parse_args()

    process_nodes(args.input, args.output)


if __name__ == '__main__':
    main()
