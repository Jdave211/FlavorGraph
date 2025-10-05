#!/usr/bin/env python3
"""
Data Preprocessing and Validation Script
Validates FlavorGraph data and prepares it for training
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from collections import Counter


class FlavorGraphDataValidator:
    """Validates and preprocesses FlavorGraph data"""

    def __init__(self, base_dir: str = None):
        # Auto-detect base directory
        if base_dir is None:
            current = Path(__file__).parent.parent.resolve()
            if current.name == "FlavorGraph":
                base_dir = str(current)
            else:
                base_dir = str(Path.cwd())
        self.base_dir = Path(base_dir)
        self.input_dir = self.base_dir / "input"

        print("=" * 70)
        print("🔍 FlavorGraph Data Validator")
        print("=" * 70 + "\n")

    def validate_nodes(self) -> Tuple[bool, Dict]:
        """Validate nodes.csv file"""
        print("📋 Validating nodes...")

        nodes_path = self.input_dir / "nodes_191120.csv"
        if not nodes_path.exists():
            print("   ❌ nodes_191120.csv not found")
            return False, {}

        try:
            df = pd.read_csv(nodes_path)

            # Check required columns
            required_cols = ['node_id', 'name', 'node_type', 'is_hub']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                print(f"   ❌ Missing columns: {missing_cols}")
                return False, {}

            stats = {
                'total_nodes': len(df),
                'ingredients': len(df[df['node_type'] == 'ingredient']),
                'compounds': len(df[df['node_type'] == 'compound']),
                'hub_nodes': len(df[df['is_hub'] == 'hub']),
            }

            print(f"   ✓ Total nodes: {stats['total_nodes']}")
            print(f"   ✓ Ingredients: {stats['ingredients']}")
            print(f"   ✓ Compounds: {stats['compounds']}")
            print(f"   ✓ Hub nodes: {stats['hub_nodes']}")

            return True, stats

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False, {}

    def validate_edges(self) -> Tuple[bool, Dict]:
        """Validate edges.csv file"""
        print("\n🔗 Validating edges...")

        edges_path = self.input_dir / "edges_191120.csv"
        if not edges_path.exists():
            print("   ❌ edges_191120.csv not found")
            return False, {}

        try:
            df = pd.read_csv(edges_path)

            # Check required columns
            required_cols = ['id_1', 'id_2', 'score', 'edge_type']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                print(f"   ❌ Missing columns: {missing_cols}")
                return False, {}

            stats = {
                'total_edges': len(df),
                'edge_types': df['edge_type'].value_counts().to_dict(),
                'avg_score': df['score'].mean(),
                'min_score': df['score'].min(),
                'max_score': df['score'].max(),
            }

            print(f"   ✓ Total edges: {stats['total_edges']}")
            print(f"   ✓ Average score: {stats['avg_score']:.4f}")
            print(f"   ✓ Score range: [{stats['min_score']:.4f}, {stats['max_score']:.4f}]")
            print(f"   ✓ Edge types: {list(stats['edge_types'].keys())}")

            return True, stats

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False, {}

    def validate_recipes(self) -> Tuple[bool, Dict]:
        """Validate recipe data"""
        print("\n🍳 Validating recipes...")

        recipes_path = self.input_dir / "recipes" / "extracted_recipes.json"
        if not recipes_path.exists():
            print("   ❌ extracted_recipes.json not found")
            return False, {}

        try:
            with open(recipes_path, 'r') as f:
                recipes = json.load(f)

            # Analyze recipes
            ingredient_counts = [len(r.get('ingredients', [])) for r in recipes]
            cooccurrence_scores = [r.get('cooccurrence_score', 0) for r in recipes if 'cooccurrence_score' in r]

            stats = {
                'total_recipes': len(recipes),
                'avg_ingredients': sum(ingredient_counts) / len(ingredient_counts) if ingredient_counts else 0,
                'min_ingredients': min(ingredient_counts) if ingredient_counts else 0,
                'max_ingredients': max(ingredient_counts) if ingredient_counts else 0,
                'avg_cooccurrence': sum(cooccurrence_scores) / len(cooccurrence_scores) if cooccurrence_scores else 0,
            }

            print(f"   ✓ Total recipes: {stats['total_recipes']}")
            print(f"   ✓ Avg ingredients per recipe: {stats['avg_ingredients']:.2f}")
            print(f"   ✓ Ingredient range: [{stats['min_ingredients']}, {stats['max_ingredients']}]")
            if cooccurrence_scores:
                print(f"   ✓ Avg cooccurrence score: {stats['avg_cooccurrence']:.4f}")

            return True, stats

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False, {}

    def validate_compound_flavors(self) -> Tuple[bool, Dict]:
        """Validate compound flavor data"""
        print("\n🧪 Validating compound flavors...")

        compound_path = self.input_dir / "compound_flavors" / "compound_flavor_mappings.json"
        if not compound_path.exists():
            print("   ❌ compound_flavor_mappings.json not found")
            return False, {}

        try:
            with open(compound_path, 'r') as f:
                compounds = json.load(f)

            # Analyze flavor profiles
            primary_flavors = [v.get('primary_flavor') for v in compounds.values() if 'primary_flavor' in v]
            flavor_counter = Counter(primary_flavors)

            stats = {
                'total_compounds': len(compounds),
                'primary_flavor_distribution': dict(flavor_counter.most_common(5)),
            }

            print(f"   ✓ Total compounds: {stats['total_compounds']}")
            print(f"   ✓ Top primary flavors:")
            for flavor, count in flavor_counter.most_common(5):
                print(f"      - {flavor}: {count}")

            return True, stats

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False, {}

    def check_training_readiness(self) -> bool:
        """Check if all data is ready for training"""
        print("\n" + "=" * 70)
        print("🎯 Training Readiness Check")
        print("=" * 70 + "\n")

        # Check if training data generator exists
        generator_path = self.base_dir / "training" / "generate_llama_training_data.py"
        if not generator_path.exists():
            print("❌ Training data generator not found")
            return False

        print("✅ Training data generator found")

        # Check if training data has been generated
        training_data_path = self.base_dir / "training" / "data" / "flavorgraph_training_data.jsonl"
        if training_data_path.exists():
            print("✅ Training data already generated")

            # Count lines
            with open(training_data_path, 'r') as f:
                num_examples = sum(1 for _ in f)
            print(f"   📊 {num_examples} training examples found")

            return True
        else:
            print("⚠️  Training data not yet generated")
            print("   Run: python training/generate_llama_training_data.py")
            return False

    def validate_all(self) -> bool:
        """Run all validations"""
        results = []

        # Validate each component
        results.append(self.validate_nodes()[0])
        results.append(self.validate_edges()[0])
        results.append(self.validate_recipes()[0])
        results.append(self.validate_compound_flavors()[0])

        # Check training readiness
        training_ready = self.check_training_readiness()

        print("\n" + "=" * 70)
        print("📊 Validation Summary")
        print("=" * 70)

        if all(results):
            print("✅ All data files validated successfully!")
        else:
            print("⚠️  Some data files have issues")

        if training_ready:
            print("✅ Ready for training!")
        else:
            print("⚠️  Not ready for training yet")

        print("=" * 70 + "\n")

        return all(results) and training_ready


def main():
    validator = FlavorGraphDataValidator()
    validator.validate_all()


if __name__ == "__main__":
    main()
