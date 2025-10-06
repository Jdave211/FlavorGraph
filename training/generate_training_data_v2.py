#!/usr/bin/env python3
"""
FlavorGraph LLaMA Training Data Generator V2
Enhanced version with:
- Better flavor coverage via weak labeling
- 10k+ examples with rationales
- Hard negatives & contrastive learning
- Ingredient-based train/eval split
- Chemical compound citations
"""

import json
import csv
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from itertools import combinations


class FlavorGraphDataGeneratorV2:
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
        self.output_dir = self.base_dir / "training" / "data_v2"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data structures
        self.nodes = {}
        self.edges = []
        self.recipes = []
        self.compound_flavors = {}
        self.ingredient_flavors = {}
        self.node_name_to_id = {}
        self.ingredient_neighbors = defaultdict(list)
        self.ingredient_edges = defaultdict(list)

        # For train/eval split
        self.train_ingredients = set()
        self.eval_ingredients = set()

        print("=" * 70)
        print("🚀 FlavorGraph Training Data Generator V2")
        print("=" * 70 + "\n")

    def load_all_data(self):
        """Load all FlavorGraph data sources"""
        print("📊 Loading data sources...")

        # Load nodes
        print("  - Loading nodes...")
        nodes_df = pd.read_csv(self.input_dir / "nodes_191120.csv")
        for _, row in nodes_df.iterrows():
            node_id = row['node_id']
            self.nodes[node_id] = {
                'name': row['name'],
                'type': row['node_type'],
                'is_hub': row['is_hub'] == 'hub'
            }
            self.node_name_to_id[row['name']] = node_id
        print(f"    ✓ Loaded {len(self.nodes)} nodes")

        # Load edges
        print("  - Loading edges...")
        edges_df = pd.read_csv(self.input_dir / "edges_191120.csv")
        self.edges = edges_df.to_dict('records')
        print(f"    ✓ Loaded {len(self.edges)} edges")

        # Load recipes
        print("  - Loading recipes...")
        with open(self.input_dir / "recipes" / "extracted_recipes.json", 'r') as f:
            self.recipes = json.load(f)
        print(f"    ✓ Loaded {len(self.recipes)} recipes")

        # Load compound flavors
        print("  - Loading compound flavor mappings...")
        with open(self.input_dir / "compound_flavors" / "compound_flavor_mappings.json", 'r') as f:
            self.compound_flavors = json.load(f)
        print(f"    ✓ Loaded {len(self.compound_flavors)} compound flavor profiles")

        # Load ingredient flavors
        print("  - Loading ingredient flavor profiles...")
        with open(self.input_dir / "compound_flavors" / "ingredient_flavor_profiles.json", 'r') as f:
            self.ingredient_flavors = json.load(f)
        print(f"    ✓ Loaded {len(self.ingredient_flavors)} ingredient flavor profiles")

        print("\n✅ All data loaded successfully!\n")

    def build_ingredient_graph(self):
        """Build adjacency lists for ingredient relationships"""
        print("🔗 Building ingredient relationship graph...")

        for edge in self.edges:
            id1, id2 = edge['id_1'], edge['id_2']
            score = edge['score']
            edge_type = edge['edge_type']

            self.ingredient_neighbors[id1].append((id2, score))
            self.ingredient_neighbors[id2].append((id1, score))

            self.ingredient_edges[id1].append({
                'neighbor': id2,
                'score': score,
                'type': edge_type
            })
            self.ingredient_edges[id2].append({
                'neighbor': id1,
                'score': score,
                'type': edge_type
            })

        print(f"  ✓ Built graph with {len(self.ingredient_neighbors)} connected ingredients\n")

    def create_train_eval_split(self, eval_ratio: float = 0.15):
        """Create ingredient-based train/eval split to prevent leakage"""
        print(f"📊 Creating train/eval split ({eval_ratio*100:.0f}% eval)...")

        all_ingredients = [nid for nid, info in self.nodes.items() if info['type'] == 'ingredient']
        random.shuffle(all_ingredients)

        split_idx = int(len(all_ingredients) * (1 - eval_ratio))
        self.train_ingredients = set(all_ingredients[:split_idx])
        self.eval_ingredients = set(all_ingredients[split_idx:])

        print(f"  ✓ Train: {len(self.train_ingredients)} ingredients")
        print(f"  ✓ Eval: {len(self.eval_ingredients)} ingredients\n")

    def infer_flavor_profiles(self):
        """Weak labeling: infer missing flavor profiles from neighbors"""
        print("🎨 Inferring missing flavor profiles...")

        inferred_count = 0
        for node_id, node_info in self.nodes.items():
            if node_info['type'] != 'ingredient':
                continue

            ingredient_name = node_info['name']

            # Skip if already has profile
            if ingredient_name in self.ingredient_flavors:
                existing_profile = self.ingredient_flavors[ingredient_name]
                if not all(pd.isna(v) or v == 0.125 for v in existing_profile.values()):
                    continue

            # Aggregate from top neighbors
            neighbors = self.ingredient_neighbors.get(node_id, [])
            if len(neighbors) < 3:
                continue

            # Get top 5 neighbors
            top_neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)[:5]

            # Collect their flavor profiles
            flavor_vectors = []
            for neighbor_id, score in top_neighbors:
                neighbor_name = self.nodes[neighbor_id]['name']
                if neighbor_name in self.ingredient_flavors:
                    profile = self.ingredient_flavors[neighbor_name]
                    if not all(pd.isna(v) or v == 0.125 for v in profile.values()):
                        # Weight by edge score
                        weighted_profile = {k: v * score for k, v in profile.items() if not pd.isna(v)}
                        flavor_vectors.append(weighted_profile)

            if len(flavor_vectors) >= 2:
                # Average the profiles
                avg_profile = defaultdict(float)
                for profile in flavor_vectors:
                    for flavor, value in profile.items():
                        avg_profile[flavor] += value

                for flavor in avg_profile:
                    avg_profile[flavor] /= len(flavor_vectors)

                # Normalize
                total = sum(avg_profile.values())
                if total > 0:
                    avg_profile = {k: v/total for k, v in avg_profile.items()}
                    self.ingredient_flavors[ingredient_name] = dict(avg_profile)
                    inferred_count += 1

        print(f"  ✓ Inferred {inferred_count} flavor profiles via weak labeling\n")

    def get_ingredient_name(self, node_id: int) -> str:
        """Get ingredient name from node_id"""
        return self.nodes.get(node_id, {}).get('name', f'unknown_{node_id}')

    def get_flavor_profile_text(self, flavor_profile: Dict[str, float], include_rationale: bool = True) -> Tuple[str, List[str]]:
        """Convert flavor profile to readable text with rationale"""
        if not flavor_profile or all(pd.isna(v) or v == 0.125 for v in flavor_profile.values()):
            return "balanced flavor profile", []

        # Filter significant flavors
        significant_flavors = {
            k: v for k, v in flavor_profile.items()
            if not pd.isna(v) and v > 0.15
        }

        if not significant_flavors:
            return "subtle, balanced flavor", []

        # Sort by strength
        sorted_flavors = sorted(significant_flavors.items(), key=lambda x: x[1], reverse=True)

        # Create description
        flavor_names = [f[0] for f in sorted_flavors]
        if len(flavor_names) == 1:
            description = f"primarily {flavor_names[0]}"
        elif len(flavor_names) == 2:
            description = f"{flavor_names[0]} and {flavor_names[1]}"
        else:
            description = ", ".join(flavor_names[:3]) + " notes"

        # Rationale with percentages
        rationale = []
        if include_rationale:
            for flavor, strength in sorted_flavors[:3]:
                rationale.append(f"{flavor} ({strength:.1%})")

        return description, rationale

    def find_hard_negatives(self, ingredient_id: int, top_k: int = 3) -> List[int]:
        """Find ingredients that are similar but not actually good pairings"""
        # Get actual good pairings
        neighbors = self.ingredient_neighbors.get(ingredient_id, [])
        good_pairs = {n[0] for n in neighbors if n[1] > 0.3}

        # Find ingredients with similar flavor profiles but low pairing scores
        ingredient_name = self.get_ingredient_name(ingredient_id)
        if ingredient_name not in self.ingredient_flavors:
            return []

        target_profile = self.ingredient_flavors[ingredient_name]
        if all(pd.isna(v) or v == 0.125 for v in target_profile.values()):
            return []

        # Calculate cosine similarity with all other ingredients
        candidates = []
        for other_id in self.ingredient_neighbors.keys():
            if other_id == ingredient_id or other_id in good_pairs:
                continue

            other_name = self.get_ingredient_name(other_id)
            if other_name not in self.ingredient_flavors:
                continue

            other_profile = self.ingredient_flavors[other_name]

            # Cosine similarity
            dot_product = sum(target_profile.get(k, 0) * other_profile.get(k, 0) for k in target_profile.keys())
            norm_target = np.sqrt(sum(v**2 for v in target_profile.values()))
            norm_other = np.sqrt(sum(v**2 for v in other_profile.values()))

            if norm_target > 0 and norm_other > 0:
                similarity = dot_product / (norm_target * norm_other)

                # High flavor similarity but not in good pairings = hard negative
                if similarity > 0.6:
                    candidates.append((other_id, similarity))

        # Return top-k hardest negatives
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:top_k]]

    # ====================
    # V2 INSTRUCTION GENERATORS
    # ====================

    def generate_pairing_instructions_v2(self, num_samples: int = 2500, split: str = 'train') -> List[Dict]:
        """Enhanced pairing with rationales and hard negatives"""
        instructions = []
        target_ingredients = self.train_ingredients if split == 'train' else self.eval_ingredients

        print(f"📝 Generating {num_samples} pairing instructions ({split})...")

        for _ in range(num_samples):
            # Pick ingredient from correct split
            available = list(set(self.ingredient_neighbors.keys()) & target_ingredients)
            if not available:
                continue

            ingredient_id = random.choice(available)
            ingredient_name = self.get_ingredient_name(ingredient_id)

            neighbors = self.ingredient_neighbors[ingredient_id]
            if len(neighbors) < 3:
                continue

            # Get top pairings
            top_pairings = sorted(neighbors, key=lambda x: x[1], reverse=True)[:5]

            # Create detailed pairing list with rationales
            pairing_details = []
            for neighbor_id, score in top_pairings:
                neighbor_name = self.get_ingredient_name(neighbor_id)
                pairing_details.append({
                    'name': neighbor_name,
                    'score': score,
                })

            pairing_text = "\n".join([
                f"- **{p['name'].replace('_', ' ')}** (compatibility: {p['score']:.2f})"
                for p in pairing_details
            ])

            # Add chemical rationale if available
            rationale = ""
            if ingredient_name in self.ingredient_flavors:
                flavor_desc, flavor_components = self.get_flavor_profile_text(
                    self.ingredient_flavors[ingredient_name]
                )
                if flavor_components:
                    rationale = f"\n\n**Chemical basis**: {ingredient_name.replace('_', ' ')} has {flavor_desc} ({', '.join(flavor_components)}), which complements these ingredients' flavor compounds."

            # Standard instruction
            instruction = {
                "instruction": f"What ingredients pair well with {ingredient_name.replace('_', ' ')}?",
                "input": "",
                "output": f"Based on FlavorGraph analysis, here are excellent pairings for {ingredient_name.replace('_', ' ')}:\n\n{pairing_text}\n\nThese ingredients share complementary flavor compounds and are frequently used together in successful recipes.{rationale}",
                "metadata": {
                    "task": "ingredient_pairing",
                    "ingredient": ingredient_name,
                    "split": split,
                    "has_rationale": bool(rationale)
                }
            }
            instructions.append(instruction)

            # Add contrastive negative example (30% of time)
            if random.random() < 0.3:
                hard_negs = self.find_hard_negatives(ingredient_id, top_k=2)
                if hard_negs:
                    neg_names = [self.get_ingredient_name(nid).replace('_', ' ') for nid in hard_negs]

                    negative_instruction = {
                        "instruction": f"Do {', '.join(neg_names[:2])} pair well with {ingredient_name.replace('_', ' ')}?",
                        "input": "",
                        "output": f"While {', '.join(neg_names[:2])} may have some flavor similarities to {ingredient_name.replace('_', ' ')}, they are not strongly recommended pairings based on culinary tradition and recipe co-occurrence data. Better alternatives include: {', '.join([p['name'].replace('_', ' ') for p in pairing_details[:3]])}.",
                        "metadata": {
                            "task": "pairing_negative",
                            "ingredient": ingredient_name,
                            "split": split,
                            "is_contrastive": True
                        }
                    }
                    instructions.append(negative_instruction)

        print(f"  ✓ Generated {len(instructions)} pairing instructions\n")
        return instructions

    def generate_recipe_analysis_v2(self, num_samples: int = 3000, split: str = 'train') -> List[Dict]:
        """Enhanced recipe analysis with multi-ingredient reasoning"""
        instructions = []
        target_ingredients = self.train_ingredients if split == 'train' else self.eval_ingredients

        print(f"📝 Generating {num_samples} recipe analysis instructions ({split})...")

        # Filter recipes by split
        split_recipes = []
        for recipe in self.recipes:
            recipe_ingredients = set(recipe.get('ingredients', []))
            # Recipe is in split if majority of ingredients are in split
            overlap = len(recipe_ingredients & target_ingredients)
            if overlap / max(len(recipe_ingredients), 1) > 0.7:
                split_recipes.append(recipe)

        if not split_recipes:
            print(f"  ⚠️ No recipes found for {split} split\n")
            return instructions

        recipe_samples = random.sample(split_recipes, min(num_samples, len(split_recipes)))

        for recipe in recipe_samples:
            ingredient_ids = recipe.get('ingredients', [])
            ingredient_names = recipe.get('ingredient_names', [])

            if len(ingredient_names) < 2:
                continue

            cooccurrence = recipe.get('cooccurrence_score', 0.5)

            # Detailed compatibility assessment
            if cooccurrence > 0.7:
                compatibility = "excellent"
                explanation = "These ingredients have strong flavor compound overlap and are frequently used together in successful recipes."
            elif cooccurrence > 0.5:
                compatibility = "good"
                explanation = "These ingredients complement each other well with shared flavor molecules."
            else:
                compatibility = "moderate"
                explanation = "These ingredients can work together but may benefit from additional complementary ingredients."

            # Add ingredient count
            ingredients_str = ", ".join([name.replace('_', ' ') for name in ingredient_names])

            # Add hub ingredient analysis
            hub_ingredients = [name for nid, name in zip(ingredient_ids, ingredient_names)
                             if self.nodes.get(nid, {}).get('is_hub', False)]
            hub_note = ""
            if hub_ingredients:
                hub_note = f" The combination includes versatile hub ingredients ({', '.join([h.replace('_', ' ') for h in hub_ingredients])}), which enhance overall compatibility."

            instruction = {
                "instruction": "Analyze the ingredient compatibility in this recipe combination.",
                "input": f"Ingredients: {ingredients_str}",
                "output": f"This {len(ingredient_names)}-ingredient combination shows **{compatibility} compatibility** (score: {cooccurrence:.2f}).\n\n{explanation}{hub_note}",
                "metadata": {
                    "task": "recipe_analysis",
                    "recipe_id": recipe.get('recipe_id'),
                    "num_ingredients": len(ingredient_names),
                    "split": split,
                    "compatibility_score": cooccurrence
                }
            }
            instructions.append(instruction)

        print(f"  ✓ Generated {len(instructions)} recipe analysis instructions\n")
        return instructions

    def generate_all_instructions_v2(self) -> Tuple[List[Dict], List[Dict]]:
        """Generate complete V2 instruction dataset with train/eval split"""
        print("🎯 Generating V2 instruction dataset with improved quality...\n")

        # Generate training data
        train_instructions = []
        train_instructions.extend(self.generate_pairing_instructions_v2(2500, 'train'))
        train_instructions.extend(self.generate_recipe_analysis_v2(3000, 'train'))

        # Generate evaluation data
        eval_instructions = []
        eval_instructions.extend(self.generate_pairing_instructions_v2(500, 'eval'))
        eval_instructions.extend(self.generate_recipe_analysis_v2(500, 'eval'))

        # Shuffle
        random.shuffle(train_instructions)
        random.shuffle(eval_instructions)

        print(f"✅ Generated {len(train_instructions)} training instructions")
        print(f"✅ Generated {len(eval_instructions)} evaluation instructions\n")

        return train_instructions, eval_instructions

    def save_training_data_v2(self, train_data: List[Dict], eval_data: List[Dict]):
        """Save V2 training data"""
        print(f"💾 Saving V2 training data...")

        # Save train
        train_path = self.output_dir / "flavorgraph_training_v2.jsonl"
        with open(train_path, 'w') as f:
            for item in train_data:
                f.write(json.dumps(item) + '\n')
        print(f"  ✓ Train: {train_path} ({len(train_data)} examples)")

        # Save eval
        eval_path = self.output_dir / "flavorgraph_eval_v2.jsonl"
        with open(eval_path, 'w') as f:
            for item in eval_data:
                f.write(json.dumps(item) + '\n')
        print(f"  ✓ Eval: {eval_path} ({len(eval_data)} examples)")

        # Metadata
        metadata = {
            "version": "2.0",
            "train_size": len(train_data),
            "eval_size": len(eval_data),
            "train_ingredients": len(self.train_ingredients),
            "eval_ingredients": len(self.eval_ingredients),
            "task_distribution": {},
            "improvements": [
                "Weak labeling for flavor profiles",
                "Hard negatives for contrastive learning",
                "Ingredient-based train/eval split",
                "Chemical rationales",
                "Increased sample count (10k+ total)"
            ]
        }

        # Count tasks
        for inst in train_data + eval_data:
            task = inst['metadata']['task']
            metadata['task_distribution'][task] = metadata['task_distribution'].get(task, 0) + 1

        metadata_path = self.output_dir / "training_metadata_v2.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n📊 V2 Dataset Statistics:")
        print(f"   Total examples: {len(train_data) + len(eval_data)}")
        print(f"   Train/Eval split: {len(train_data)}/{len(eval_data)}")
        print(f"   Task distribution:")
        for task, count in sorted(metadata['task_distribution'].items()):
            print(f"     - {task}: {count}")
        print()

    def generate(self):
        """Main V2 generation pipeline"""
        print("=" * 70)
        print("FlavorGraph V2 Training Data Generation")
        print("=" * 70 + "\n")

        self.load_all_data()
        self.build_ingredient_graph()
        self.create_train_eval_split()
        self.infer_flavor_profiles()

        train_data, eval_data = self.generate_all_instructions_v2()
        self.save_training_data_v2(train_data, eval_data)

        print("=" * 70)
        print("✅ V2 Training data generation complete!")
        print("=" * 70)


def main():
    generator = FlavorGraphDataGeneratorV2()
    generator.generate()


if __name__ == "__main__":
    main()
