#!/usr/bin/env python3
"""
FlavorGraph LLaMA Training Data Generator
Generates instruction-following dataset combining:
- Food ingredient relationships from graph
- Chemical compound flavor profiles
- Recipe mappings and co-occurrence patterns
"""

import json
import csv
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import pandas as pd
import numpy as np


class FlavorGraphDataGenerator:
    def __init__(self, base_dir: str = "/Users/davejaga/Desktop/Startups/FlavorGraph"):
        self.base_dir = Path(base_dir)
        self.input_dir = self.base_dir / "input"
        self.output_dir = self.base_dir / "training" / "data"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data structures
        self.nodes = {}  # node_id -> node info
        self.edges = []  # list of edges
        self.recipes = []  # recipe data
        self.compound_flavors = {}  # compound -> flavor profile
        self.ingredient_flavors = {}  # ingredient -> flavor profile
        self.node_name_to_id = {}  # name -> node_id

        print("🚀 Initializing FlavorGraph Training Data Generator")

    def load_all_data(self):
        """Load all FlavorGraph data sources"""
        print("\n📊 Loading data sources...")

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

        self.ingredient_neighbors = defaultdict(list)
        self.ingredient_edges = defaultdict(list)

        for edge in self.edges:
            id1, id2 = edge['id_1'], edge['id_2']
            score = edge['score']
            edge_type = edge['edge_type']

            # Store bidirectional relationships
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

    def get_ingredient_name(self, node_id: int) -> str:
        """Get ingredient name from node_id"""
        return self.nodes.get(node_id, {}).get('name', f'unknown_{node_id}')

    def get_flavor_profile_text(self, flavor_profile: Dict[str, float]) -> str:
        """Convert flavor profile to readable text"""
        if not flavor_profile or all(pd.isna(v) or v == 0.125 for v in flavor_profile.values()):
            return "balanced flavor profile"

        # Filter out NaN and low values
        significant_flavors = {
            k: v for k, v in flavor_profile.items()
            if not pd.isna(v) and v > 0.15
        }

        if not significant_flavors:
            return "subtle, balanced flavor"

        # Sort by strength
        sorted_flavors = sorted(significant_flavors.items(), key=lambda x: x[1], reverse=True)

        # Create description
        if len(sorted_flavors) == 1:
            return f"primarily {sorted_flavors[0][0]}"
        elif len(sorted_flavors) == 2:
            return f"{sorted_flavors[0][0]} and {sorted_flavors[1][0]}"
        else:
            flavors_str = ", ".join([f[0] for f in sorted_flavors[:3]])
            return f"{flavors_str} notes"

    # ====================
    # INSTRUCTION GENERATORS
    # ====================

    def generate_pairing_instructions(self, num_samples: int = 1000) -> List[Dict]:
        """Generate ingredient pairing recommendation instructions"""
        instructions = []

        print(f"📝 Generating {num_samples} ingredient pairing instructions...")

        for _ in range(num_samples):
            # Pick random ingredient with neighbors
            if not self.ingredient_neighbors:
                continue

            ingredient_id = random.choice(list(self.ingredient_neighbors.keys()))
            ingredient_name = self.get_ingredient_name(ingredient_id)

            # Get top pairings
            neighbors = self.ingredient_neighbors[ingredient_id]
            if len(neighbors) < 3:
                continue

            # Sort by score and get top 5
            top_pairings = sorted(neighbors, key=lambda x: x[1], reverse=True)[:5]

            # Create pairing list
            pairing_list = []
            for neighbor_id, score in top_pairings:
                neighbor_name = self.get_ingredient_name(neighbor_id)
                pairing_list.append(f"- {neighbor_name} (compatibility: {score:.2f})")

            pairing_text = "\n".join(pairing_list)

            # Generate instruction
            instruction = {
                "instruction": f"What ingredients pair well with {ingredient_name.replace('_', ' ')}?",
                "input": "",
                "output": f"Based on FlavorGraph analysis, here are excellent pairings for {ingredient_name.replace('_', ' ')}:\n\n{pairing_text}\n\nThese ingredients share complementary flavor compounds and are frequently used together in successful recipes.",
                "metadata": {
                    "task": "ingredient_pairing",
                    "ingredient": ingredient_name,
                    "num_pairings": len(top_pairings)
                }
            }
            instructions.append(instruction)

        print(f"  ✓ Generated {len(instructions)} pairing instructions\n")
        return instructions

    def generate_flavor_profile_instructions(self, num_samples: int = 500) -> List[Dict]:
        """Generate flavor profile explanation instructions"""
        instructions = []

        print(f"📝 Generating {num_samples} flavor profile instructions...")

        # Sample from compounds
        compound_samples = random.sample(
            list(self.compound_flavors.items()),
            min(num_samples // 2, len(self.compound_flavors))
        )

        for compound_name, data in compound_samples:
            flavor_profile = data.get('flavor_profile', {})
            flavor_text = self.get_flavor_profile_text(flavor_profile)
            primary_flavor = data.get('primary_flavor', 'balanced')

            instruction = {
                "instruction": f"Describe the flavor profile of the compound {compound_name}.",
                "input": "",
                "output": f"The compound {compound_name} has a {flavor_text} flavor profile. Its primary characteristic is {primary_flavor}, which contributes to the overall taste and aroma of foods containing this molecule.",
                "metadata": {
                    "task": "flavor_profile",
                    "compound": compound_name,
                    "type": "compound"
                }
            }
            instructions.append(instruction)

        # Sample from ingredients
        ingredient_samples = random.sample(
            list(self.nodes.items()),
            min(num_samples // 2, len(self.nodes))
        )

        for node_id, node_info in ingredient_samples:
            ingredient_name = node_info['name']

            # Check if we have flavor data
            if ingredient_name in self.ingredient_flavors:
                flavor_profile = self.ingredient_flavors[ingredient_name]
                flavor_text = self.get_flavor_profile_text(flavor_profile)
            else:
                # Use generic description
                flavor_text = "complex, multi-dimensional"

            is_hub = node_info.get('is_hub', False)
            hub_text = " It's a foundational ingredient that connects well with many other foods." if is_hub else ""

            instruction = {
                "instruction": f"What is the flavor profile of {ingredient_name.replace('_', ' ')}?",
                "input": "",
                "output": f"{ingredient_name.replace('_', ' ').title()} has a {flavor_text} flavor profile.{hub_text}",
                "metadata": {
                    "task": "flavor_profile",
                    "ingredient": ingredient_name,
                    "type": "ingredient"
                }
            }
            instructions.append(instruction)

        print(f"  ✓ Generated {len(instructions)} flavor profile instructions\n")
        return instructions

    def generate_recipe_analysis_instructions(self, num_samples: int = 800) -> List[Dict]:
        """Generate recipe understanding instructions"""
        instructions = []

        print(f"📝 Generating {num_samples} recipe analysis instructions...")

        recipe_samples = random.sample(self.recipes, min(num_samples, len(self.recipes)))

        for recipe in recipe_samples:
            ingredient_names = recipe.get('ingredient_names', [])
            if len(ingredient_names) < 2:
                continue

            # Format ingredients
            ingredients_str = ", ".join([name.replace('_', ' ') for name in ingredient_names])
            cooccurrence = recipe.get('cooccurrence_score', 0.5)

            # Determine compatibility
            if cooccurrence > 0.7:
                compatibility = "excellent"
                explanation = "These ingredients have strong flavor compound overlap and are frequently used together."
            elif cooccurrence > 0.5:
                compatibility = "good"
                explanation = "These ingredients complement each other well with shared flavor molecules."
            else:
                compatibility = "moderate"
                explanation = "These ingredients can work together but may benefit from additional complementary ingredients."

            instruction = {
                "instruction": "Analyze the ingredient compatibility in this recipe combination.",
                "input": f"Ingredients: {ingredients_str}",
                "output": f"This combination shows {compatibility} compatibility (score: {cooccurrence:.2f}). {explanation}",
                "metadata": {
                    "task": "recipe_analysis",
                    "recipe_id": recipe.get('recipe_id'),
                    "num_ingredients": len(ingredient_names)
                }
            }
            instructions.append(instruction)

        # Add "why do ingredients work" questions
        for recipe in recipe_samples[:num_samples//4]:
            ingredient_names = recipe.get('ingredient_names', [])
            if len(ingredient_names) < 2:
                continue

            ingredients_str = " and ".join([name.replace('_', ' ') for name in ingredient_names[:2]])

            instruction = {
                "instruction": f"Why do {ingredients_str} work well together?",
                "input": "",
                "output": f"{ingredients_str.title()} pair well because they share complementary flavor compounds. Their chemical profiles create a harmonious taste experience when combined, which is why they frequently appear together in recipes.",
                "metadata": {
                    "task": "pairing_explanation",
                    "ingredients": ingredient_names[:2]
                }
            }
            instructions.append(instruction)

        print(f"  ✓ Generated {len(instructions)} recipe analysis instructions\n")
        return instructions

    def generate_substitution_instructions(self, num_samples: int = 500) -> List[Dict]:
        """Generate ingredient substitution recommendations"""
        instructions = []

        print(f"📝 Generating {num_samples} substitution instructions...")

        for _ in range(num_samples):
            if not self.ingredient_neighbors:
                continue

            # Pick ingredient with good neighbors
            ingredient_id = random.choice(list(self.ingredient_neighbors.keys()))
            ingredient_name = self.get_ingredient_name(ingredient_id)

            neighbors = self.ingredient_neighbors[ingredient_id]
            if len(neighbors) < 2:
                continue

            # Get top substitutes (high similarity)
            top_substitutes = sorted(neighbors, key=lambda x: x[1], reverse=True)[:3]

            substitute_list = []
            for neighbor_id, score in top_substitutes:
                neighbor_name = self.get_ingredient_name(neighbor_id)
                substitute_list.append(f"- {neighbor_name.replace('_', ' ')}")

            substitutes_text = "\n".join(substitute_list)

            instruction = {
                "instruction": f"What can I substitute for {ingredient_name.replace('_', ' ')} in a recipe?",
                "input": "",
                "output": f"Good substitutes for {ingredient_name.replace('_', ' ')} include:\n\n{substitutes_text}\n\nThese alternatives have similar flavor profiles and chemical compositions, making them suitable replacements.",
                "metadata": {
                    "task": "substitution",
                    "ingredient": ingredient_name
                }
            }
            instructions.append(instruction)

        print(f"  ✓ Generated {len(instructions)} substitution instructions\n")
        return instructions

    def generate_chemical_relationship_instructions(self, num_samples: int = 300) -> List[Dict]:
        """Generate instructions about chemical compounds in food"""
        instructions = []

        print(f"📝 Generating {num_samples} chemical relationship instructions...")

        compound_samples = random.sample(
            list(self.compound_flavors.items()),
            min(num_samples, len(self.compound_flavors))
        )

        for compound_name, data in compound_samples:
            primary_flavor = data.get('primary_flavor', 'balanced')
            flavor_strength = data.get('flavor_strength', 0.5)

            strength_desc = "strong" if flavor_strength > 0.5 else "subtle"

            instruction = {
                "instruction": f"What role does {compound_name} play in food flavor?",
                "input": "",
                "output": f"{compound_name} is a flavor compound that contributes {strength_desc} {primary_flavor} notes to foods. It's one of the molecular building blocks that creates the overall taste and aroma profile.",
                "metadata": {
                    "task": "chemical_role",
                    "compound": compound_name
                }
            }
            instructions.append(instruction)

        print(f"  ✓ Generated {len(instructions)} chemical relationship instructions\n")
        return instructions

    def generate_hub_ingredient_instructions(self, num_samples: int = 200) -> List[Dict]:
        """Generate instructions about hub ingredients (highly connected)"""
        instructions = []

        print(f"📝 Generating hub ingredient instructions...")

        # Find hub ingredients
        hub_ingredients = [(nid, info) for nid, info in self.nodes.items() if info.get('is_hub')]

        if not hub_ingredients:
            print("  ⚠️  No hub ingredients found\n")
            return instructions

        for node_id, node_info in hub_ingredients[:num_samples]:
            ingredient_name = node_info['name']

            # Get number of connections
            num_connections = len(self.ingredient_neighbors.get(node_id, []))

            instruction = {
                "instruction": f"What makes {ingredient_name.replace('_', ' ')} a versatile ingredient?",
                "input": "",
                "output": f"{ingredient_name.replace('_', ' ').title()} is a highly versatile hub ingredient that pairs well with {num_connections}+ other ingredients. Its balanced flavor profile and chemical composition make it compatible with a wide range of foods, making it a foundational element in many recipes.",
                "metadata": {
                    "task": "hub_ingredient",
                    "ingredient": ingredient_name,
                    "connections": num_connections
                }
            }
            instructions.append(instruction)

        print(f"  ✓ Generated {len(instructions)} hub ingredient instructions\n")
        return instructions

    def generate_all_instructions(self) -> List[Dict]:
        """Generate complete instruction dataset"""
        print("🎯 Generating complete instruction dataset...\n")

        all_instructions = []

        # Generate different types of instructions
        all_instructions.extend(self.generate_pairing_instructions(1000))
        all_instructions.extend(self.generate_flavor_profile_instructions(500))
        all_instructions.extend(self.generate_recipe_analysis_instructions(800))
        all_instructions.extend(self.generate_substitution_instructions(500))
        all_instructions.extend(self.generate_chemical_relationship_instructions(300))
        all_instructions.extend(self.generate_hub_ingredient_instructions(200))

        # Shuffle to mix instruction types
        random.shuffle(all_instructions)

        print(f"✅ Generated {len(all_instructions)} total instructions\n")
        return all_instructions

    def save_training_data(self, instructions: List[Dict], output_file: str = "flavorgraph_training_data.jsonl"):
        """Save instructions as JSONL for training"""
        output_path = self.output_dir / output_file

        print(f"💾 Saving training data to {output_path}...")

        with open(output_path, 'w') as f:
            for instruction in instructions:
                f.write(json.dumps(instruction) + '\n')

        print(f"✅ Saved {len(instructions)} instructions\n")

        # Also save metadata
        metadata = {
            "total_instructions": len(instructions),
            "task_distribution": {},
            "data_sources": {
                "nodes": len(self.nodes),
                "edges": len(self.edges),
                "recipes": len(self.recipes),
                "compounds": len(self.compound_flavors)
            }
        }

        # Count tasks
        for inst in instructions:
            task = inst['metadata']['task']
            metadata['task_distribution'][task] = metadata['task_distribution'].get(task, 0) + 1

        metadata_path = self.output_dir / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"📊 Training data statistics:")
        print(f"   Total instructions: {metadata['total_instructions']}")
        print(f"   Task distribution:")
        for task, count in sorted(metadata['task_distribution'].items()):
            print(f"     - {task}: {count}")
        print()

    def generate(self):
        """Main generation pipeline"""
        print("=" * 60)
        print("FlavorGraph LLaMA Training Data Generation")
        print("=" * 60 + "\n")

        # Load all data
        self.load_all_data()

        # Build graph
        self.build_ingredient_graph()

        # Generate instructions
        instructions = self.generate_all_instructions()

        # Save
        self.save_training_data(instructions)

        print("=" * 60)
        print("✅ Training data generation complete!")
        print("=" * 60)


def main():
    generator = FlavorGraphDataGenerator()
    generator.generate()


if __name__ == "__main__":
    main()
