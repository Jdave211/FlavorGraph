#!/usr/bin/env python3
"""
FlavorGraph Model Evaluation Script
Evaluates fine-tuned LLaMA models on food understanding tasks
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    task: str
    accuracy: float
    examples: List[Dict]
    metrics: Dict[str, float]


class FlavorGraphEvaluator:
    """Evaluates FlavorGraph-trained models"""

    def __init__(self, model_path: str, base_model: str = None):
        self.model_path = Path(model_path)

        print("=" * 70)
        print("🎯 FlavorGraph Model Evaluator")
        print("=" * 70 + "\n")

        # Load model and tokenizer
        print(f"📥 Loading model from: {model_path}")
        self.load_model(base_model)

    def load_model(self, base_model: str = None):
        """Load fine-tuned model"""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Load base model + LoRA adapter
        if base_model:
            print(f"   Loading base model: {base_model}")
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.model = PeftModel.from_pretrained(base, self.model_path)
        else:
            # Try to load merged model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )

        self.model.eval()
        print("✅ Model loaded successfully\n")

    def generate_response(self, prompt: str, max_length: int = 200) -> str:
        """Generate model response for a prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the response part (after "### Response:")
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()

        return response

    def evaluate_ingredient_pairing(self, test_cases: List[Dict]) -> EvaluationResult:
        """Evaluate ingredient pairing recommendations"""
        print("🔗 Evaluating ingredient pairing...")

        results = []
        correct = 0

        for case in tqdm(test_cases, desc="Pairing tasks"):
            ingredient = case['ingredient']
            expected_pairings = set(case['expected_pairings'])

            # Generate prompt
            prompt = f"### Instruction:\nWhat ingredients pair well with {ingredient}?\n\n### Response:\n"

            # Get model response
            response = self.generate_response(prompt)

            # Check if expected pairings appear in response
            mentioned = sum(1 for p in expected_pairings if p.lower() in response.lower())
            score = mentioned / len(expected_pairings) if expected_pairings else 0

            if score >= 0.5:  # At least half of expected pairings mentioned
                correct += 1

            results.append({
                'ingredient': ingredient,
                'expected': list(expected_pairings),
                'response': response,
                'score': score
            })

        accuracy = correct / len(test_cases) if test_cases else 0

        print(f"   ✓ Accuracy: {accuracy:.2%}\n")

        return EvaluationResult(
            task="ingredient_pairing",
            accuracy=accuracy,
            examples=results[:5],  # Keep first 5 for inspection
            metrics={'mention_rate': np.mean([r['score'] for r in results])}
        )

    def evaluate_flavor_profile(self, test_cases: List[Dict]) -> EvaluationResult:
        """Evaluate flavor profile understanding"""
        print("🎨 Evaluating flavor profile understanding...")

        results = []
        correct = 0

        flavor_keywords = {
            'sweet': ['sweet', 'sugar', 'honey'],
            'salty': ['salt', 'salty', 'savory'],
            'sour': ['sour', 'acid', 'tart', 'tangy'],
            'bitter': ['bitter'],
            'umami': ['umami', 'savory'],
            'aromatic': ['aromatic', 'fragrant', 'floral'],
        }

        for case in tqdm(test_cases, desc="Flavor tasks"):
            ingredient = case['ingredient']
            expected_flavors = case['expected_flavors']

            # Generate prompt
            prompt = f"### Instruction:\nDescribe the flavor profile of {ingredient}.\n\n### Response:\n"

            # Get model response
            response = self.generate_response(prompt)

            # Check if expected flavors are mentioned
            response_lower = response.lower()
            mentioned_flavors = []

            for flavor in expected_flavors:
                keywords = flavor_keywords.get(flavor, [flavor])
                if any(kw in response_lower for kw in keywords):
                    mentioned_flavors.append(flavor)

            score = len(mentioned_flavors) / len(expected_flavors) if expected_flavors else 0

            if score >= 0.5:
                correct += 1

            results.append({
                'ingredient': ingredient,
                'expected_flavors': expected_flavors,
                'mentioned_flavors': mentioned_flavors,
                'response': response,
                'score': score
            })

        accuracy = correct / len(test_cases) if test_cases else 0

        print(f"   ✓ Accuracy: {accuracy:.2%}\n")

        return EvaluationResult(
            task="flavor_profile",
            accuracy=accuracy,
            examples=results[:5],
            metrics={'flavor_mention_rate': np.mean([r['score'] for r in results])}
        )

    def evaluate_substitution(self, test_cases: List[Dict]) -> EvaluationResult:
        """Evaluate ingredient substitution recommendations"""
        print("🔄 Evaluating substitution recommendations...")

        results = []
        correct = 0

        for case in tqdm(test_cases, desc="Substitution tasks"):
            ingredient = case['ingredient']
            valid_substitutes = set(case['valid_substitutes'])

            # Generate prompt
            prompt = f"### Instruction:\nWhat can I substitute for {ingredient}?\n\n### Response:\n"

            # Get model response
            response = self.generate_response(prompt)

            # Check if valid substitutes appear
            mentioned = sum(1 for s in valid_substitutes if s.lower() in response.lower())
            score = mentioned / len(valid_substitutes) if valid_substitutes else 0

            if score >= 0.3:  # At least 30% of valid substitutes mentioned
                correct += 1

            results.append({
                'ingredient': ingredient,
                'valid_substitutes': list(valid_substitutes),
                'response': response,
                'score': score
            })

        accuracy = correct / len(test_cases) if test_cases else 0

        print(f"   ✓ Accuracy: {accuracy:.2%}\n")

        return EvaluationResult(
            task="substitution",
            accuracy=accuracy,
            examples=results[:5],
            metrics={'substitute_mention_rate': np.mean([r['score'] for r in results])}
        )

    def evaluate_recipe_analysis(self, test_cases: List[Dict]) -> EvaluationResult:
        """Evaluate recipe compatibility analysis"""
        print("🍳 Evaluating recipe analysis...")

        results = []
        correct = 0

        for case in tqdm(test_cases, desc="Recipe tasks"):
            ingredients = case['ingredients']
            expected_compatibility = case['compatibility']  # 'good', 'moderate', 'poor'

            # Generate prompt
            ingredients_str = ", ".join(ingredients)
            prompt = f"### Instruction:\nAnalyze the ingredient compatibility in this recipe combination.\n\n### Input:\nIngredients: {ingredients_str}\n\n### Response:\n"

            # Get model response
            response = self.generate_response(prompt)

            # Simple keyword matching for compatibility assessment
            response_lower = response.lower()
            predicted_compatibility = None

            if any(word in response_lower for word in ['excellent', 'great', 'very good']):
                predicted_compatibility = 'good'
            elif any(word in response_lower for word in ['moderate', 'decent', 'acceptable']):
                predicted_compatibility = 'moderate'
            elif any(word in response_lower for word in ['poor', 'bad', 'weak']):
                predicted_compatibility = 'poor'

            is_correct = (predicted_compatibility == expected_compatibility)
            if is_correct:
                correct += 1

            results.append({
                'ingredients': ingredients,
                'expected': expected_compatibility,
                'predicted': predicted_compatibility,
                'response': response,
                'correct': is_correct
            })

        accuracy = correct / len(test_cases) if test_cases else 0

        print(f"   ✓ Accuracy: {accuracy:.2%}\n")

        return EvaluationResult(
            task="recipe_analysis",
            accuracy=accuracy,
            examples=results[:5],
            metrics={'classification_accuracy': accuracy}
        )

    def run_full_evaluation(self, test_data_path: str = None) -> Dict[str, Any]:
        """Run comprehensive evaluation"""
        print("\n🎯 Running full evaluation suite...\n")

        # Load or generate test cases
        test_cases = self.load_test_cases(test_data_path)

        # Run all evaluations
        results = {
            'pairing': self.evaluate_ingredient_pairing(test_cases.get('pairing', [])),
            'flavor': self.evaluate_flavor_profile(test_cases.get('flavor', [])),
            'substitution': self.evaluate_substitution(test_cases.get('substitution', [])),
            'recipe': self.evaluate_recipe_analysis(test_cases.get('recipe', []))
        }

        # Calculate overall metrics
        overall_accuracy = np.mean([r.accuracy for r in results.values()])

        print("=" * 70)
        print("📊 EVALUATION SUMMARY")
        print("=" * 70)
        print(f"\n🎯 Overall Accuracy: {overall_accuracy:.2%}\n")

        for task_name, result in results.items():
            print(f"   {task_name.upper()}: {result.accuracy:.2%}")

        print("\n" + "=" * 70)

        return {
            'overall_accuracy': overall_accuracy,
            'task_results': results
        }

    def load_test_cases(self, test_data_path: str = None) -> Dict[str, List[Dict]]:
        """Load or generate test cases"""
        if test_data_path and Path(test_data_path).exists():
            with open(test_data_path, 'r') as f:
                return json.load(f)

        # Generate sample test cases
        print("⚠️  No test data provided, using sample test cases\n")

        return {
            'pairing': [
                {'ingredient': 'tomato', 'expected_pairings': ['basil', 'mozzarella', 'garlic']},
                {'ingredient': 'chicken', 'expected_pairings': ['lemon', 'garlic', 'thyme']},
                {'ingredient': 'chocolate', 'expected_pairings': ['coffee', 'vanilla', 'strawberry']},
            ],
            'flavor': [
                {'ingredient': 'lemon', 'expected_flavors': ['sour', 'aromatic']},
                {'ingredient': 'vanilla', 'expected_flavors': ['sweet', 'aromatic']},
                {'ingredient': 'soy sauce', 'expected_flavors': ['salty', 'umami']},
            ],
            'substitution': [
                {'ingredient': 'butter', 'valid_substitutes': ['oil', 'margarine', 'ghee']},
                {'ingredient': 'milk', 'valid_substitutes': ['cream', 'almond milk', 'soy milk']},
            ],
            'recipe': [
                {'ingredients': ['tomato', 'basil', 'mozzarella'], 'compatibility': 'good'},
                {'ingredients': ['peanut butter', 'jelly', 'bread'], 'compatibility': 'good'},
            ]
        }

    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert results to serializable format
        serializable = {
            'overall_accuracy': results['overall_accuracy'],
            'tasks': {}
        }

        for task_name, result in results['task_results'].items():
            serializable['tasks'][task_name] = {
                'accuracy': result.accuracy,
                'metrics': result.metrics,
                'examples': result.examples
            }

        with open(output_file, 'w') as f:
            json.dump(serializable, f, indent=2)

        print(f"\n💾 Results saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate FlavorGraph Model")
    parser.add_argument("--model", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--base_model", type=str, help="Base model name (if using LoRA)")
    parser.add_argument("--test_data", type=str, help="Path to test data JSON")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file")

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = FlavorGraphEvaluator(args.model, args.base_model)

    # Run evaluation
    results = evaluator.run_full_evaluation(args.test_data)

    # Save results
    evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()
