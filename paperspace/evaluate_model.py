#!/usr/bin/env python3
"""
FlavorGraph AI Model Evaluation and Inference Script
Tests the trained model on various FlavorGraph tasks
"""

import torch
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class FlavorGraphEvaluator:
    def __init__(self, model_path: str, base_model: str = None):
        self.model_path = Path(model_path)
        self.base_model = base_model
        self.tokenizer = None
        self.model = None
        self.embeddings_reference = None
        
        print(f"🧠 Initializing FlavorGraph AI Evaluator")
        print(f"📁 Model path: {model_path}")
        
        self.load_model()
        self.load_reference_data()
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        print("🤖 Loading model and tokenizer...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        if self.base_model:
            base_model_path = self.base_model
        else:
            # Try to infer from config
            config_path = self.model_path / "training_config.yaml"
            if config_path.exists():
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                base_model_path = config['model_config']['base_model']
            else:
                raise ValueError("Base model not specified and config not found")
        
        print(f"📦 Loading base model: {base_model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load LoRA weights
        print("⚡ Loading LoRA weights...")
        self.model = PeftModel.from_pretrained(self.model, self.model_path)
        self.model.eval()
        
        print("✅ Model loaded successfully")
    
    def load_reference_data(self):
        """Load FlavorGraph reference data for evaluation"""
        print("📊 Loading reference data...")
        
        # Load embeddings reference
        embeddings_path = Path("paperspace/training_data/embeddings_reference.pkl")
        if embeddings_path.exists():
            with open(embeddings_path, 'rb') as f:
                self.embeddings_reference = pickle.load(f)
            print(f"✅ Loaded {len(self.embeddings_reference)} reference embeddings")
        
        # Load nodes data
        nodes_path = Path("input/cleaned/nodes_cleaned_basic.csv")
        if nodes_path.exists():
            self.nodes_df = pd.read_csv(nodes_path)
            print(f"✅ Loaded {len(self.nodes_df)} nodes")
        
        # Load categories
        categories_path = Path("input/dict_ingr2cate - Top300+FDB400+HyperFoods104=616.csv")
        if categories_path.exists():
            self.categories_df = pd.read_csv(categories_path)
            print(f"✅ Loaded {len(self.categories_df)} categories")
    
    def generate_response(self, instruction: str, input_text: str = "", max_length: int = 256, 
                         temperature: float = 0.7, top_p: float = 0.9) -> str:
        """Generate response for a given instruction"""
        
        # Format prompt
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the response part
        if "### Response:\n" in full_response:
            response = full_response.split("### Response:\n")[-1].strip()
        else:
            response = full_response[len(prompt):].strip()
        
        return response
    
    def evaluate_ingredient_knowledge(self, test_ingredients: List[str] = None) -> Dict[str, Any]:
        """Evaluate ingredient knowledge understanding"""
        print("🥕 Evaluating ingredient knowledge...")
        
        if test_ingredients is None:
            # Sample some ingredients for testing
            test_ingredients = ['garlic', 'tomato', 'basil', 'chicken', 'olive_oil', 'onion', 'cheese']
        
        results = {}
        
        for ingredient in test_ingredients:
            print(f"   Testing: {ingredient}")
            
            # Test basic knowledge
            response = self.generate_response(
                f"What can you tell me about {ingredient.replace('_', ' ')}?",
                ingredient
            )
            
            results[ingredient] = {
                'knowledge_response': response,
                'response_length': len(response.split()),
                'mentions_category': any(cat in response.lower() for cat in ['fruit', 'vegetable', 'spice', 'dairy', 'meat', 'grain']),
                'mentions_flavor': any(flavor in response.lower() for flavor in ['sweet', 'salty', 'sour', 'bitter', 'umami', 'spicy'])
            }
        
        # Calculate summary metrics
        avg_length = np.mean([r['response_length'] for r in results.values()])
        category_mention_rate = np.mean([r['mentions_category'] for r in results.values()])
        flavor_mention_rate = np.mean([r['mentions_flavor'] for r in results.values()])
        
        summary = {
            'avg_response_length': avg_length,
            'category_mention_rate': category_mention_rate,
            'flavor_mention_rate': flavor_mention_rate,
            'individual_results': results
        }
        
        print(f"   ✅ Average response length: {avg_length:.1f} words")
        print(f"   ✅ Category mention rate: {category_mention_rate:.2%}")
        print(f"   ✅ Flavor mention rate: {flavor_mention_rate:.2%}")
        
        return summary
    
    def evaluate_substitution_quality(self, test_pairs: List[tuple] = None) -> Dict[str, Any]:
        """Evaluate ingredient substitution suggestions"""
        print("🔄 Evaluating substitution quality...")
        
        if test_pairs is None:
            # Test some common substitution scenarios
            test_pairs = [
                ('butter', 'dairy'),
                ('tomato', 'fruit'),
                ('basil', 'spice'),
                ('chicken', 'meat'),
                ('flour', 'grain')
            ]
        
        results = {}
        
        for ingredient, expected_category in test_pairs:
            print(f"   Testing substitution for: {ingredient}")
            
            response = self.generate_response(
                f"What can I substitute for {ingredient.replace('_', ' ')} in cooking?",
                ingredient
            )
            
            # Check if response mentions category-appropriate substitutes
            category_appropriate = expected_category.lower() in response.lower()
            
            results[ingredient] = {
                'substitution_response': response,
                'expected_category': expected_category,
                'mentions_category': category_appropriate,
                'response_length': len(response.split()),
                'provides_explanation': 'because' in response.lower() or 'similar' in response.lower()
            }
        
        # Calculate metrics
        category_accuracy = np.mean([r['mentions_category'] for r in results.values()])
        explanation_rate = np.mean([r['provides_explanation'] for r in results.values()])
        
        summary = {
            'category_accuracy': category_accuracy,
            'explanation_rate': explanation_rate,
            'individual_results': results
        }
        
        print(f"   ✅ Category accuracy: {category_accuracy:.2%}")
        print(f"   ✅ Explanation rate: {explanation_rate:.2%}")
        
        return summary
    
    def evaluate_flavor_understanding(self, test_compounds: List[str] = None) -> Dict[str, Any]:
        """Evaluate chemical compound flavor understanding"""
        print("🧪 Evaluating flavor understanding...")
        
        if test_compounds is None:
            test_compounds = ['capsaicin', 'vanillin', 'limonene', 'ethanol', 'lactic_acid']
        
        results = {}
        
        for compound in test_compounds:
            print(f"   Testing: {compound}")
            
            response = self.generate_response(
                f"Analyze the flavor profile of {compound}",
                compound
            )
            
            # Check for flavor-related terms
            flavor_terms = ['sweet', 'salty', 'sour', 'bitter', 'umami', 'spicy', 'aromatic', 'heat', 'burn', 'citrus', 'vanilla', 'alcohol']
            mentions_flavors = [term for term in flavor_terms if term in response.lower()]
            
            results[compound] = {
                'flavor_response': response,
                'mentioned_flavors': mentions_flavors,
                'flavor_count': len(mentions_flavors),
                'chemical_context': 'compound' in response.lower() or 'chemical' in response.lower()
            }
        
        # Calculate metrics
        avg_flavor_mentions = np.mean([r['flavor_count'] for r in results.values()])
        chemical_context_rate = np.mean([r['chemical_context'] for r in results.values()])
        
        summary = {
            'avg_flavor_mentions': avg_flavor_mentions,
            'chemical_context_rate': chemical_context_rate,
            'individual_results': results
        }
        
        print(f"   ✅ Average flavor mentions: {avg_flavor_mentions:.1f}")
        print(f"   ✅ Chemical context rate: {chemical_context_rate:.2%}")
        
        return summary
    
    def run_interactive_demo(self):
        """Run interactive demo for testing the model"""
        print("\n🎮 Interactive FlavorGraph AI Demo")
        print("Type 'quit' to exit")
        print("-" * 40)
        
        while True:
            try:
                instruction = input("\n📝 Enter instruction: ").strip()
                if instruction.lower() == 'quit':
                    break
                
                input_text = input("📝 Enter input (optional): ").strip()
                
                print("\n🤖 Generating response...")
                response = self.generate_response(instruction, input_text)
                
                print(f"\n✨ Response:\n{response}")
                
            except KeyboardInterrupt:
                break
        
        print("\n👋 Demo ended!")
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation suite"""
        print("🎯 Running Full FlavorGraph AI Evaluation")
        print("=" * 50)
        
        results = {}
        
        # Evaluate different aspects
        results['ingredient_knowledge'] = self.evaluate_ingredient_knowledge()
        results['substitution_quality'] = self.evaluate_substitution_quality()
        results['flavor_understanding'] = self.evaluate_flavor_understanding()
        
        # Overall summary
        overall_score = (
            results['ingredient_knowledge']['category_mention_rate'] * 0.3 +
            results['substitution_quality']['category_accuracy'] * 0.4 +
            results['flavor_understanding']['chemical_context_rate'] * 0.3
        )
        
        results['overall_score'] = overall_score
        
        print(f"\n🎉 Overall FlavorGraph AI Score: {overall_score:.2%}")
        
        return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate FlavorGraph AI Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--base_model", type=str, help="Base model name (auto-detected if not provided)")
    parser.add_argument("--interactive", action="store_true", help="Run interactive demo")
    parser.add_argument("--output", type=str, help="Save evaluation results to file")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = FlavorGraphEvaluator(args.model_path, args.base_model)
    
    if args.interactive:
        evaluator.run_interactive_demo()
    else:
        # Run full evaluation
        results = evaluator.run_full_evaluation()
        
        # Save results if specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Results saved to {args.output}")

if __name__ == "__main__":
    main()
