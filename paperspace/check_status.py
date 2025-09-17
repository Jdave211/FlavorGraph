#!/usr/bin/env python3
"""
FlavorGraph AI Training Status Checker
Verifies all components are ready for Paperspace training
"""

import json
import pickle
from pathlib import Path
import pandas as pd

def check_status():
    print("🎯 FlavorGraph AI Training Status Check")
    print("=" * 50)
    
    base_path = Path("/Users/davejaga/Desktop/Startups/FlavorGraph")
    paperspace_path = base_path / "paperspace"
    training_data_path = paperspace_path / "training_data"
    
    status = {"ready": True, "issues": []}
    
    # Check base FlavorGraph data
    print("📊 Checking FlavorGraph Data...")
    
    required_files = [
        "input/cleaned/nodes_cleaned_basic.csv",
        "input/edges_191120.csv", 
        "input/dict_ingr2cate - Top300+FDB400+HyperFoods104=616.csv",
        "input/compound_flavors/compound_flavor_mappings.csv"
    ]
    
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path}")
            status["ready"] = False
            status["issues"].append(f"Missing: {file_path}")
    
    # Check for embeddings
    embedding_files = list((base_path / "output").glob("*embedding*.pickle"))
    if embedding_files:
        latest_embedding = max(embedding_files, key=lambda p: p.stat().st_mtime)
        print(f"   ✅ Latest embedding: {latest_embedding.name}")
    else:
        print("   ❌ No embedding files found")
        status["ready"] = False
        status["issues"].append("No FlavorGraph embeddings found")
    
    # Check Paperspace setup
    print("\n🚀 Checking Paperspace Setup...")
    
    paperspace_files = [
        "setup_paperspace.sh",
        "requirements.txt", 
        "prepare_training_data.py",
        "train_flavor_model.py",
        "evaluate_model.py",
        "configs/llama7b_lora.yaml",
        "configs/mistral7b_qlora.yaml"
    ]
    
    for file_path in paperspace_files:
        full_path = paperspace_path / file_path
        if full_path.exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path}")
            status["ready"] = False
            status["issues"].append(f"Missing: paperspace/{file_path}")
    
    # Check training data
    print("\n🧠 Checking Training Data...")
    
    if training_data_path.exists():
        metadata_file = training_data_path / "dataset_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            print(f"   ✅ Total training examples: {metadata['total_examples']:,}")
            print(f"   ✅ Ingredient examples: {metadata['ingredient_examples']:,}")
            print(f"   ✅ Flavor examples: {metadata['flavor_examples']:,}")
            print(f"   ✅ Substitution examples: {metadata['substitution_examples']:,}")
            print(f"   ✅ Recipe examples: {metadata['recipe_examples']:,}")
            print(f"   ✅ Total embeddings: {metadata['total_embeddings']:,}")
            print(f"   ✅ Embedding dimensions: {metadata['embedding_dimensions']}")
            
            # Check individual files
            training_files = [
                "combined_training.jsonl",
                "ingredient_knowledge.jsonl", 
                "flavor_analysis.jsonl",
                "substitution_pairs.jsonl",
                "recipe_analysis.jsonl",
                "embeddings_reference.pkl"
            ]
            
            for file_name in training_files:
                file_path = training_data_path / file_name
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    print(f"   ✅ {file_name} ({size_mb:.1f} MB)")
                else:
                    print(f"   ❌ {file_name}")
                    status["ready"] = False
                    status["issues"].append(f"Missing training file: {file_name}")
        else:
            print("   ❌ dataset_metadata.json not found")
            status["ready"] = False
            status["issues"].append("Training data metadata missing")
    else:
        print("   ❌ training_data directory not found")
        status["ready"] = False
        status["issues"].append("Training data directory missing")
    
    # Hardware recommendations
    print("\n💻 Hardware Recommendations...")
    if training_data_path.exists() and (training_data_path / "dataset_metadata.json").exists():
        with open(training_data_path / "dataset_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        total_examples = metadata['total_examples']
        
        print(f"   📊 Training examples: {total_examples:,}")
        
        if total_examples > 8000:
            print("   🔥 Recommended: A100 40GB+ (Llama 7B LoRA)")
            print("   ⚡ Alternative: RTX4000+ (Mistral 7B QLoRA)")
        elif total_examples > 5000:
            print("   ⚡ Recommended: RTX4000+ (Mistral 7B QLoRA)")
            print("   💡 Alternative: RTX3080+ (smaller batch size)")
        else:
            print("   💡 Minimum: RTX3080+ should work")
        
        estimated_time = total_examples / 1000 * 2  # Rough estimate: 2 hours per 1k examples
        print(f"   ⏱️  Estimated training time: {estimated_time:.1f} hours")
    
    # Final status
    print(f"\n🎉 Status Summary")
    print("=" * 30)
    
    if status["ready"]:
        print("✅ READY FOR PAPERSPACE TRAINING!")
        print("\n🚀 Next steps:")
        print("   1. Upload FlavorGraph folder to Paperspace")
        print("   2. Run: ./paperspace/setup_paperspace.sh")
        print("   3. Choose config and start training:")
        print("      python3 train_flavor_model.py --config configs/llama7b_lora.yaml")
    else:
        print("❌ NOT READY - Issues found:")
        for issue in status["issues"]:
            print(f"   • {issue}")
        print("\n🔧 Fix the issues above before training")
    
    return status

if __name__ == "__main__":
    check_status()
