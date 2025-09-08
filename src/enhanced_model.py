#!/usr/bin/env python3
"""
Enhanced FlavorGraph Model with Chemical Compound Flavor Integration
Incorporates flavor profiles directly into the embedding training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import pickle
import pandas as pd
import numpy as np
import json
import os
import re

class FlavorEnhancedSkipGramModel(nn.Module):
    def __init__(self, emb_size, emb_dimension, nodes_file=None, compound_flavor_file=None):
        super(FlavorEnhancedSkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        # Define base and derived flavor categories
        self.base_flavor_categories = [
            'salt', 'fat', 'acid', 'heat', 'umami', 'sweet', 'bitter', 'aromatic'
        ]
        # Derived sub-flavors to increase separation/granularity
        self.derived_flavor_categories = [
            'citrus',            # limonene, citral
            'mint',              # menthol, menthone
            'floral',            # linalool, geraniol
            'pine_woody',        # pinene, terpinene
            'smoky_roasted',     # guaiacol, phenols
            'green_herbaceous',  # hexenal, hexenol
            'fruity_ester',      # acetate/esters
            'sulfur_allium',     # thiol, sulfide
            'caramel_browned',   # maltol, furans
            'vanilla_spicy',     # vanillin, eugenol
            'earthy_musty',      # geosmin
            'buttery_dairy'      # butyric, diacetyl
        ]
        self.flavor_categories = self.base_flavor_categories + self.derived_flavor_categories
        self.flavor_dimension = len(self.flavor_categories)
        
        # Load flavor data
        self.flavor_profiles = self.load_flavor_data(nodes_file, compound_flavor_file)
        
        # Core embeddings
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        
        # Flavor enhancement layers
        self.flavor_encoder = nn.Linear(self.flavor_dimension, emb_dimension // 4)
        self.flavor_fusion = nn.Linear(emb_dimension + emb_dimension // 4, emb_dimension)
        
        # Initialize embeddings
        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)
        
        # Initialize flavor layers
        init.xavier_uniform_(self.flavor_encoder.weight)
        init.xavier_uniform_(self.flavor_fusion.weight)
        
        print(f"FlavorEnhanced model initialized:")
        print(f"- Embeddings: {emb_size} x {emb_dimension}")
        print(f"- Flavor profiles loaded: {len(self.flavor_profiles)}")
        
    def load_flavor_data(self, nodes_file, compound_flavor_file):
        """Load and process flavor profile data."""
        flavor_profiles = {}
        
        if not nodes_file or not compound_flavor_file or not os.path.exists(compound_flavor_file):
            print("Warning: Flavor data not found. Using default neutral profiles.")
            return flavor_profiles
            
        try:
            # Load nodes to get node_id mappings
            nodes_df = pd.read_csv(nodes_file)
            node_id_to_name = dict(zip(nodes_df['node_id'], nodes_df.get('cleaned_name', nodes_df['name'])))
            
            # Load compound flavor mappings
            compound_df = pd.read_csv(compound_flavor_file)
            
            # Create flavor profiles for compounds
            for _, row in compound_df.iterrows():
                node_id = row['node_id']
                # Base 8 flavors
                base_vector = np.array([
                    row.get('salt', 0), row.get('fat', 0), row.get('acid', 0), 
                    row.get('heat', 0), row.get('umami', 0), row.get('sweet', 0), 
                    row.get('bitter', 0), row.get('aromatic', 0)
                ], dtype=float)

                # Derived sub-flavors via name heuristics
                name_blob = f"{row.get('compound', '')} {row.get('original_name', '')} {row.get('primary_flavor', '')}"
                name_blob = str(name_blob).lower()
                derived_vector = np.zeros(len(self.derived_flavor_categories), dtype=float)

                # Keyword → index mapping
                keyword_map = [
                    (['limonene', 'citral', 'citrus', 'lemon', 'orange'], 'citrus'),
                    (['menthol', 'menthone', 'menthyl', 'mint'], 'mint'),
                    (['linalool', 'geraniol', 'nerol', 'rose', 'floral'], 'floral'),
                    (['pinene', 'terpinene', 'pine', 'cedrene'], 'pine_woody'),
                    (['guaiacol', 'phenol', 'smoky', 'roasted', 'pyrazine'], 'smoky_roasted'),
                    (['hexenal', 'hexenol', 'leaf', 'green', 'herb'], 'green_herbaceous'),
                    (['acetate', 'ester', 'ethyl', 'isoamyl'], 'fruity_ester'),
                    (['thiol', 'sulfide', 'sulphide', 'sulfur', 'alliin', 'allyl'], 'sulfur_allium'),
                    (['maltol', 'furfural', 'caramel', 'browned', 'maillard'], 'caramel_browned'),
                    (['vanillin', 'eugenol', 'clove', 'vanilla'], 'vanilla_spicy'),
                    (['geosmin', 'm earthy', 'earthy', 'humus'], 'earthy_musty'),
                    (['butyric', 'diacetyl', 'butter', 'dairy'], 'buttery_dairy'),
                ]

                derived_index = {name: i for i, name in enumerate(self.derived_flavor_categories)}

                for keys, cat in keyword_map:
                    if any(k in name_blob for k in keys):
                        idx = derived_index[cat]
                        # Weight: use flavor_strength if available else 1.0
                        strength = row.get('flavor_strength', 1.0)
                        try:
                            strength = float(strength) if not pd.isna(strength) else 1.0
                        except Exception:
                            strength = 1.0
                        derived_vector[idx] = max(derived_vector[idx], min(max(strength, 0.1), 1.0))

                # If base vector appears uniform/neutral, slightly upweight derived cues to prevent collapse
                if np.allclose(base_vector, np.full_like(base_vector, base_vector.mean())) and derived_vector.sum() > 0:
                    derived_vector = np.clip(derived_vector * 1.25, 0.0, 1.0)

                full_vector = np.concatenate([base_vector, derived_vector])
                flavor_profiles[node_id] = full_vector
            
            # Load ingredient flavor profiles if available
            ingredient_flavor_file = compound_flavor_file.replace('compound_flavor_mappings.csv', 'ingredient_flavor_profiles.csv')
            if os.path.exists(ingredient_flavor_file):
                ingredient_df = pd.read_csv(ingredient_flavor_file)
                name_to_node_id = {v: k for k, v in node_id_to_name.items()}
                
                for _, row in ingredient_df.iterrows():
                    ingredient_name = row['ingredient']
                    if ingredient_name in name_to_node_id:
                        node_id = name_to_node_id[ingredient_name]
                        base_vector = np.array([
                            row.get('salt', 0), row.get('fat', 0), row.get('acid', 0), 
                            row.get('heat', 0), row.get('umami', 0), row.get('sweet', 0), 
                            row.get('bitter', 0), row.get('aromatic', 0)
                        ], dtype=float)
                        # No reliable name heuristics for ingredients here; keep derived zeros
                        derived_vector = np.zeros(len(self.derived_flavor_categories), dtype=float)
                        flavor_vector = np.concatenate([base_vector, derived_vector])
                        # Only use if not all zeros
                        if np.sum(flavor_vector) > 0:
                            flavor_profiles[node_id] = flavor_vector
                            
        except Exception as e:
            print(f"Error loading flavor data: {e}")
            
        return flavor_profiles
    
    def get_flavor_enhanced_embedding(self, node_ids, embedding_layer):
        """Get embeddings enhanced with flavor information."""
        # Get base embeddings
        base_embeddings = embedding_layer(node_ids)
        
        # Get flavor profiles for these nodes
        batch_size = node_ids.size(0)
        flavor_vectors = torch.zeros(batch_size, self.flavor_dimension).to(node_ids.device)
        
        for i, node_id in enumerate(node_ids):
            node_id_int = node_id.item()
            if node_id_int in self.flavor_profiles:
                flavor_vectors[i] = torch.tensor(self.flavor_profiles[node_id_int], 
                                               dtype=torch.float32, device=node_ids.device)
        
        # Encode flavor information
        flavor_encoded = self.flavor_encoder(flavor_vectors)
        flavor_encoded = F.relu(flavor_encoded)
        
        # Fuse base embedding with flavor information
        combined = torch.cat([base_embeddings, flavor_encoded], dim=1)
        enhanced_embedding = self.flavor_fusion(combined)
        enhanced_embedding = F.tanh(enhanced_embedding)  # Bounded activation
        
        return enhanced_embedding
    
    def forward(self, pos_u, pos_v, neg_v):
        # Get flavor-enhanced embeddings
        emb_u = self.get_flavor_enhanced_embedding(pos_u, self.u_embeddings)
        emb_v = self.get_flavor_enhanced_embedding(pos_v, self.v_embeddings)
        emb_neg_v = self.get_flavor_enhanced_embedding(neg_v.view(-1), self.v_embeddings)
        emb_neg_v = emb_neg_v.view(neg_v.size(0), neg_v.size(1), -1)
        
        # Standard skip-gram loss computation
        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)
        
        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)
        
        return torch.mean(score + neg_score)
    
    def save_embedding(self, id2word, file_name):
        """Save flavor-enhanced embeddings."""
        embed_dict = dict()
        
        # Get all node IDs
        all_node_ids = torch.arange(self.emb_size).to(next(self.parameters()).device)
        
        # Get enhanced embeddings for all nodes
        with torch.no_grad():
            enhanced_embeddings = self.get_flavor_enhanced_embedding(all_node_ids, self.u_embeddings)
            enhanced_embeddings = enhanced_embeddings.cpu().numpy()
        
        # Map to ingredient names
        for node_id, name in id2word.items():
            if node_id < len(enhanced_embeddings):
                embed_dict[name] = enhanced_embeddings[node_id]
        
        # Save enhanced embeddings
        with open(file_name, "wb") as handle:
            pickle.dump(embed_dict, handle)
            
        print(f"Saved {len(embed_dict)} flavor-enhanced embeddings to {file_name}")
        
        # Also save flavor analysis
        flavor_analysis_file = file_name.replace('.pickle', '_flavor_analysis.json')
        self.save_flavor_analysis(embed_dict, flavor_analysis_file)
    
    def save_flavor_analysis(self, embeddings, analysis_file):
        """Save flavor analysis of the embeddings."""
        analysis = {
            'total_embeddings': len(embeddings),
            'embedding_dimension': len(next(iter(embeddings.values()))),
            'flavor_enhanced_nodes': len(self.flavor_profiles),
            'flavor_categories': self.flavor_categories,
            'num_base_categories': len(self.base_flavor_categories),
            'num_derived_categories': len(self.derived_flavor_categories)
        }
        
        # Sample flavor profiles
        sample_profiles = {}
        count = 0
        for node_id, flavor_vector in self.flavor_profiles.items():
            if count < 10:  # Sample first 10
                sample_profiles[str(node_id)] = flavor_vector.tolist()
                count += 1
        
        analysis['sample_flavor_profiles'] = sample_profiles
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Saved flavor analysis to {analysis_file}")


def create_enhanced_model(emb_size, emb_dimension, nodes_file=None):
    """Create a flavor-enhanced model."""
    compound_flavor_file = './input/compound_flavors/compound_flavor_mappings.csv'
    
    if os.path.exists(compound_flavor_file):
        print("Creating FlavorEnhanced model with chemical compound data...")
        return FlavorEnhancedSkipGramModel(emb_size, emb_dimension, nodes_file, compound_flavor_file)
    else:
        print("Compound flavor data not found. Creating standard model...")
        from model import SkipGramModel
        return SkipGramModel(emb_size, emb_dimension)


if __name__ == "__main__":
    # Test the enhanced model
    print("Testing FlavorEnhanced model...")
    
    # Mock parameters
    emb_size = 100
    emb_dimension = 64
    nodes_file = './input/cleaned/nodes_cleaned_basic.csv'
    
    model = create_enhanced_model(emb_size, emb_dimension, nodes_file)
    print(f"Model created: {type(model).__name__}")
    
    # Test forward pass
    pos_u = torch.randint(0, emb_size, (32,))
    pos_v = torch.randint(0, emb_size, (32,))
    neg_v = torch.randint(0, emb_size, (32, 5))
    
    loss = model(pos_u, pos_v, neg_v)
    print(f"Test loss: {loss.item():.4f}")
    print("FlavorEnhanced model test completed!")
