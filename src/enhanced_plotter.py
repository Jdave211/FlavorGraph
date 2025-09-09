#!/usr/bin/env python3
"""
Enhanced Plotter with Flavor Region Annotations
Adds ingredient type annotations to identify flavor neighborhoods in embedding space
"""

import random
import time
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import itertools
import operator
import datetime
from collections import defaultdict
import chart_studio.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def analyze_flavor_regions(node_name2vec_tsne, node_name2is_hub, node_names, n_regions=8):
    """
    Analyze embedding space to identify flavor regions and expected ingredient types.
    """
    print("Analyzing flavor regions...")
    
    # Get coordinates and create mapping
    points = np.array([node_name2vec_tsne[i] for i in range(len(node_names))])
    
    # Cluster the embedding space into regions
    kmeans = KMeans(n_clusters=n_regions, random_state=42)
    region_labels = kmeans.fit_predict(points)
    cluster_centers = kmeans.cluster_centers_
    
    # Analyze each region
    region_analysis = {}
    for region_id in range(n_regions):
        region_mask = region_labels == region_id
        region_nodes = [node_names[i] for i in range(len(node_names)) if region_mask[i]]
        
        # Count ingredient types in this region
        type_counts = defaultdict(int)
        ingredient_examples = defaultdict(list)
        
        for node_name in region_nodes:
            if node_name in node_name2is_hub:
                node_type = node_name2is_hub[node_name]
                type_counts[node_type] += 1
                
                # Collect examples for each type
                if len(ingredient_examples[node_type]) < 5:
                    ingredient_examples[node_type].append(node_name)
        
        # Determine dominant type and create description
        if type_counts:
            dominant_type = max(type_counts.items(), key=lambda x: x[1])
            total_nodes = sum(type_counts.values())
            
            # Create region description
            description = f"Region {region_id + 1}"
            if dominant_type[0] == 'hub':
                description += "\nHub Ingredients"
                if 'food' in type_counts:
                    description += f"\n+ {type_counts['food']} Food Compounds"
            elif dominant_type[0] == 'food':
                description += "\nFood Compounds"
                if 'hub' in type_counts:
                    description += f"\n+ {type_counts['hub']} Hub Ingredients"
            elif dominant_type[0] == 'drug':
                description += "\nDrug Compounds"
            elif dominant_type[0] == 'no_hub':
                description += "\nNon-hub Ingredients"
            
            # Add examples
            examples = []
            for node_type, example_list in ingredient_examples.items():
                if example_list:
                    examples.extend(example_list[:2])  # Top 2 examples per type
            
            if examples:
                description += f"\nEx: {', '.join(examples[:3])}"
            
            region_analysis[region_id] = {
                'center': cluster_centers[region_id],
                'description': description,
                'dominant_type': dominant_type[0],
                'type_counts': dict(type_counts),
                'total_nodes': total_nodes,
                'examples': examples
            }
    
    return region_analysis

def create_flavor_annotations(region_analysis, points_range):
    """
    Create plotly annotations for flavor regions.
    """
    annotations = []
    
    for region_id, info in region_analysis.items():
        center = info['center']
        description = info['description']
        
        # Position annotation slightly offset from center
        x_pos = center[0]
        y_pos = center[1]
        
        # Choose color based on dominant type
        color_map = {
            'hub': '#ff7f0e',      # Orange
            'food': '#2ca02c',     # Green  
            'drug': '#d62728',     # Red
            'no_hub': '#1f77b4'    # Blue
        }
        
        bg_color = color_map.get(info['dominant_type'], '#808080')
        
        annotations.append(go.layout.Annotation(
            x=x_pos,
            y=y_pos,
            xref="x",
            yref="y",
            text=description,
            showarrow=True,
            font=dict(
                family="Arial, sans-serif",
                size=10,
                color="white"
            ),
            align="center",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=bg_color,
            ax=20,
            ay=-20,
            bordercolor=bg_color,
            borderwidth=2,
            borderpad=4,
            bgcolor=bg_color,
            opacity=0.8
        ))
    
    return annotations

def plot_embedding_with_regions(args, graph, mode=None):
    """
    Enhanced plot embedding with flavor region annotations.
    """
    print("\nPlot Embedding with Flavor Regions...")
    node2node_name = {}
    node_name2is_hub = {}
    for node in graph.nodes():
        node_info = graph.nodes[node]
        node_name = node_info['name']
        node2node_name[node] = node_name
        node_name2is_hub[node_name] = node_info['is_hub']

    # Load embeddings (same logic as original plotter)
    if args.idx_embed == 'Node2vec':
        file = "{}{}-embedding_{}-deepwalk_{}-dim_{}-initial_lr_{}-window_size_{}-iterations_{}-min_count.pickle".format(
                            args.output_path, args.idx_embed, args.idx_metapath, args.dim, args.initial_lr, args.window_size, args.iterations, args.min_count)
    else:
        file = "{}{}-embedding_{}_300-dim_{}-initial_lr_{}-window_size_{}-iterations_{}-min_count-_{}-isCSP_{}-CSPcoef.pickle".format(
                            args.output_path, args.idx_embed, args.idx_metapath, args.initial_lr, args.window_size, args.iterations, args.min_count, args.CSP_train, args.CSP_coef)

    with open(file, "rb") as pickle_file:
        vectors = pickle.load(pickle_file)
    
    node_name2vec = {}
    for node in vectors:
        node_name = node2node_name[int(node)]
        node_name2vec[node_name] = vectors[node]

    # TSNE
    from plotter import load_TSNE
    node_name2vec_tsne = load_TSNE(node_name2vec, dim=2)
    
    # Get node names in order
    node_names = list(node_name2vec.keys())
    
    # Analyze flavor regions
    region_analysis = analyze_flavor_regions(node_name2vec_tsne, node_name2is_hub, node_names)
    
    # Create enhanced plot with region annotations
    save_path = file.replace('.pickle', '_with_regions.html')
    plot_category_with_regions(node_name2vec, node_name2vec_tsne, save_path, node2node_name, node_name2is_hub, region_analysis, True)
    
    return region_analysis

def plot_category_with_regions(node2vec, node2vec_tsne, path, node2name, node2is_hub, region_analysis, withLegends=False):
    """
    Enhanced plotting function with flavor region annotations.
    """
    labels = []
    for label in node2vec:
        labels.append(label)

    if withLegends:
        categories = []
        for label in labels:
            try:
                if node2is_hub[label] == 'hub':
                    categories.append('Hub_Ingredient')
                elif node2is_hub[label] == 'no_hub':
                    categories.append('Non_hub_Ingredient')
                elif node2is_hub[label] == 'food':
                    categories.append('Food_like_Compound')
                elif node2is_hub[label] == 'drug':
                    categories.append('Drug_like_Compound')
                else:
                    print(label)
            except KeyError:
                categories.append("None")

        category2color = {
            'Hub_Ingredient': sns.xkcd_rgb["orange"],
            'Non_hub_Ingredient': sns.xkcd_rgb["goldenrod"],
            'Food_like_Compound': sns.xkcd_rgb["green"],
            'Drug_like_Compound': sns.xkcd_rgb["pink"],
            'None': sns.xkcd_rgb["black"]
        }
        
        category2marker = {
            'Hub_Ingredient': 'diamond-x',
            'Non_hub_Ingredient': 'square',
            'Food_like_Compound': 'circle',
            'Drug_like_Compound': 'circle'
        }
        
        category2size = {
            'Hub_Ingredient': 14,
            'Non_hub_Ingredient': 8,
            'Food_like_Compound': 8,
            'Drug_like_Compound': 9
        }

        # Create plot with region annotations
        make_plot_with_labels_legends_and_regions(
            path.replace('.html', ''),
            node2vec_tsne,
            labels,
            {},  # label_to_plot - empty for now
            categories,
            ['Non_hub_Ingredient', 'Food_like_Compound', 'Drug_like_Compound', 'Hub_Ingredient'],
            category2color,
            category2marker,
            category2size,
            lambda x: x.replace('_', ' '),
            region_analysis,
            publish=False
        )

def make_plot_with_labels_legends_and_regions(name, points, labels, label_to_plot, legend_labels, legend_order, legend_label_to_color, legend_label_marker, legend_label_size, pretty_legend_label, region_analysis, publish):
    """
    Enhanced plotting function that includes flavor region annotations.
    """
    lst = zip(points, labels, legend_labels)
    full = sorted(lst, key=lambda x: x[2])
    traces = []
    
    # Create scatter traces for each category (same as original)
    for legend_label, group in itertools.groupby(full, lambda x: x[2]):
        group_points = []
        group_labels = []
        for tup in group:
            point, label, _ = tup
            group_points.append(point)
            group_labels.append(label)
            
        group_points = np.stack(group_points)
        traces.append(go.Scattergl(
            x=group_points[:, 0],
            y=group_points[:, 1],
            mode='markers',
            marker=dict(
                symbol=legend_label_marker[legend_label],
                color=legend_label_to_color[legend_label],
                size=legend_label_size[legend_label],
                opacity=1,
                line=dict(width=0.5)
            ),
            text=['{} ({})'.format(label, pretty_legend_label(legend_label)) for label in group_labels],
            hoverinfo='text',
            name=legend_label
        ))

    # Create flavor region annotations
    region_annotations = create_flavor_annotations(region_analysis, points)

    layout = go.Layout(
        title="FlavorGraph Embeddings with Flavor Regions",
        xaxis=dict(
            autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False
        ),
        yaxis=dict(
            autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False
        ),
        annotations=region_annotations,
        showlegend=True
    )

    fig = go.Figure(data=traces, layout=layout)
    
    if publish:
        plotter = py.iplot
    else:
        plotter = offline.plot
    
    plotter(fig, filename=name + '_with_regions.html')
    print(f"Enhanced plot saved as: {name}_with_regions.html")

# Test function
if __name__ == "__main__":
    print("Enhanced plotter with flavor region annotations loaded!")
