import torch
from parser import parameter_parser

from utils import tab_printer, graph_reader, evaluate
from dataloader import DataReader, DatasetLoader
from graph2vec import Metapath2Vec, Node2Vec
from plotter import plot_embedding
from enhanced_plotter import plot_embedding_with_regions

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    """
    Parsing command line parameters.
    Reading data, embedding base graph, creating persona graph and learning a splitter.
    Saving the persona mapping and the embedding.
    """
    args = parameter_parser()
    torch.manual_seed(args.seed)
    tab_printer(args)

    """
    1. read graph and load as torch dataset
    """
    graph, graph_ingr_only = graph_reader(args.input_nodes, args.input_edges)

    """
    2. Metapath2vec with MetaPathWalker - Ingredient-Ingredient / Ingredient-Food-like Compound / Ingredient-Drug-like Compound
    """
    
    if args.idx_embed == 'Node2vec':
        node2vec = Node2Vec(args, graph)
        node2vec.train()

    else:
        metapath2vec = Metapath2Vec(args, graph)
        metapath2vec.train()

    """
    3. Plot your embedding if you like
    """
    plot_embedding(args, graph)
    
    # Create enhanced plot with flavor regions if using flavor enhancement
    if hasattr(args, 'flavor_enhanced') and args.flavor_enhanced:
        print("\nCreating enhanced plot with flavor regions...")
        region_analysis = plot_embedding_with_regions(args, graph)
        print(f"Identified {len(region_analysis)} flavor regions in embedding space")

    """
    4. Evaluate Node Classification & Node Clustering
    """
    evaluate(args, graph)

if __name__ == "__main__":
    main()
