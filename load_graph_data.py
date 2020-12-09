# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 18:03:29 2020

@author: Zach Nguyen

Requirement:
    !pip install torch-cluster==latest+cu101 torch-scatter==latest+cu101 torch-sparse==latest+cu101 -f https://s3.eu-central-1.amazonaws.com/pytorch-geometric.com/whl/torch-1.7.0.html
    !pip install torch-geometric
    
    *** REMEMBER TO PLACE DATA OBJECTS IN data folder****
"""

import pandas as pd
import numpy as np
import json, gzip, pickle
import os
import warnings
warnings.filterwarnings("ignore")

path = os.getcwd()
code_dir = os.path.join(path, 'src/')
data_dir = os.path.join(path, 'data/')

def construct_graph_dataloader():
    patent_df, citation_df, labels, node_features = load_files(os.path.join(data_dir, 'processed_text_df_zach.csv'),
                                                    os.path.join(data_dir, 'cit-Patents.txt.gz'),
                                                    os.path.join(data_dir, 'labels.txt'),
                                                    os.path.join(data_dir, 'tfidf_trun_vec.pickle')
                                                    )
    labels = [label - 1 for label in labels]
    patent_ids_list, citations_df_final = filter_citations(citation_df=citation_df,
                                                           patent_df=patent_df,
                                                           )
    patent2node, node2patent, citations_df_final = create_mapping(citations_df_final)
    train_mask, val_mask = create_train_val_mask(labels)
    graph_data = read_citation(citations_df_final, node_features, labels, train_mask, val_mask)
    display_graph_info(graph_data)
    return graph_data, patent2node, node2patent
    
    
def load_files(patent_data_path, citation_data_path, labels_path, tfidf_features_path):
    """ Load all files used to construct graph data
    arg:
        patent_data_path: data path with .csv (USE CASE currently only takes 1990-1999 data)
        citation_data_path: data path with .gz
        labels: data path for labels with .txt
    return:
        patent dataframe, citation dataframe and category label dataframe
    """
    # Load necessary file to contruct graph data
    filename = patent_data_path
    patent_df = pd.read_csv(filename, index_col=0)
    with gzip.open(os.path.join(data_dir, 'cit-Patents.txt.gz'),'rt') as file:
        citation_df = pd.read_csv(file, sep='\t', header=None, skiprows=4, dtype=np.int64, names=['source', 'target'])
    with open(labels_path, 'r') as outfile:
        labels = json.load(outfile)
    node_features = pickle.load(open(tfidf_features_path, "rb"))
    return patent_df, citation_df, labels, node_features

    
    # Filter citations to contruct graph
def filter_citations(patent_df, citation_df):
    """ Filter citations based on patent ID 
    arg: patent_df: a pandas dataframe with patent information,
        citation_df: a pandas dataframe with source-target citation
    return:
        a list of patent id used, this will be used to assign node indices
        filtered citation dataframe
    """
    print(f'Current citation data has {len(citation_df)} edges')
    patent_ids = list(patent_df.PID)
    citation_final_df = citation_df[(citation_df.source.isin(patent_ids) & citation_df.target.isin(patent_ids))]
    print(f'New citation data has {len(citation_final_df)} edges')
    return patent_ids, citation_final_df


def create_mapping(citations_df_final):
    """ Create a mapping of patent number to ids for graph - 1990 - 1999
    arg:
        citation_df_final: pandas dataframe of citation
    return:
        patent2node, node2patent and dataframe of final citation.
    """
    sample_patents = list(set(citations_df_final.source).union(citations_df_final.target))
    print(f'There are {len(sample_patents)} sampled patents in the citation network ...')
    patent2node = {}
    for index in range(len(sample_patents)):
        patent2node[sample_patents[index]] = index
    node2patent = {value: key for key, value in patent2node.items()}
    citations_df_final['source'] = citations_df_final['source'].map(patent2node)
    citations_df_final['target'] = citations_df_final['target'].map(patent2node)
    citations_df_final = citations_df_final.reset_index(drop=True)
    return patent2node, node2patent, citations_df_final

def create_train_val_mask(labels):
    """ Create train val mask based on the label array"""
    np.random.seed(42)
    TRAIN_SIZE = 0.80
    train_mask = np.random.rand(len(labels)) < TRAIN_SIZE
    val_mask = [not elem for elem in train_mask]
    print(len(train_mask), len(val_mask))
    print(sum(train_mask), sum(val_mask))
    return train_mask, val_mask

    
def read_citation(sampled_dataset, node_features, labels, train_mask, val_mask):
    """ READ CITATIONS INTO GRAPH """
    import torch
    from torch_sparse import coalesce
    from torch_geometric.data import Data
    edge_index = torch.from_numpy(sampled_dataset.values).t()
    num_nodes = edge_index.max().item() + 1
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    data = Data(x= node_features, edge_index=edge_index, num_nodes=num_nodes, y=torch.LongTensor(labels))
    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
    
def display_graph_info(data):
    """ print graph attributes """
    print("Number of nodes: ",data.num_nodes)
    print("Number of edges:",data.num_edges)
    print("Is this an undirected graph?",data.is_undirected())
    print("Number of features per node (length of feature vector)",data.num_node_features,"\n")
    print("Number of features per edge (length of feature vector)",data.num_edge_features,"\n")
    
