import networkx as nx
import scipy.sparse as sp
from sklearn.preprocessing import MaxAbsScaler
import os
import numpy as np
import pandas as pd



def normalize_features(features):
    scaler = MaxAbsScaler()
    normalized_features = scaler.fit_transform(features)
    return normalized_features



def load_npz_dataset(file_name):
    """Load a graph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    graph : dict
        Dictionary that contains:
            * 'A' : The adjacency matrix in sparse matrix format
            * 'X' : The attribute matrix in sparse matrix format
            * 'z' : The ground truth class labels
            * Further dictionaries mapping node, class and attribute IDs

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        edge_index = loader['adj_indices'].copy()
        A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                           loader['adj_indptr']), shape=loader['adj_shape'])

        X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])






        # 归一化特征矩阵
        scaler = MaxAbsScaler()
        X = scaler.fit_transform(X)



        graph = {
            'A': A,
            'X': X,
        }
        return graph




def load_data(pathway="../demo/", dataset='Non-Specific Dataset mDC 1000+ '):
    os.makedirs(pathway, exist_ok=True)
    dataset_path = os.path.join(pathway, '{}.npz'.format(dataset))
    g = load_npz_dataset(dataset_path)
    adj, feature = g['A'], g['X']

    return adj, feature,dataset
