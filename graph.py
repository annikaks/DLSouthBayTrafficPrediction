#for gnn
from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch

def build_knn_graph(latitudes, longitudes, k=5):
    """
    Builds a k-NN graph in (edge_index) format for PyTorch Geometric.
    """
    coords = np.stack([latitudes, longitudes], axis=1)
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    edges = []
    N = len(latitudes)

    for i in range(N):
        for j in indices[i][1:]:   # skip self-loop
            edges.append([i, j])
            edges.append([j, i])  # undirected

    edge_index = torch.tensor(edges, dtype=torch.long).t()  # (2, E)
    return edge_index
