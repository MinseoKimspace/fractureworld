import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def component_labels(cell_count, edges, cuts):
    edges = np.asarray(edges, dtype=np.int64)
    cuts = np.asarray(cuts, dtype=bool)
    if edges.shape != (len(cuts), 2):
        raise ValueError("edges and cuts must have matching shapes")
    if cell_count < 1 or (edges < 0).any() or (edges >= cell_count).any():
        raise ValueError("invalid cell indices")
    if len(edges) and (edges[:, 0] == edges[:, 1]).any():
        raise ValueError("self edges are not material interfaces")
    intact = edges[~cuts]
    graph = coo_matrix(
        (np.ones(len(intact)), (intact[:, 0], intact[:, 1])),
        shape=(cell_count, cell_count),
    ).tocsr()
    return connected_components(graph, directed=False)[1]
