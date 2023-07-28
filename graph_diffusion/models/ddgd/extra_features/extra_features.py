"""
This file contains the code for computing the features. These are hardcoded computations that are added to the the input of the graph transformer. It uses functions defined the `eigen_features.py` and `cycle_features.py` files.
"""
import jax.numpy as np
import ipdb
from .cycle_features import node_cycle_features
from .eigen_features import eigen_features
from ....shared.graph import graph_distribution as gd
from .container import Container

GraphDistribution = gd.GraphDistribution


def compute(
    graph: GraphDistribution,
    max_n_nodes: int,
    features_type="all",
) -> Container:
    n = np.sum(graph.nodes_mask, axis=1, keepdims=True) / max_n_nodes
    x_cycles, y_cycles = node_cycle_features(graph)  # (bs, n_cycles)

    if features_type == "cycles":
        E = graph.edges
        extra_edge_attr = np.zeros((*E.shape[:-1], 0), dtype=E.dtype)
        return Container(nodes=x_cycles, edges=extra_edge_attr, y=np.hstack((n, y_cycles)))

    elif features_type == "eigenvalues":
        eigenfeatures = eigen_features(features_type, graph)
        E = graph.edges
        extra_edge_attr = np.zeros((*E.shape[:-1], 0), dtype=E.dtype)
        n_components, batched_eigenvalues = eigenfeatures  # (bs, 1), (bs, 10)
        return Container(
            nodes=x_cycles,
            edges=extra_edge_attr,
            y=np.hstack((n, y_cycles, n_components, batched_eigenvalues)),
        )
    elif features_type == "all":
        eigenfeatures = eigen_features(features_type, graph)
        E = graph.edges
        extra_edge_attr = np.zeros((*E.shape[:-1], 0), dtype=E.dtype)
        (
            n_components,
            batched_eigenvalues,
            nonlcc_indicator,
            k_lowest_eigvec,
        ) = eigenfeatures  # (bs, 1), (bs, 10),
        # (bs, n, 1), (bs, n, 2)
        y = np.concatenate((n, y_cycles, n_components, batched_eigenvalues), axis=-1)
        x = np.concatenate((x_cycles, nonlcc_indicator, k_lowest_eigvec), axis=-1)
        return Container(
            edges=extra_edge_attr,
            nodes=x,
            y=y,
        )
    else:
        raise ValueError(f"Features type {features_type} not implemented")
