import jax.numpy as np
from .cycle_features import node_cycle_features
from .eigen_features import eigen_features
from .utils import NoisyData, Graph


def extra_features(noisy_data: NoisyData, features_type="all", max_n_nodes=100):
    n = np.sum(noisy_data.graph.mask, axis=1, keepdims=True) / max_n_nodes
    x_cycles, y_cycles = node_cycle_features(noisy_data)  # (bs, n_cycles)

    if features_type == "cycles":
        E = noisy_data.graph.e
        extra_edge_attr = np.zeros((*E.shape[:-1], 0), dtype=E.dtype)
        return Graph(x=x_cycles, e=extra_edge_attr, y=np.hstack((n, y_cycles)))

    elif features_type == "eigenvalues":
        eigenfeatures = eigen_features(features_type, noisy_data)
        E = noisy_data.graph.e
        extra_edge_attr = np.zeros((*E.shape[:-1], 0), dtype=E.dtype)
        n_components, batched_eigenvalues = eigenfeatures  # (bs, 1), (bs, 10)
        return Graph(
            x=x_cycles,
            e=extra_edge_attr,
            y=np.hstack((n, y_cycles, n_components, batched_eigenvalues)),
        )
    elif features_type == "all":
        eigenfeatures = eigen_features(features_type, noisy_data)
        E = noisy_data.graph.e
        extra_edge_attr = np.zeros((*E.shape[:-1], 0), dtype=E.dtype)
        (
            n_components,
            batched_eigenvalues,
            nonlcc_indicator,
            k_lowest_eigvec,
        ) = eigenfeatures  # (bs, 1), (bs, 10),
        # (bs, n, 1), (bs, n, 2)

        return Graph(
            x=np.concatenate((x_cycles, nonlcc_indicator, k_lowest_eigvec), axis=-1),
            e=extra_edge_attr,
            y=np.hstack((n, y_cycles, n_components, batched_eigenvalues)),
        )
    else:
        raise ValueError(f"Features type {features_type} not implemented")
