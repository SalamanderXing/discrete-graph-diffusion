"""
This file contains the code for computing the egienvalues/vectors features. These are hardcoded computations that are added to the the input of the graph transformer.
"""
from jax import numpy as np
from jax import Array
from jax import random

# imports mode from scipy.stats
from jax.scipy.stats import mode
from .diffusion_types import Graph
import ipdb


def get_eigenvectors_features(
    vectors: Array,
    node_mask: Array,
    n_connected: Array,
    k: int = 2,
) -> tuple[Array, Array]:
    bs, n = vectors.shape[0], vectors.shape[1]

    # Create an indicator for the nodes outside the largest connected components
    first_ev = np.round(vectors[:, :, 0], decimals=3) * node_mask
    rng_key = random.PRNGKey(0)
    random_values = random.normal(rng_key, (bs, n)) * ~node_mask
    first_ev = first_ev + random_values
    most_common = np.squeeze(mode(first_ev, axis=1).mode)
    mask = ~(first_ev == most_common[:, None])
    not_lcc_indicator = (mask * node_mask)[:, :, None].astype(np.float32)

    # Get the eigenvectors corresponding to the first nonzero eigenvalues
    to_extend = max(n_connected) + k - n
    if to_extend > 0:
        vectors = np.concatenate(
            (vectors, np.zeros((bs, n, to_extend), dtype=vectors.dtype)), axis=2
        )
    indices = np.arange(k, dtype=np.int32)[None, None, :] + n_connected[:, :, None]
    indices = np.broadcast_to(indices, (bs, n, k))
    first_k_ev = np.take_along_axis(vectors, indices, axis=2)
    first_k_ev = first_k_ev * node_mask[:, :, None]

    return not_lcc_indicator, first_k_ev


def get_eigenvalues_features(
    eigenvalues: Array,
    k: int = 5,
) -> tuple[Array, Array]:
    ev = eigenvalues
    bs, n = ev.shape
    n_connected_components = np.sum(ev < 1e-5, axis=-1)
    assert np.all(n_connected_components > 0), (n_connected_components, ev)

    to_extend = max(n_connected_components) + k - n
    if to_extend > 0:
        eigenvalues = np.hstack(
            (eigenvalues, 2 * np.ones((bs, to_extend), dtype=eigenvalues.dtype))
        )
    indices = np.arange(k, dtype=np.int32)[None, :] + n_connected_components[:, None]
    first_k_ev = np.take_along_axis(eigenvalues, indices, axis=1)
    return n_connected_components[..., None], first_k_ev


def compute_laplacian(adjacency: Array, normalize: bool) -> Array:
    """
    adjacency : batched adjacency matrix (bs, n, n)
    normalize: can be None, 'sym' or 'rw' for the combinatorial, symmetric normalized or random walk Laplacians
    Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    diag = np.sum(adjacency, axis=-1)  # (bs, n)
    n = diag.shape[-1]
    D = np.diag(diag)  # Degree matrix      # (bs, n, n)
    combinatorial = D - adjacency  # (bs, n, n)

    if not normalize:
        return (combinatorial + combinatorial.transpose((0, 2, 1))) / 2

    diag0 = diag.copy()
    diag = np.where(diag == 0, 1e-12, diag)

    diag_norm = 1 / np.sqrt(diag)  # (bs, n)
    D_norm = np.diag(diag_norm)  # (bs, n, n)
    L = np.eye(n)[None, :, :] - np.matmul(np.matmul(D_norm, adjacency), D_norm)
    L = np.where(diag0[:, :, None] == 0, 0, L)
    return (L + L.transpose((0, 2, 1))) / 2


def eigen_features(mode: str, graph: Graph) -> tuple[Array, ...]:
    E_t = graph.e
    mask = graph.mask
    A = E_t[..., 1:].sum(axis=-1).astype(np.float32) * mask[:, None] * mask[:, :, None]
    L = compute_laplacian(A, normalize=False)
    mask_diag = 2 * L.shape[-1] * np.eye(A.shape[-1], dtype=L.dtype)[None, :, :]
    mask_diag = mask_diag * (~mask)[:, None] * (~mask)[:, :, None]
    L = L * mask[:, None] * mask[:, :, None] + mask_diag

    if mode == "eigenvalues":
        eigvals = np.linalg.eigvalsh(L)  # bs, n
        eigvals = eigvals.astype(A.dtype) / np.sum(mask, axis=1, keepdims=True)

        n_connected_comp, batch_eigenvalues = get_eigenvalues_features(
            eigenvalues=eigvals
        )
        return n_connected_comp.astype(A.dtype), batch_eigenvalues.astype(A.dtype)

    elif mode == "all":
        eigvals, eigvectors = np.linalg.eigh(L)
        eigvals = eigvals.astype(A.dtype) / np.sum(mask, axis=1, keepdims=True)
        eigvectors = eigvectors * mask[..., None] * mask[:, None]
        # Retrieve eigenvalues features
        n_connected_comp, batch_eigenvalues = get_eigenvalues_features(
            eigenvalues=eigvals
        )

        # Retrieve eigenvectors features
        nonlcc_indicator, k_lowest_eigenvector = get_eigenvectors_features(
            vectors=eigvectors,
            node_mask=graph.mask,
            n_connected=n_connected_comp,
        )
        return (
            n_connected_comp,
            batch_eigenvalues,
            nonlcc_indicator,
            k_lowest_eigenvector,
        )
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented")
