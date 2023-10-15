# import numpy as np
from jax import Array
from jax import numpy as np
import ipdb
from jax import Array
from jax import jit
from functools import partial
from typing import Union


def subgraph(
    subset: Array,
    edge_index: Array,
    edge_attr: Array | None = None,
    relabel_nodes: bool = False,
    num_nodes: int | None = None,
    return_edge_mask: bool = False,
) -> Union[tuple[Array, Array | None], tuple[Array, Array | None, Array | None],]:
    ipdb.set_trace()
    if subset.dtype != np.bool_:
        num_nodes = edge_index.max() + 1 if num_nodes is None else num_nodes
        node_mask = np.isin(np.arange(num_nodes), subset)
    else:
        num_nodes = len(subset)
        node_mask = subset
        subset = np.nonzero(subset)[0]

    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    if relabel_nodes:
        # Relabeling logic
        new_labels = {node: i for i, node in enumerate(subset)}
        edge_index = np.vectorize(new_labels.get)(edge_index)

    if return_edge_mask:
        return edge_index, edge_attr, edge_mask
    else:
        return edge_index, edge_attr


from functools import partial
from jax import jit


# @jit
# def remove_self_loops(
#     edge_index: np.ndarray | Array, edge_attr: np.ndarray | Array
# ) -> tuple[np.ndarray | Array, np.ndarray | Array | None]:
#     """
#     Removes every self-loop in the graph given by `edge_index`.
#
#     Args:
#         edge_index (ndarray): The edge indices.
#         edge_attr (ndarray, optional): Edge weights or multi-dimensional edge features.
#
#     Returns:
#         tuple: Edge indices and edge attributes without self-loops.
#     """
#
#     # Create a mask for self-loops
#     mask = edge_index[0, :] != edge_index[1, :]
#
#     # Remove self-loops from edge_index
#     edge_index_filtered = edge_index[:, mask]
#
#     # Remove corresponding attributes of self-loops
#     edge_attr_filtered = edge_attr[mask]
#     return edge_index_filtered, edge_attr_filtered
#

import jax.numpy as np


# @partial(jit, static_argnames=("num_nodes",))
# def to_dense_adj(
#     edge_index: np.ndarray,
#     batch: np.ndarray,
#     edge_attr: np.ndarray,
#     max_num_nodes: Array,
#     batch_size: Array,
# ) -> np.ndarray:
#     """Converts batched sparse adjacency matrices to a dense batched adjacency matrix."""
#
#     num_nodes = edge_index.max() + 1 if edge_index.size > 0 else 0
#     batch = np.zeros(num_nodes, dtype=int)
#
#     # split from here!
#     num_nodes_per_batch = np.bincount(batch, minlength=batch_size)
#     cum_nodes = np.cumsum(num_nodes_per_batch) - num_nodes_per_batch
#
#     idx0 = batch[edge_index[0]]
#     idx1 = edge_index[0] - cum_nodes[batch[edge_index[0]]]
#     idx2 = edge_index[1] - cum_nodes[batch[edge_index[1]]]
#
#     mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
#     idx0 = idx0[mask]
#     idx1 = idx1[mask]
#     idx2 = idx2[mask]
#     edge_attr = edge_attr[mask]
#
#     size = [batch_size, max_num_nodes, max_num_nodes]
#     size += list(edge_attr.shape[1:])
#     adj = np.zeros(size)
#     update_mask = np.zeros_like(adj)
#     index_tuple = (idx0[:, None], idx1[:, None], idx2[:, None])
#     update_mask = update_mask.at[index_tuple].set(1)
#
#     # Use np.where to replace the loop
#     adj = np.where(update_mask, adj + np.broadcast_to(edge_attr, adj.shape), adj)
#
#     return adj


def to_dense_adj(
    edge_index: np.ndarray,
    batch: np.ndarray,
    edge_attr: np.ndarray,
    max_num_nodes: int,
    batch_size: int,
) -> np.ndarray:
    """Converts batched sparse adjacency matrices to a dense batched adjacency matrix."""

    num_nodes = edge_index.max() + 1 if edge_index.size > 0 else 0
    batch = np.zeros(num_nodes, dtype=int)
    num_nodes_per_batch = np.bincount(batch, minlength=batch_size)
    return jittable_function(
        edge_index,
        batch,
        edge_attr.shape[-1],
        edge_attr,
        max_num_nodes,
        batch_size,
        num_nodes_per_batch,
    )


@partial(jit, static_argnames=("num_edges_features", "batch_size", "max_num_nodes"))
def jittable_function(
    edge_index: np.ndarray,
    batch: np.ndarray,
    num_edges_features: int | Array,
    edge_attr: np.ndarray,
    max_num_nodes: int | Array,
    batch_size: int | Array,
    num_nodes_per_batch: int | Array,
) -> np.ndarray:
    cum_nodes = np.cumsum(num_nodes_per_batch) - num_nodes_per_batch

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch[edge_index[0]]]
    idx2 = edge_index[1] - cum_nodes[batch[edge_index[1]]]

    mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
    idx0 = np.where(mask, idx0, 0)
    idx1 = np.where(mask, idx1, 0)
    idx2 = np.where(mask, idx2, 0)
    # edge_attr = edge_attr[mask]
    edge_attr = np.where(np.expand_dims(mask, axis=-1), edge_attr, 0)

    size = [batch_size, max_num_nodes, max_num_nodes, num_edges_features]
    adj = np.zeros(size)

    update_mask = np.zeros_like(adj)
    index_tuple = (idx0[:, None], idx1[:, None], idx2[:, None])
    update_mask = update_mask.at[index_tuple].set(1)

    ipdb.set_trace()
    adj = np.where(update_mask, adj + edge_attr[:, None, None], adj)

    return adj


##


# def to_dense_adj(
#     edge_index: np.ndarray,
#     batch: np.ndarray,
#     edge_attr: np.ndarray,
#     max_num_nodes: int | Array,
#     batch_size: int | Array,
# ) -> np.ndarray:
#     """Converts batched sparse adjacency matrices to a dense batched adjacency matrix."""
#
#     # if batch is None:
#     num_nodes = edge_index.max() + 1 if edge_index.size > 0 else 0
#     batch = np.zeros(num_nodes, dtype=int)
#
#     # if batch_size is None:
#     #     batch_size = batch.max() + 1 if batch.size > 0 else 1
#
#     num_nodes_per_batch = np.bincount(batch, minlength=batch_size)
#     cum_nodes = np.cumsum(num_nodes_per_batch) - num_nodes_per_batch
#
#     idx0 = batch[edge_index[0]]
#     idx1 = edge_index[0] - cum_nodes[batch[edge_index[0]]]
#     idx2 = edge_index[1] - cum_nodes[batch[edge_index[1]]]
#
#     mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
#     idx0 = idx0[mask]
#     idx1 = idx1[mask]
#     idx2 = idx2[mask]
#     edge_attr = edge_attr[mask]
#
#     # assert batch_size is not None
#     size = [batch_size, max_num_nodes, max_num_nodes]
#     # if edge_attr.ndim > 1:
#     size += list(edge_attr.shape[1:])
#     adj = np.zeros(size)
#
#     for i, (i0, i1, i2) in enumerate(zip(idx0, idx1, idx2)):
#         # adj[i0, i1, i2] += edge_attr[i]
#         adj = adj.at[i0, i1, i2].set(adj[i0, i1, i2] + edge_attr[i])
#
#     return adj
#


# @partial(jit, static_argnames=("batch_size", "max_num_nodes"))
def to_dense_batch(
    x: Array,
    batch: np.ndarray | Array,
    fill_value: float = 0.0,
    max_num_nodes: int | None = None,
    batch_size: int | None = None,
) -> tuple[Array, Array]:
    if batch is None and max_num_nodes is None:
        mask = np.ones((1, x.shape[0]), dtype=bool)
        return np.expand_dims(x, axis=0), mask

    if batch is None:
        batch = np.zeros(x.shape[0], dtype=int)

    if batch_size is None:
        batch_size = batch.max() + 1

    num_nodes = np.bincount(batch, minlength=batch_size)
    cum_nodes = np.cumsum(num_nodes) - num_nodes

    # if max_num_nodes is None:
    max_num_nodes = num_nodes.max()

    tmp = np.arange(batch.size) - cum_nodes[batch]
    idx = tmp + (batch * max_num_nodes)

    mask = tmp < max_num_nodes
    x_filtered, idx_filtered = x[mask], idx[mask]

    size = [batch_size * max_num_nodes] + list(x.shape)[1:]
    out = np.full(size, fill_value, dtype=x.dtype)
    out = out.at[idx_filtered].set(x_filtered)
    out = out.reshape([batch_size, max_num_nodes] + list(x.shape)[1:])

    mask_out = np.zeros(batch_size * max_num_nodes, dtype=bool)
    mask_out = mask_out.at[idx_filtered].set(True)
    mask_out = mask_out.reshape(batch_size, max_num_nodes)

    return out, mask_out
