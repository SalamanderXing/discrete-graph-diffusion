from typing import Optional, Tuple
import jax
from jax import Array
import jax.numpy as np
import ipdb


def remove_self_loops(
    edge_index: Array,
    edge_attr: Optional[Array] = None,
) -> Tuple[Array, Optional[Array]]:
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]

    if edge_attr is None:
        return edge_index, None
    else:
        return edge_index, edge_attr[mask]


def to_dense_batch(
    x: Array,
    batch: Optional[Array] = None,
    fill_value: float = 0.0,
    max_num_nodes: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Tuple[Array, Array]:
    if batch is None and max_num_nodes is None:
        mask = np.ones((1, x.shape[0]), dtype=np.bool_)
        return np.expand_dims(x, 0), mask

    if batch is None:
        batch = np.zeros(x.shape[0], dtype=np.int32)

    if batch_size is None:
        batch_size = int(batch.max()) + 1

    num_nodes = jax.ops.segment_sum(np.ones(x.shape[0]), batch, num_segments=batch_size)
    cum_nodes = np.concatenate([np.zeros(1, dtype=np.int32), np.cumsum(num_nodes)[:-1]])

    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())
    else:
        filter_nodes = num_nodes > max_num_nodes
        if filter_nodes.any():
            raise NotImplementedError(
                "Filtering nodes is not implemented in this JAX version."
            )
    tmp = np.arange(batch.size) - cum_nodes[batch]
    idx = (tmp + (batch * max_num_nodes)).astype(np.int32)
    size = [batch_size * max_num_nodes] + list(x.shape)[1:]
    out = np.full(size, fill_value)
    out = out.at[idx].set(x)
    out = out.reshape([batch_size, max_num_nodes] + list(x.shape)[1:])

    mask = np.zeros(batch_size * max_num_nodes, dtype=np.bool_)
    mask = mask.at[idx].set(True)
    mask = mask.reshape(batch_size, max_num_nodes)

    return out, mask


def to_dense_adj(
    edge_index: Array,
    batch: Optional[Array] = None,
    edge_attr: Optional[Array] = None,
    max_num_nodes: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Array:
    if batch is None:
        num_nodes = int(edge_index.max()) + 1 if edge_index.size > 0 else 0
        batch = np.zeros(num_nodes, dtype=np.int32)

    if batch_size is None:
        batch_size = int(batch.max()) + 1 if batch.size > 0 else 1

    one = np.ones(batch.shape[0], dtype=np.int32)
    num_nodes = jax.ops.segment_sum(one, batch, num_segments=batch_size)
    cum_nodes = np.concatenate([np.zeros(1, dtype=np.int32), np.cumsum(num_nodes)[:-1]])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())
    elif (idx1.size > 0 and idx1.max() >= max_num_nodes) or (
        idx2.size > 0 and idx2.max() >= max_num_nodes
    ):
        mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
        idx0 = idx0[mask]
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        edge_attr = None if edge_attr is None else edge_attr[mask]

    if edge_attr is None:
        edge_attr = np.ones(idx0.size, dtype=np.float32)

    size = [batch_size, max_num_nodes, max_num_nodes]
    size += list(edge_attr.shape)[1:]
    flattened_size = batch_size * max_num_nodes * max_num_nodes

    idx = idx0 * max_num_nodes * max_num_nodes + idx1 * max_num_nodes + idx2
    adj = jax.ops.segment_sum(edge_attr, idx, num_segments=flattened_size)
    adj = adj.reshape(size)

    return adj


def encode_no_edge(E: Array) -> Array:
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = np.sum(E, axis=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt = np.where(no_edge, 1, first_elt)
    E = E.at[:, :, :, 0].set(first_elt)
    diag = (
        np.eye(E.shape[1], dtype=np.bool_)
        .reshape(1, E.shape[1], E.shape[1])
        .repeat(E.shape[0], axis=0)
    )
    E = E.at[diag].set(0)
    return E


def to_dense(x: Array, edge_index: Array, edge_attr: Array, batch: Array):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    # node_mask = node_mask.float()
    first, second = remove_self_loops(edge_index, edge_attr)
    assert second is not None, "edge_attr must be provided TODO check this"
    edge_index, edge_attr = first, second
    # TODO: carefully check if setting node_mask as a bool breaks the continuous case
    max_num_nodes = X.shape[1]
    E = to_dense_adj(
        edge_index=edge_index,
        batch=batch,
        edge_attr=edge_attr,
        max_num_nodes=max_num_nodes,
    )
    E = encode_no_edge(E)

    return (
        X,
        E,
        node_mask,
    )
