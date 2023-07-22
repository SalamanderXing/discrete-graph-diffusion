"""
This file contains the code for computing the cycle features. These are hardcoded computations that are added to the the input of the graph transformer.
"""
import jax.numpy as np
from jax import Array

from . import GraphDistribution

check = lambda x, y="": None


def batch_trace(X: Array) -> Array:
    """
    Expect a matrix of shape B N N, returns the trace in shape B
    :param X:
    :return:
    """
    diag = np.diagonal(X, axis1=-2, axis2=-1)
    trace = diag.sum(axis=-1)
    return trace


def batch_diagonal(X: Array) -> Array:
    """
    Extracts the diagonal from the last two dims of a tensor
    :param X:
    :return:
    """
    return np.diagonal(X, axis1=-2, axis2=-1)


def k3_cycle(k3_matrix: Array) -> tuple[Array, Array]:
    c3 = batch_diagonal(k3_matrix)
    return (c3 / 2)[:, None].astype(np.float32), (np.sum(c3, axis=-1) / 6)[
        :, None
    ].astype(np.float32)


def k4_cycle(adj_matrix: Array, k4_matrix: Array, d: Array) -> tuple[Array, Array]:
    diag_a4 = batch_diagonal(k4_matrix)
    last = (adj_matrix @ d[..., None]).sum(axis=-1)
    second = d * (d - 1)
    c4 = diag_a4 - second - last
    return (c4 / 2)[:, None].astype(np.float32), (np.sum(c4, axis=-1) / 8)[
        :, None
    ].astype(np.float32)


def k5_cycle(
    k5_matrix: Array, k3_matrix: Array, adj_matrix: Array, d: Array
) -> tuple[Array, Array]:
    diag_a5 = batch_diagonal(k5_matrix)
    triangles = batch_diagonal(k3_matrix)

    c5 = (
        diag_a5
        - 2 * triangles * d
        - (adj_matrix @ triangles[..., None]).sum(axis=-1)
        + triangles
    )
    return (c5 / 2)[:, None].astype(np.float32), (c5.sum(axis=-1) / 10)[:, None].astype(
        np.float32
    )


def k6_cycle(
    k6_matrix: Array,
    k2_matrix: Array,
    k4_matrix: Array,
    k3_matrix: Array,
    adj_matrix: Array,
) -> tuple[None, Array]:
    term_1_t = batch_trace(k6_matrix)
    term_2_t = batch_trace(k3_matrix**2)
    term3_t = np.sum(adj_matrix * k2_matrix**2, axis=(-2, -1))
    d_t4 = batch_diagonal(k2_matrix)
    a_4_t = batch_diagonal(k4_matrix)
    term_4_t = (d_t4 * a_4_t).sum(axis=-1)
    term_5_t = batch_trace(k4_matrix)
    term_6_t = batch_trace(k3_matrix)
    term_7_t = (batch_diagonal(k2_matrix).clip(min=0) ** 3).sum(axis=-1)
    term8_t = np.sum(k3_matrix, axis=(-2, -1))
    term9_t = (batch_diagonal(k2_matrix).clip(min=0) ** 2).sum(axis=-1)
    term10_t = batch_trace(k2_matrix)

    c6_t = (
        term_1_t
        - 3 * term_2_t
        + 9 * term3_t
        - 6 * term_4_t
        + 6 * term_5_t
        - 4 * term_6_t
        + 4 * term_7_t
        + 3 * term8_t
        - 12 * term9_t
        + 4 * term10_t
    )
    return None, (c6_t / 12)[:, None].astype(np.float32)


def k_cycles(adj_matrix: Array) -> tuple[Array, Array]:
    # Calculate k powers
    d = adj_matrix.sum(axis=-1)
    k1_matrix = adj_matrix.astype(np.float32)
    k2_matrix = k1_matrix @ adj_matrix.astype(np.float32)
    k3_matrix = k2_matrix @ adj_matrix.astype(np.float32)
    k4_matrix = k3_matrix @ adj_matrix.astype(np.float32)
    k5_matrix = k4_matrix @ adj_matrix.astype(np.float32)
    k6_matrix = k5_matrix @ adj_matrix.astype(np.float32)

    k3x, k3y = k3_cycle(k3_matrix)
    check((k3x >= -0.1).all())

    k4x, k4y = k4_cycle(adj_matrix, k4_matrix, d)
    check((k4x >= -0.1).all())

    k5x, k5y = k5_cycle(k5_matrix, k3_matrix, adj_matrix, d)
    check((k5x >= -0.1).all(), str(k5x))

    _, k6y = k6_cycle(k6_matrix, k2_matrix, k4_matrix, k3_matrix, adj_matrix)
    check((k6y >= -0.1).all())

    kcyclesx = np.concatenate(
        [k3x.swapaxes(1, 2), k4x.swapaxes(1, 2), k5x.swapaxes(1, 2)], axis=-1
    )
    kcyclesy = np.concatenate([k3y, k4y, k5y, k6y], axis=-1)
    # FIXME: why did I have to transpose Xs?? could be a bug
    # print(f"{kcyclesx.shape=}, {kcyclesy.shape=}")
    # print(f"{k3x.shape=}, {k4x.shape=}, {k5x.shape=}")
    # print(f"{k3y.shape=}, {k4y.shape=}, {k5y.shape=}, {k6y.shape=}")
    return kcyclesx, kcyclesy


def node_cycle_features(graph: GraphDistribution) -> tuple[Array, Array]:
    adj_matrix = graph.e[..., 1:].sum(axis=-1).astype(float)

    x_cycles, y_cycles = k_cycles(adj_matrix=adj_matrix)  # (bs, n_cycles)
    x_cycles = x_cycles.astype(adj_matrix.dtype) * graph.mask[..., None]
    # Avoid large values when the graph is dense
    x_cycles = x_cycles / 10
    y_cycles = y_cycles / 10
    x_cycles = np.where(x_cycles > 1, 1, x_cycles)  # x_cycles.at[x_cycles > 1].set(1)
    y_cycles = np.where(y_cycles > 1, 1, y_cycles)  # y_cycles.at[y_cycles > 1].set(1)
    return x_cycles, y_cycles
