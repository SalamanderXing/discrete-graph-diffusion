from jax import numpy as np, Array, jit
import jax
import matplotlib.pyplot as plt
import numpy
import networkx as nx
import optax
from rich import print
import ipdb
import wandb

# import jax_dataclasses as jdc
from mate.jax import SFloat, SInt, SBool, Key
from jaxtyping import Float, Bool, Int, jaxtyped
from typing import Sequence
from .graph_distribution import (
    GraphDistribution,
    EdgeDistribution,
    OneHotGraph,
    DenseGraphDistribution,
    StructureOneHotGraph,
)
from beartype import beartype
import einops as e

from typing import Sequence
from jax import random
from jax import lax
from jax.scipy.special import xlogy
from .q import Q

## only used for testing ##
import torch.nn.functional as F
import torch as t


NodeDistribution = Float[Array, "b n en"]
EdgeDistribution = Float[Array, "b n n ee"]
NodeMaskType = Bool[Array, "b n"]
EdgeMaskType = Bool[Array, "b n n"]
EdgeCountType = Int[Array, "b"]


@jaxtyped
@beartype
def _kl_div(
    p: Array,
    q: Array,
) -> Array:
    zero = 0.0
    both_gt_zero_mask = lax.bitwise_and(lax.gt(p, zero), lax.gt(q, zero))
    one_zero_mask = lax.bitwise_and(lax.eq(p, zero), lax.ge(q, zero))

    safe_p = np.where(both_gt_zero_mask, p, 1)
    safe_q = np.where(both_gt_zero_mask, q, 1)

    log_val = lax.sub(
        lax.add(
            lax.sub(xlogy(safe_p, safe_p), xlogy(safe_p, safe_q)),
            safe_q,
        ),
        safe_p,
    )
    result = np.where(both_gt_zero_mask, log_val, np.where(one_zero_mask, q, np.inf))
    return result


def typed(f):
    return jaxtyped(beartype(f))


def dense_to_structure_dense(g: DenseGraphDistribution):
    nodes_structure = np.concatenate(
        (g.nodes[..., 0][..., None], 1 - g.nodes[..., 1][..., None]), axis=-1
    )
    edges_structure = np.concatenate(
        (g.edges[..., 0][..., None], 1 - g.edges[..., 1][..., None]), axis=-1
    )
    return DenseGraphDistribution.create(
        nodes=nodes_structure,
        edges=edges_structure,
        nodes_mask=g.nodes_mask,
        edges_mask=g.edges_mask,
    )


def one_hot_structure_to_dense(
    structure: StructureOneHotGraph, node_feature_count: SInt, edge_feature_count: SInt
) -> DenseGraphDistribution:
    node_structure_mask = structure.nodes.argmax(-1)
    edge_structure_mask = structure.edges.argmax(-1)
    one_nodes = np.eye(node_feature_count)[0]
    structure_yes = one_nodes  # 1 / (node_feature_count - 1) * one_nodes
    structure_no = (1 - one_nodes) * (1 / (node_feature_count - 1))
    dense_nodes = np.where(
        node_structure_mask[..., None],
        structure_yes[None, None],
        structure_no[None, None],
    )
    one_edges = np.eye(edge_feature_count)[0]
    structure_yes = one_edges
    structure_no = (1 - one_edges) * (1 / (edge_feature_count - 1))
    dense_edges = np.where(
        edge_structure_mask[..., None],
        structure_yes[None, None, None],
        structure_no[None, None, None],
    )
    return DenseGraphDistribution.create(
        nodes=dense_nodes,
        edges=dense_edges,
        nodes_mask=structure.nodes_mask,
        edges_mask=structure.edges_mask,
    )


@typed
def concatenate(items: Sequence[GraphDistribution]) -> GraphDistribution:
    return GraphDistribution.create(
        nodes=np.concatenate(tuple(item.nodes for item in items)),
        edges=np.concatenate(tuple(item.edges for item in items)),
        nodes_mask=np.concatenate(tuple(item.nodes_mask for item in items)),
        edges_mask=np.concatenate(tuple(item.edges_mask for item in items)),
    )


# @typed
# def softmax_cross_entropy(
#     target: DenseGraphDistribution,
#     labels: OneHotGraph,
#     weights: Float[Array, "2"] | None = None,
# ) -> Float[Array, "b"]:
#     if weights is None:
#         weights = np.array([1.0, 1.0])
#     assert weights is not None
#     assert weights.shape == (2,)
#     loss_fn = optax.softmax_cross_entropy  # (logits=x, labels=y).sum()
#     nodes_loss = loss_fn(target.nodes, labels.nodes).sum(-1) * weights[0]
#     edges_loss = loss_fn(target.edges, labels.edges).sum(axis=(1, 2)) * weights[1]
#     return nodes_loss + edges_loss
#
@typed
def softmax_cross_entropy(
    target: DenseGraphDistribution,
    labels: OneHotGraph,
    weights: Float[Array, "2"] | None = None,
) -> Float[Array, "b"]:
    if weights is None:
        weights = np.array([1.0, 1.0])
    assert weights is not None
    assert weights.shape == (2,)
    loss_fn = optax.softmax_cross_entropy  # (logits=x, labels=y).sum()
    nodes_loss = loss_fn(target.nodes, labels.nodes).sum(-1) * weights[0]
    edges_loss = loss_fn(target.edges, labels.edges).sum(axis=(1, 2)) * weights[1]
    return nodes_loss + edges_loss


def check_same(a, b):
    from scipy.special import kl_div as s_kl_div

    first = np.array(s_kl_div(a, b))
    second = _kl_div(a, b)
    assert np.allclose(first, second, atol=1e-7), f"{a=} {b=} {first=} {second=}"


@typed
def __normalize(x: Array):
    x += 1e-11
    den = np.sum(x, axis=-1, keepdims=True)
    # x /= np.where(den > 0, den, 1)
    x /= den
    return x


@typed
def normalize(g: DenseGraphDistribution):
    return DenseGraphDistribution.create(
        nodes=__normalize(g.nodes),
        edges=__normalize(g.edges),
        nodes_mask=g.nodes_mask,
        edges_mask=g.edges_mask,
    )


@typed
def normalize_and_mask(g: DenseGraphDistribution):
    """
    Normalizes and then masks in a way that avoids distributions
    containing zeros (namely, a uniform distribution)
    """
    mask_x, mask_e = g.nodes_mask, g.edges_mask
    nodes = g.nodes
    edges = g.edges
    nodes_masked = np.where(mask_x[..., None], nodes, 1 / nodes.shape[-1])
    nodes_norm = __normalize(nodes_masked)
    edges_masked = np.where(mask_e[..., None], edges, 1 / edges.shape[-1])
    edges_norm = __normalize(edges_masked)
    return DenseGraphDistribution.create(
        nodes=nodes_norm,
        edges=edges_norm,
        nodes_mask=g.nodes_mask,
        edges_mask=g.edges_mask,
    )


# @jax.jit
@typed
def kl_div(input: DenseGraphDistribution, target: DenseGraphDistribution):
    kl_div_nodes = _kl_div(input.nodes, target.nodes)
    # kl_div_nodes = np.log(input.nodes + 1e-7) - np.log(target.nodes + 1e-7)
    kl_div_nodes = np.where(kl_div_nodes < np.inf, kl_div_nodes, 0)
    summed_kl_div_nodes = kl_div_nodes.sum((1, 2))
    kl_div_edges = _kl_div(input.edges, target.edges)
    # kl_div_edges = np.log(input.edges + 1e-7) - np.log(target.edges + 1e-7)
    kl_div_edges = np.where(kl_div_edges < np.inf, kl_div_edges, 0)
    summed_kl_div_edges = kl_div_edges.sum((1, 2, 3))
    res = summed_kl_div_nodes + summed_kl_div_edges
    return res


@jit
def softmax(g: DenseGraphDistribution) -> DenseGraphDistribution:
    return DenseGraphDistribution.create(
        nodes=jax.nn.softmax(g.nodes),
        edges=jax.nn.softmax(g.edges),
        nodes_mask=g.nodes_mask,
        edges_mask=g.edges_mask,
    )


@typed
def plot(
    rows: Sequence[GraphDistribution] | GraphDistribution,
    location: str | None = None,
    shared_position: str | None = None,  # can be "row" or "col", or none
    title: str | None = None,
):
    if isinstance(rows, GraphDistribution):
        rows = [rows]
    assert isinstance(rows, Sequence)
    assert shared_position in [None, "row", "col", "all"]
    location = (
        f"{location}.png"
        if location is not None and not location.endswith(".png")
        else location
    )
    original_len = len(rows[0])
    skip = (len(rows[0]) // 15) if len(rows[0]) > 15 else 1
    rows_skips = [(row[::skip], row[np.array([-1])]) for row in rows]
    rows = [concatenate(row) for row in rows_skips]
    lrows = len(rows)
    lcolumn = len(rows[0])
    try:
        _, axs = plt.subplots(
            lrows,
            lcolumn,
            figsize=(100, 10),
        )
    except Exception as e:
        ipdb.set_trace()
    if len(axs.shape) == 1:
        axs = axs[None, :]

    if rows[0].nodes.shape[-1] > 2:
        cmap_edge = plt.cm.viridis(np.linspace(0, 1, rows[0].edges.shape[-1] - 1))
        cmap_node = plt.cm.viridis(np.linspace(0, 1, rows[0].nodes.shape[-1]))
        cmap_edge = numpy.concatenate([numpy.zeros((1, 4)), cmap_edge], axis=0)
    else:
        cmap_edge = plt.cm.viridis(np.linspace(0, 1, rows[0].edges.shape[-1] - 1))
        cmap_node = plt.cm.viridis(np.linspace(0, 1, rows[0].nodes.shape[-1] - 1))
        cmap_edge = numpy.concatenate([numpy.zeros((1, 4)), cmap_edge], axis=0)
        cmap_node = numpy.concatenate([numpy.zeros((1, 4)), cmap_node], axis=0)
    node_size = 10.0
    positions = [None] * len(rows[0])

    # x s.t len(row) / x = 15
    # => x = len(row) / 15

    for i, (row, ax_row) in enumerate(zip(rows, axs)):
        xs = row.nodes.argmax(-1)
        es = row.edges.argmax(-1)
        n_nodes_row = row.nodes_counts
        for j in range(len(ax_row)):
            ax = ax_row[j]
            j *= skip
            x = xs[j]
            e = es[j]
            n_nodes = n_nodes_row[j]

            nodes = x[:n_nodes]
            edges_features = e[:n_nodes, :n_nodes]

            indices = np.indices(edges_features.shape).reshape(2, -1).T
            mask = edges_features.flatten() != 0
            edges = indices[mask]

            # c_values_edges = np.array([cmap_edge[i] for i in edges[:, -1]])

            G = nx.Graph()
            for i in range(n_nodes):
                G.add_node(i)
            for i in range(edges.shape[0]):
                G.add_edge(edges[i, 0].tolist(), edges[i, 1].tolist())

            if shared_position is not None:
                if positions[j] is None:
                    if j > 0 and shared_position in (
                        "row",
                        "all",
                    ):  # FIXME: "row" is not working, does the same as all
                        positions[j] = positions[0]
                    else:
                        positions[j] = nx.spring_layout(G)  # positions for all nodes
                position = positions[j]
            else:
                position = nx.spring_layout(G)

            color_nodes = numpy.array([cmap_node[i] for i in nodes])
            color_edges = numpy.array(
                [cmap_edge[edges_features[i, j]] for (i, j) in G.edges]
            )
            nx.draw(
                G,
                position,
                node_size=node_size,
                edge_color=color_edges,
                node_color=color_nodes,
                ax=ax,
            )
            ax.set_title(f"t={j*skip if not j == len(row)-1 else original_len-1}")
            ax.set_aspect("equal")
            ax.set_axis_off()

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if title is not None:
        plt.suptitle(title, fontsize=16)

    if location is None:
        plt.show()
    elif location == "wandb":

        wandb.log({"prediction": wandb.Image(plt)})
    else:
        plt.savefig(location)
        plt.clf()
        plt.close()
        # wandb.log({"prediction": wandb.Image(location)})
    plt.clf()
    plt.close()


@typed
def add_dense(
    g: DenseGraphDistribution, other: DenseGraphDistribution
) -> DenseGraphDistribution:
    return DenseGraphDistribution.create(
        nodes=g.nodes + other.nodes,
        edges=g.edges + other.edges,
        nodes_mask=g.nodes_mask,
        edges_mask=g.edges_mask,
    )


@typed
def logprobs_at(
    g: GraphDistribution, one_hot: GraphDistribution
) -> Float[Array, "batch_size"]:
    """Returns the probability of the given one-hot vector."""
    probs = OneHotGraph.create(
        nodes=g.nodes * one_hot.nodes,
        edges=g.edges * one_hot.edges,
        nodes_mask=g.nodes_mask,
        edges_mask=g.edges_mask,
    )
    nodes_probs_sum = probs.nodes.sum(-1)
    edges_probs_sum = probs.edges.sum(-1)
    safe_probs_nodes = np.where(nodes_probs_sum > 0, probs.nodes.sum(-1), 1)
    safe_probs_edges = np.where(edges_probs_sum > 0, probs.edges.sum(-1), 1)
    # TODO: the version below is safer cause it implies a kind of check
    # i.e. do the probs sum up to 1 for each node/edge? (wherever the mask is True)
    # however, that does not work with the feature diffusion where we have a
    # lot of zeros
    # safe_probs_nodes = np.where(g.nodes_mask, probs.nodes.sum(-1), 1)
    # safe_probs_edges = np.where(g.edges_mask, probs.edges.sum(-1), 1)
    nodes_logprob = np.log(safe_probs_nodes)
    edges_logprob = np.log(safe_probs_edges)
    return nodes_logprob.sum(1) + edges_logprob.sum((1, 2))


@typed
def to_symmetric(edges: EdgeDistribution) -> EdgeDistribution:
    upper = e.rearrange(
        np.triu(np.ones((edges.shape[1], edges.shape[2]))), "n1 n2 -> 1 n1 n2 1"
    )
    return np.where(upper, edges, e.rearrange(edges, "b n1 n2 ee -> b n2 n1 ee"))


@typed
def diag_to_zero(edges: EdgeDistribution) -> EdgeDistribution:
    return np.where(np.eye(edges.shape[1])[None, :, :, None], 0, edges)


import torch as t


@typed
def sample_one_hot(g: DenseGraphDistribution, rng_key: Key) -> OneHotGraph:
    """Sample features from multinomial distribution with given probabilities (probX, probE, proby)
    :param probX: bs, n, dx_out        node features
    :param probE: bs, n, n, de_out     edge features
    :param rng_key: random.PRNGKey     random key for JAX operations
    """

    rng_nodes, rng_edges = random.split(rng_key)
    b, n, ne = g.nodes.shape
    _, _, _, ee = g.edges.shape
    # mask = self.mask
    prob_x = g.nodes
    prob_e = g.edges

    # Noise X
    prob_x = np.where(g.nodes_mask[..., None], prob_x, 1 / ne)
    x_t = random.categorical(rng_nodes, np.log(prob_x), axis=-1)  # (bs * n,)
    prob_e = np.where(g.edges_mask[..., None], prob_e, 1 / ee)
    e_t = random.categorical(rng_edges, np.log(prob_e))
    embedded_x = jax.nn.one_hot(x_t, num_classes=ne)
    embedded_e = jax.nn.one_hot(e_t, num_classes=ee)
    return OneHotGraph.create_and_mask(
        nodes=embedded_x,
        edges=embedded_e,
        nodes_mask=g.nodes_mask,
        edges_mask=g.edges_mask,
    )


# @jax.jit
def matmul(g: OneHotGraph, q: Q):
    x = g.nodes @ q.nodes
    e = g.edges @ q.edges[:, None]
    return DenseGraphDistribution.create_and_mask(
        nodes=x, edges=e, nodes_mask=g.nodes_mask, edges_mask=g.edges_mask
    )


def sum(g: GraphDistribution) -> Array:
    node_mask, edge_mask = g.nodes_mask, g.edges_mask
    nodes = (g.nodes * node_mask[..., None]).mean(-1)
    edges = (g.edges * edge_mask[..., None]).mean(-1)
    return np.einsum("bn->b", nodes) + np.einsum("bjk->b", edges)


@typed
def get_masks(
    nodes_counts: Int[Array, "bs"], n: SInt
) -> tuple[NodeMaskType, EdgeMaskType]:
    n_range = np.arange(n)
    bs = nodes_counts.shape[0]
    mask_x = e.repeat(n_range, "n -> bs n", bs=bs) < e.repeat(
        nodes_counts, "bs -> bs n", n=n
    )
    e_ranges = e.repeat(mask_x, "bs n -> bs n n1", n1=n)
    e_diag = e.rearrange(np.eye(n, dtype=bool), "n1 n2 -> 1 n1 n2")
    mask_e = e_ranges & e.rearrange(e_ranges, "bs n n1 -> bs n1 n") & ~e_diag
    return mask_x, mask_e


# @typed
# def create_dense_from_counts(
#     nodes: NodeDistribution,
#     edges: EdgeDistribution,
#     nodes_counts: EdgeCountType,
# ):
#     nodes_mask, edges_mask = get_masks(nodes_counts, nodes.shape[1])
#     return DenseGraphDistribution(
#         nodes=nodes,
#         edges=edges,
#         nodes_mask=nodes_mask,
#         edges_mask=edges_mask,
#         _created_internally=True,
#     )
