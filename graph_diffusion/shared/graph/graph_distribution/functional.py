from jax import numpy as np, Array
import jax
import networkx as nx
import optax
from jax.experimental.checkify import check
from rich import print
import ipdb
import jax_dataclasses as jdc
from mate.jax import SFloat, SInt, typed, SBool, Key
from jaxtyping import Float, Bool, Int
from typing import Sequence
from jax.scipy.special import logit
from .graph_distribution import GraphDistribution

# from .geometric import to_dense
from jax import random
from flax import linen as nn

# optax.softmax_cross_entropy_with_integer_labels


@typed
def concatenate(items: Sequence[GraphDistribution]) -> GraphDistribution:
    return GraphDistribution(
        nodes=np.concatenate(tuple(item.nodes for item in items)),
        edges=np.concatenate(tuple(item.edges for item in items)),
        nodes_counts=items[0].nodes_counts,
        edges_counts=items[0].edges_counts,
        _created_internally=True,
    )


@typed
def __cross_entropy(p: Array, q: Array) -> Array:
    q *= 10
    p *= 10
    q = jax.nn.softmax(q, axis=-1)
    p = jax.nn.softmax(p, axis=-1)
    return -np.sum(p * np.log(q), axis=-1)


def softmax_kl_div(p: GraphDistribution, q: GraphDistribution) -> Array:
    # q *= np.array(10)
    # p *= np.array(10)
    q = q.softmax()
    p = p.softmax()
    return kl_div(p, q)


@typed
def cross_entropy(
    target: GraphDistribution, labels: GraphDistribution
) -> Float[Array, "b"]:
    # loss_fn = optax.softmax_cross_entropy  # (logits=x, labels=y).sum()
    loss_fn = __cross_entropy
    # Remove masked rows
    mask_nodes, mask_edges = labels.masks()

    nodes_loss = (loss_fn(target.nodes, labels.nodes) * mask_nodes).sum(axis=1)
    edges_loss = (loss_fn(target.edges, target.edges) * mask_edges).sum(axis=(1, 2))
    return nodes_loss + edges_loss


@typed
def __kl_div(
    p: Array,
    q: Array,
    # eps: SFloat = 1**-17,
) -> Array:
    """Calculates the Kullback-Leibler divergence between arrays p and q."""
    # p += eps
    # q += eps
    p = np.where(p.argmax(-1)[..., None] == 0, 1 / p.shape[-1], p)
    q = np.where(q.argmax(-1)[..., None] == 0, 1 / q.shape[-1], q)
    return np.sum(p * np.log(p / q), axis=-1)


@typed
def __normalize(x: Array):
    x += 1e-7
    x /= np.sum(x, axis=-1, keepdims=True)
    return x


def normalize(g: GraphDistribution):
    return GraphDistribution.create(
        nodes=__normalize(g.nodes),
        edges=__normalize(g.edges),
        nodes_counts=g.nodes_counts,
        edges_counts=g.edges_counts,
    )


def normalize_and_mask(g: GraphDistribution):
    mask_x, mask_e = g.masks()
    new_nodes = __normalize(g.nodes)
    new_nodes = np.where(mask_x[..., None], new_nodes, 1 / new_nodes.shape[-1])
    new_edges = __normalize(g.edges)
    new_edges = np.where(mask_e[..., None], new_edges, 1 / new_edges.shape[-1])
    return GraphDistribution.create(
        nodes=new_nodes,
        edges=new_edges,
        nodes_counts=g.nodes_counts,
        edges_counts=g.edges_counts,
    )


@typed
def kl_div(input: GraphDistribution, target: GraphDistribution) -> Float[Array, "b"]:
    uba = __kl_div(input.nodes, target.nodes)
    nodes_kl = (uba).sum(axis=1)
    alla = __kl_div(input.edges, target.edges)
    edges_kl = (alla).sum(axis=(1, 2))
    result = nodes_kl + edges_kl
    return result


from typing import Sequence


@typed
def plot_compare(rows: Sequence[GraphDistribution], location: str | None = None):
    import matplotlib.pyplot as plt
    import numpy
    from tqdm import tqdm

    _, axs = plt.subplots(
        2,
        len(rows[0]),
        figsize=(100, 10),
    )

    cmap_edge = plt.cm.viridis(np.linspace(0, 1, rows[0].edges.shape[-1]))
    cmap_node = plt.cm.viridis(np.linspace(0, 1, rows[0].nodes.shape[-1]))
    cmap_edge[0, -1] = 0
    cmap_node[0, -1] = 0
    # cmap_edge[1, -1] = 0
    # cmap_node[1, -1] = 0
    node_size = 10.0
    pos = None

    for i, (row, ax_row) in enumerate(zip(rows, axs)):
        xs = row.nodes.argmax(-1)
        es = row.edges.argmax(-1)
        n_nodes = row.nodes_counts[0]
        for j in range(len(row)):
            ax = ax_row[j]
            x = xs[j]
            e = es[j]

            nodes = x[:n_nodes]
            edges_features = e[:n_nodes, :n_nodes]
            # c_values_nodes = np.array([cmap_node[0 if i == 0 else i + 10] for i in nodes])

            indices = np.indices(edges_features.shape).reshape(2, -1).T
            mask = edges_features.flatten() != 0
            edges = indices[mask]

            # c_values_edges = np.array([cmap_edge[i] for i in edges[:, -1]])

            G = nx.Graph()
            for i in range(n_nodes):
                G.add_node(i)
            for i in range(edges.shape[0]):
                G.add_edge(edges[i, 0].tolist(), edges[i, 1].tolist())

            if pos is None:
                pos = nx.spring_layout(G)  # positions for all nodes

            color_nodes = numpy.array([cmap_node[i] for i in nodes])
            color_edges = numpy.array(
                [cmap_edge[edges_features[i, j]] for (i, j) in G.edges]
            )
            nx.draw(
                G,
                pos,
                node_size=node_size,
                edge_color=color_edges,
                node_color=color_nodes,
                ax=ax,
            )
            ax.set_title(f"t={j}")
            ax.set_aspect("equal")

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    if location is None:
        plt.show()
    else:
        plt.savefig(location)
