from jax import numpy as np, Array, jit
import jax
import networkx as nx
import optax
from jax.experimental.checkify import check
from rich import print
import ipdb
import jax_dataclasses as jdc
from mate.jax import SFloat, SInt, SBool, Key
from jaxtyping import Float, Bool, Int, jaxtyped
from typing import Sequence
from jax.scipy.special import logit, kl_div as _kl_div
from .graph_distribution import GraphDistribution, EdgeDistribution
from .one_hot_graph_distribution import OneHotGraph
from .dense_graph_distribution import DenseGraphDistribution
from einop import einop
from beartype import beartype

# from .geometric import to_dense
from jax import random
from flax import linen as nn
from .q import Q

# optax.softmax_cross_entropy_with_integer_labels

NodeDistribution = Float[Array, "b n en"]
EdgeDistribution = Float[Array, "b n n ee"]
NodeMaskType = Bool[Array, "b n"]
EdgeMaskType = Bool[Array, "b n n"]
EdgeCountType = Int[Array, "b"]

# @jaxtyped
# @beartype

# merges the two decorators into one


def typed(f):
    return jaxtyped(beartype(f))


@typed
def concatenate(items: Sequence[GraphDistribution]) -> GraphDistribution:
    return create(
        nodes=np.concatenate(tuple(item.nodes for item in items)),
        edges=np.concatenate(tuple(item.edges for item in items)),
        nodes_mask=np.concatenate(tuple(item.nodes_mask for item in items)),
        edges_mask=np.concatenate(tuple(item.edges_mask for item in items)),
    )


# @typed
# def __cross_entropy(p: Array, q: Array) -> Array:
#     q *= 10
#     p *= 10
#     q = jax.nn.softmax(q, axis=-1)
#     p = jax.nn.softmax(p, axis=-1)
#     return -np.sum(p * np.log(q), axis=-1)
#


@typed
def softmax_kl_div(p: GraphDistribution, q: GraphDistribution) -> Array:
    # q *= .array(10)
    # p *= np.array(10)
    q = softmax(q)
    p = softmax(p)
    return kl_div(p, q)


@typed
def cross_entropy(
    target: DenseGraphDistribution, labels: OneHotGraph
) -> Float[Array, "b"]:
    loss_fn = optax.softmax_cross_entropy  # (logits=x, labels=y).sum()
    i1 = loss_fn(target.nodes, labels.nodes)
    i2 = loss_fn(target.edges, labels.edges)
    nodes_loss = (i1 * labels.nodes_mask).sum(axis=1)
    edges_loss = (i2 * labels.edges_mask).sum(axis=(1, 2))
    return nodes_loss + edges_loss


from jax import lax


ArrayLike = Array
# from jax._src.typing import Array, ArrayLike


# from jax.scipy.special import xlogy
#
#
def _test_kl_div(
    p: ArrayLike,
    q: ArrayLike,
) -> Array:
    """Calculates the Kullback-Leibler divergence between arrays p and q."""
    # p, q = q, p
    p = lax.log(p)
    q = lax.log(q)
    # p = lax.log(p)
    # result = xlogy(q, q) - lax.mul(q, p)
    # ipdb.set_trace()
    result = lax.mul(lax.exp(q), lax.sub(q, p))
    return result


@typed
def __normalize(x: Array):
    x += 1e-7
    den = np.sum(x, axis=-1, keepdims=True)
    x /= np.where(den > 0, den, 1)
    return x


@typed
def normalize(g: DenseGraphDistribution):
    return create_dense(
        nodes=__normalize(g.nodes),
        edges=__normalize(g.edges),
        nodes_mask=g.nodes_mask,
        edges_mask=g.edges_mask,
    )


@typed
def normalize_and_mask(g: DenseGraphDistribution):
    """Normalizes and then masks in a way that avoids distributions containing zeros (namely, a uniform distribution)"""
    mask_x, mask_e = g.nodes_mask, g.edges_mask
    new_nodes = __normalize(g.nodes)
    new_nodes = np.where(mask_x[..., None], new_nodes, 1 / new_nodes.shape[-1])
    new_edges = __normalize(g.edges)
    new_edges = np.where(mask_e[..., None], new_edges, 1 / new_edges.shape[-1])
    return create_dense(
        nodes=new_nodes,
        edges=new_edges,
        nodes_mask=g.nodes_mask,
        edges_mask=g.edges_mask,
    )


# def _kl_div(
#     p: ArrayLike,
#     q: ArrayLike,
# ) -> Array:
#     """Calculates the Kullback-Leibler divergence between arrays p and q."""
#     both_gt_zero_mask = lax.bitwise_and(lax.gt(p, 0.0), lax.gt(q, 0.0))
#     one_zero_mask = lax.bitwise_and(lax.eq(p, 0.0), lax.ge(q, 0.0))
#
#     one_filler = lax.full_like(p, 1.0)
#     inf_filler = lax.full_like(p, np.inf)
#
#     safe_p = lax.select(both_gt_zero_mask, p, one_filler)
#     safe_q = lax.select(both_gt_zero_mask, q, one_filler)
#
#     log_val = lax.add(
#         lax.sub(lax.mul(safe_p, lax.log(lax.div(safe_p, safe_q))), safe_p), safe_q
#     )
#     result = lax.select(
#         both_gt_zero_mask, log_val, lax.select(one_zero_mask, q, inf_filler)
#     )
#     return result

from jax.scipy.special import xlogy, entr


# def _kl_div(
#     p: ArrayLike,
#     q: ArrayLike,
# ):
#     both_gt_zero_mask = lax.bitwise_and(lax.gt(p, 0.0), lax.gt(q, 0.0))
#     one_zero_mask = lax.bitwise_and(lax.eq(p, 0.0), lax.ge(q, 0.0))
#     inf_filler = lax.full_like(p, np.inf)
#     log_val = -xlogy(p, q) - entr(p)
#     base_case= lax.select(
#         both_gt_zero_mask, log_val, lax.select(one_zero_mask, q, inf_filler)
#     )
#     return result


def check_same(a, b):
    from scipy.special import kl_div as s_kl_div

    first = np.array(s_kl_div(a, b))
    second = _kl_div(a, b)
    assert np.allclose(first, second, atol=1e-7), f"{a=} {b=} {first=} {second=}"


# @jax.jit
@typed
def kl_div(
    input: DenseGraphDistribution, target: DenseGraphDistribution
) -> Float[Array, "b"]:
    # from scipy.special import kl_div as s_kl_div

    # from torch.nn.functional import kl_div as t_kl_div
    # import torch as t
    # import numpy as npo

    input = normalize_and_mask(input)
    target = normalize_and_mask(target)
    # t_kl_div_nodes = np.array(
    #     t_kl_div(
    #         t.tensor(npo.array(input.nodes)), t.log(t.tensor(npo.array(target.nodes)))
    #     ).numpy()
    # )
    # np_kl_div_nodes = np.array(s_kl_div(input.nodes, target.nodes)).sum(-1)
    kl_div_nodes = _kl_div(input.nodes, target.nodes).sum(-1)
    # kl_div_test = _test_kl_div(input.nodes, target.nodes).sum(-1)

    # assert np.allclose(kl_div_nodes, np_kl_div_nodes, atol=1e-6), (
    #     kl_div_nodes[0],
    #     np_kl_div_nodes[0],
    # )
    # check_same(np.array([0.0, 1.0]), np.array([1.0, 0.0]))
    # check_same(np.array([0.9, 0.1]), np.array([0.9, 0.1]))
    # n1 = input.nodes
    # n2 = target.nodes
    # test2 = n1 * np.log(n1 / n2)
    # test3 = n1 * np.log(n1 / n2) - n1 + n2
    nodes_kl = kl_div_nodes.sum(axis=1)
    kl_div_edges = _kl_div(input.edges, target.edges).sum(-1)
    edges_kl = kl_div_edges.sum(axis=(1, 2))
    # print(f"[green]{kl_div_nodes.sum()=}[/green]")
    # print(f"[green]{kl_div_edges.sum()=}[/green]")
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


@jit
def softmax(g: DenseGraphDistribution) -> DenseGraphDistribution:
    return create_dense(
        nodes=jax.nn.softmax(g.nodes),
        edges=jax.nn.softmax(g.edges),
        nodes_mask=g.nodes_mask,
        edges_mask=g.edges_mask,
    )


@typed
def plot(
    rows: Sequence["GraphDistribution"],
    location: str | None = None,
    shared_position: str | None = None,  # can be "row" or "col", or none
    title: str | None = None,
):
    assert shared_position in [None, "row", "col", "all"]
    location = (
        f"{location}.png"
        if location is not None and not location.endswith(".png")
        else location
    )
    original_len = len(rows[0])
    skip = (len(rows[0]) // 15) if len(rows[0]) > 15 else 1
    rows = [concatenate((row[::skip], row[np.array([-1])])) for row in rows]

    # assert 1 <= len(rows) <= 2, "can only plot 1 or 2 rows"
    import matplotlib.pyplot as plt
    import numpy
    from tqdm import tqdm

    fig, axs = plt.subplots(
        len(rows),
        len(rows[0]),
        figsize=(100, 10),
    )
    if len(axs.shape) == 1:
        axs = axs[None, :]

    cmap_edge = plt.cm.viridis(np.linspace(0, 1, rows[0].edges.shape[-1] - 1))
    cmap_node = plt.cm.viridis(np.linspace(0, 1, rows[0].nodes.shape[-1]))
    cmap_edge = numpy.concatenate([numpy.zeros((1, 4)), cmap_edge], axis=0)
    # cmap_node = numpy.concatenate([cmap_node, numpy.zeros((1, 4))], axis=0)
    node_size = 10.0
    positions = [None] * len(rows[0])

    # x s.t len(row) / x = 15
    # => x = len(row) / 15

    for i, (row, ax_row) in enumerate(zip(rows, axs)):
        xs = row.nodes.argmax(-1)
        es = row.edges.argmax(-1)
        n_nodes_row = row.nodes_counts
        for j in range(len(row)):
            ax = ax_row[j]
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
    else:
        plt.savefig(location)
        plt.clf()
        plt.close()
        # wandb.log({"prediction": wandb.Image(location)})
    plt.clf()
    plt.close()


@typed
def argmax(g: GraphDistribution) -> OneHotGraph:
    id_nodes = np.eye(g.nodes.shape[-1])
    id_edges = np.eye(g.edges.shape[-1])
    nodes = id_nodes[g.nodes.argmax(-1)]
    edges = id_edges[g.edges.argmax(-1)]
    return create_one_hot(
        nodes=nodes,
        edges=edges,
        nodes_mask=g.nodes_mask,
        edges_mask=g.edges_mask,
    )


@typed
def repeat_dense(g: DenseGraphDistribution, n: int) -> DenseGraphDistribution:
    return create_dense(
        nodes=np.repeat(g.nodes, n, axis=0),
        edges=np.repeat(g.edges, n, axis=0),
        nodes_mask=np.repeat(g.nodes_mask, n, axis=0),
        edges_mask=np.repeat(g.edges_mask, n, axis=0),
    )


@typed
def repeat(g: GraphDistribution, n: int) -> GraphDistribution:
    return create(
        nodes=np.repeat(g.nodes, n, axis=0),
        edges=np.repeat(g.edges, n, axis=0),
        nodes_mask=np.repeat(g.nodes_mask, n, axis=0),
        edges_mask=np.repeat(g.edges_mask, n, axis=0),
    )


def __truediv__(
    self, other: "GraphDistribution | SFloat | SInt"
) -> "GraphDistribution":
    if isinstance(other, (SFloat, SInt)):
        return self.__class__.create(
            nodes=self.nodes / other,
            edges=self.edges / other,
            nodes_mask=self.nodes_mask,
            edges_mask=self.edges_mask,
        )
    elif isinstance(other, GraphDistribution):
        pseudo_assert((other.nodes > 0).all())
        pseudo_assert((other.edges > 0).all())
        new_nodes = self.nodes / other.nodes
        new_edges = self.edges / other.edges
        try:
            pseudo_assert(np.allclose(new_nodes.sum(-1), 1, 0.15))
        except:
            misfit = np.where(~np.allclose(new_nodes.sum(-1, keepdims=True), 1, 0.15))
            ipdb.set_trace()
        pseudo_assert(np.allclose(new_edges.sum(-1), 1, 0.15))
        new_nodes = new_nodes / new_nodes.sum(-1, keepdims=True)
        new_edges = new_edges / new_edges.sum(-1, keepdims=True)
        return self.__class__.create(
            nodes=new_nodes,
            edges=new_edges,
            nodes_mask=self.nodes_mask,
            edges_mask=self.edges_mask,
        )


@typed
def add_dense(
    g: DenseGraphDistribution, other: DenseGraphDistribution
) -> DenseGraphDistribution:
    return create_dense(
        nodes=g.nodes + other.nodes,
        edges=g.edges + other.edges,
        nodes_mask=g.nodes_mask,
        edges_mask=g.edges_mask,
    )

    #
    # else:
    #     return self.__class__.create(
    #         nodes=self.nodes + other,
    #         edges=self.edges + other,
    #         nodes_mask=self.nodes_mask,
    #         edges_mask=self.edges_mask,
    #     )


@typed
def at(
    g: DenseGraphDistribution,
    other: OneHotGraph,
) -> GraphDistribution:
    return create_one_hot(
        nodes=g.nodes * other.nodes,
        edges=g.edges * other.edges,
        nodes_mask=g.nodes_mask,
        edges_mask=g.edges_mask,
    )
    # return self.__class__.create(
    #     nodes=g.nodes * other[..., None, None],
    #     edges=g.edges * other[..., None, None, None],
    #     nodes_mask=g.nodes_mask,
    #     edges_mask=g.edges_mask,
    # )
    if isinstance(other, Array):
        return self.__class__.create(
            nodes=self.nodes * other[..., None, None],
            edges=self.edges * other[..., None, None, None],
            nodes_mask=self.nodes_mask,
            edges_mask=self.edges_mask,
        )
    elif isinstance(other, GraphDistribution):
        return self.__class__.create(
            nodes=self.nodes * other.nodes,
            edges=self.edges * other.edges,
            nodes_mask=self.nodes_mask,
            edges_mask=self.edges_mask,
            _safe=_safe,
        )


@typed
def logprobs_at(
    g: DenseGraphDistribution, one_hot: OneHotGraph
) -> Float[Array, "batch_size"]:
    """Returns the probability of the given one-hot vector."""
    probs_at_vector = at(g, one_hot)
    nodes_mask = g.nodes_mask
    edges_mask = g.edges_mask
    nodes_logprob = (
        np.log(np.where(nodes_mask, probs_at_vector.nodes.sum(-1), 1)) * nodes_mask
    )
    edges_logprob = (
        np.log(np.where(edges_mask, probs_at_vector.edges.sum(-1), 1)) * edges_mask
    )
    res = nodes_logprob.sum(-1) + edges_logprob.sum((-1, -2))
    return res


@typed
def to_symmetric(edges: EdgeDistribution) -> EdgeDistribution:
    upper = einop(
        np.triu(np.ones((edges.shape[1], edges.shape[2]))), "n1 n2 -> 1 n1 n2 1"
    )
    return np.where(upper, edges, einop(edges, "b n1 n2 ee -> b n2 n1 ee"))


@typed
def sample_one_hot(g: DenseGraphDistribution, rng_key: Key) -> OneHotGraph:
    """Sample features from multinomial distribution with given probabilities (probX, probE, proby)
    :param probX: bs, n, dx_out        node features
    :param probE: bs, n, n, de_out     edge features
    :param rng_key: random.PRNGKey     random key for JAX operations
    """
    bs, n, ne = g.nodes.shape
    _, _, _, ee = g.edges.shape
    epsilon = 1e-8
    # mask = self.mask
    prob_x = g.nodes
    prob_e = g.edges

    # Noise X
    # The masked rows should define probability distributions as well
    # probX = probX.at[~node_mask].set(1 / probX.shape[-1])  # , probX)
    # prob_x = np.clip(prob_x, epsilon, 1 - epsilon)
    # Flatten the probability tensor to sample with categorical distribution
    prob_x = np.where(g.nodes_mask[..., None], prob_x, 1 / prob_x.shape[-1])
    prob_x = prob_x / prob_x.sum(-1, keepdims=True)
    # prob_x = einop(prob_x, "bs n ne -> (bs n) ne")  # (bs * n, dx_out)
    prob_x = logit(prob_x)
    # try:

    # logit_x = logit(prob_x)
    # except:
    #     ipdb.set_trace()
    rng_key, subkey = random.split(rng_key)
    x_t = random.categorical(subkey, prob_x, axis=-1)  # (bs * n,)

    # x_t = einop(x_t, "(bs n) -> bs n", n=n)  # (bs, n)

    prob_e = np.where(g.edges_mask[..., None], prob_e, 1 / prob_e.shape[-1])
    prob_e = prob_e / prob_e.sum(-1, keepdims=True)
    # prob_e = einop(
    #     prob_e,
    #     "bs n1 n2 ee -> (bs n1 n2) ee",
    # )
    prob_e = logit(prob_e)
    rng_key, subkey = random.split(rng_key)
    e_t = random.categorical(subkey, prob_e, axis=-1)
    # e_t = einop(e_t, "(bs n1 n2) -> bs n1 n2", n1=n, n2=n)
    embedded_x = jax.nn.one_hot(x_t, num_classes=ne)
    embedded_e = jax.nn.one_hot(e_t, num_classes=ee)
    embedded_e = to_symmetric(embedded_e)

    return create_one_hot_and_mask(
        nodes=embedded_x,
        edges=embedded_e,
        nodes_mask=g.nodes_mask,
        edges_mask=g.edges_mask,
    )


@jax.jit
def matmul(g: OneHotGraph, q: Q):
    x = g.nodes @ q.nodes
    e = g.edges @ q.edges[:, None]
    return create_dense_and_mask(
        nodes=x, edges=e, nodes_mask=g.nodes_mask, edges_mask=g.edges_mask
    )


@typed
def mask_dense(g: DenseGraphDistribution) -> DenseGraphDistribution:
    nodes = np.where(g.nodes_mask[..., None], g.nodes, 0)
    edges = np.where(g.edges_mask[..., None], g.edges, 0)
    return create_dense(
        nodes=nodes,
        edges=edges,
        nodes_mask=g.nodes_mask,
        edges_mask=g.edges_mask,
    )


@typed
def mask_one_hot(g: OneHotGraph) -> OneHotGraph:
    nodes = np.where(g.nodes_mask[..., None], g.nodes, 0)
    edges = np.where(g.edges_mask[..., None], g.edges, 0)
    return create_one_hot(
        nodes=nodes,
        edges=edges,
        nodes_mask=g.nodes_mask,
        edges_mask=g.edges_mask,
    )


@jax.jit
def sum(g: GraphDistribution) -> Array:
    node_mask, edge_mask = g.nodes_mask, g.edges_mask
    nodes = (g.nodes * node_mask[..., None]).mean(-1)
    edges = (g.edges * edge_mask[..., None]).mean(-1)
    return np.einsum("bn->b", nodes) + np.einsum("bjk->b", edges)


@typed
def create_dense_minimal(
    nodes: NodeDistribution,
    edges: EdgeDistribution,
    nodes_mask: NodeMaskType,
):
    _, edges_mask = get_masks(nodes_mask.sum(-1), edges.shape[1])
    return DenseGraphDistribution(
        nodes=nodes,
        edges=edges,
        nodes_mask=nodes_mask,
        edges_mask=edges_mask,
        _created_internally=True,
    )


@typed
def create_one_hot_minimal(
    nodes: NodeDistribution,
    edges: EdgeDistribution,
    nodes_mask: NodeMaskType,
):
    _, edges_mask = get_masks(nodes_mask.sum(-1), edges.shape[1])
    return OneHotGraph(
        nodes=nodes,
        edges=edges,
        nodes_mask=nodes_mask,
        edges_mask=edges_mask,
        _created_internally=True,
    )


@typed
def create_dense(
    nodes: NodeDistribution,
    edges: EdgeDistribution,
    nodes_mask: NodeMaskType,
    edges_mask: EdgeMaskType,
):
    return DenseGraphDistribution(
        nodes=nodes,
        edges=edges,
        nodes_mask=nodes_mask,
        edges_mask=edges_mask,
        _created_internally=True,
    )


@typed
def create_one_hot(
    nodes: NodeDistribution,
    edges: EdgeDistribution,
    nodes_mask: NodeMaskType,
    edges_mask: EdgeMaskType,
):
    return OneHotGraph(
        nodes=nodes,
        edges=edges,
        nodes_mask=nodes_mask,
        edges_mask=edges_mask,
        _created_internally=True,
    )


@typed
def create_dense_and_mask(
    nodes: NodeDistribution,
    edges: EdgeDistribution,
    nodes_mask: NodeMaskType,
    edges_mask: EdgeMaskType,
):
    nodes = np.where(nodes_mask[..., None], nodes, 0)
    edges = np.where(edges_mask[..., None], edges, 0)
    return DenseGraphDistribution(
        nodes=nodes,
        edges=edges,
        nodes_mask=nodes_mask,
        edges_mask=edges_mask,
        _created_internally=True,
    )


@typed
def create(
    nodes: NodeDistribution,
    edges: EdgeDistribution,
    nodes_mask: NodeMaskType,
    edges_mask: EdgeMaskType,
):
    return GraphDistribution(
        nodes=nodes,
        edges=edges,
        nodes_mask=nodes_mask,
        edges_mask=edges_mask,
        _created_internally=True,
    )


@typed
def create_one_hot_and_mask(
    nodes: NodeDistribution,
    edges: EdgeDistribution,
    nodes_mask: NodeMaskType,
    edges_mask: EdgeMaskType,
):
    nodes = np.where(nodes_mask[..., None], nodes, 0)
    edges = np.where(edges_mask[..., None], edges, 0)
    return OneHotGraph(
        nodes=nodes,
        edges=edges,
        nodes_mask=nodes_mask,
        edges_mask=edges_mask,
        _created_internally=True,
    )


@typed
def get_masks(
    nodes_counts: Int[Array, "bs"], n: SInt
) -> tuple[NodeMaskType, EdgeMaskType]:
    n_range = np.arange(n)
    bs = nodes_counts.shape[0]
    mask_x = einop(n_range, "n -> bs n", bs=bs) < einop(nodes_counts, "bs -> bs n", n=n)
    e_ranges = einop(mask_x, "bs n -> bs n n1", n1=n)
    e_diag = einop(np.eye(n, dtype=bool), "n1 n2 -> 1 n1 n2")
    mask_e = e_ranges & einop(e_ranges, "bs n n1 -> bs n1 n") & ~e_diag
    return mask_x, mask_e


@typed
def create_one_hot_from_counts(
    nodes: NodeDistribution,
    edges: EdgeDistribution,
    nodes_counts: EdgeCountType,
):
    # is_nodes_dist = np.logical_or(is_dist(nodes), (~_safe))
    # if not is_nodes_dist:
    #     ipdb.set_trace()
    # pseudo_assert(is_nodes_dist)
    # is_edges_dist = np.logical_or(is_dist(edges), (~_safe))
    # if not is_edges_dist:
    #     ipdb.set_trace()
    # pseudo_assert(is_edges_dist)
    nodes_mask, edges_mask = get_masks(nodes_counts, nodes.shape[1])
    return OneHotGraph(
        nodes=nodes,
        edges=edges,
        nodes_mask=nodes_mask,
        edges_mask=edges_mask,
        _created_internally=True,
    )


@typed
def create_dense_from_counts(
    nodes: NodeDistribution,
    edges: EdgeDistribution,
    nodes_counts: EdgeCountType,
):
    nodes_mask, edges_mask = get_masks(nodes_counts, nodes.shape[1])
    return DenseGraphDistribution(
        nodes=nodes,
        edges=edges,
        nodes_mask=nodes_mask,
        edges_mask=edges_mask,
        _created_internally=True,
    )
