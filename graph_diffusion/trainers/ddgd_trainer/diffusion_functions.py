"""
A lot of functions related to diffusion models. But which do not depend on the on the model itself.
"""
import ipdb
from typing import Callable
import jax
from jax import Array, random
import jax
from jax.debug import print as jprint  # type: ignore
from jax import numpy as np
from flax import linen as nn
import mate as m
from mate.jax import SInt, SFloat, Key, typed, SBool
from jaxtyping import Int, Float, Bool
from einop import einop
from ...shared.graph import graph_distribution as gd
from .types import TransitionModel
from einop import einop

GraphDistribution = gd.GraphDistribution
EdgeDistribution = gd.EdgeDistribution
NodeDistribution = gd.NodeDistribution
Q = gd.Q

GetProbabilityType = Callable[
    [GraphDistribution, Int[Array, "batch_size"]], GraphDistribution
]
check = lambda _, __: None


# prob_true = diffusion_utils.posterior_distributions(
#     X=X,
#     E=E,
#     y=y,
#     X_t=noisy_data["X_t"],
#     E_t=noisy_data["E_t"],
#     y_t=noisy_data["y_t"],
#     Qt=Qt,
#     Qsb=Qsb,
#     Qtb=Qtb,
# )


# def compute_posterior_distribution(M, M_t, Qt_M, Qsb_M, Qtb_M):
#     """M: X or E
#     Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T
#
#     Compute xt @ Qt.T * x0 @ Qsb / sum(x0 @ Qtb * xt, -1)
#     """
#     # Flatten feature tensors
#     M = M.flatten(start_dim=1, end_dim=-2).to(
#         torch.float32
#     )  # (bs, N, d) with N = n or n * n
#     M_t = M_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)  # same
#
#     Qt_M_T = torch.transpose(Qt_M, -2, -1)  # (bs, d, d)
#
#     left_term = M_t @ Qt_M_T  # (bs, N, d)
#     right_term = M @ Qsb_M  # (bs, N, d)
#     product = left_term * right_term  # (bs, N, d)
#
#     denom = M @ Qtb_M  # (bs, N, d) @ (bs, d, d) = (bs, N, d)
#     denom = (denom * M_t).sum(dim=-1)  # (bs, N, d) * (bs, N, d) + sum = (bs, N)
#     # denom = product.sum(dim=-1)
#     # denom[denom == 0.] = 1
#
#     prob = product / denom.unsqueeze(-1)  # (bs, N, d)
#     return prob
#


@typed
def __compute_posterior_distribution_edges(
    edges: EdgeDistribution,
    edges_t: EdgeDistribution,
    q_t: Array,
    q_s_bar: Array,
    q_t_bar: Array,
):
    q_t_e_transposed = einop(
        q_t, "bs e1 e2 -> bs n1 n2 e2 e1", n1=edges.shape[1], n2=edges.shape[2]
    )
    left_term = einop(
        edges_t, q_t_e_transposed, "bs n1 n2 de, bs n1 n2 de e1 -> bs n1 n2 e1"
    )

    right_term = einop(edges, q_s_bar, "bs n1 n2 de, bs de e1 -> bs n1 n2 e1")
    product = left_term * right_term
    denom = einop(edges, q_t_bar, "bs n1 n2 de, bs de e1 -> bs n1 n2 e1")
    denom = einop(denom * edges_t, "bs n1 n2 e -> bs n1 n2", reduction="sum")
    denom = np.where(denom == 0, 1, denom)
    prob = product / denom[..., None]
    return prob


@typed
def __compute_posterior_distribution_nodes(
    nodes: NodeDistribution,
    nodes_t: NodeDistribution,
    q_t: Array,
    q_s_bar: Array,
    q_t_bar: Array,
):
    q_t_e_transposed = einop(q_t, "bs e1 e2 -> bs n e2 e1", n=nodes.shape[1])
    left_term = einop(nodes_t, q_t_e_transposed, "bs n de, bs n de e1 -> bs n e1")
    right_term = nodes @ q_s_bar
    product = left_term * right_term

    denom = nodes @ q_t_bar
    denom = einop(denom * nodes_t, "bs n e -> bs n", reduction="sum")
    denom = np.where(denom == 0, 1, denom)
    prob = product / denom[..., None]  # type:ignore
    return prob


# def posterior_distributions(X, E, y, X_t, E_t, y_t, Qt, Qsb, Qtb):
#     prob_X = compute_posterior_distribution(
#         M=X, M_t=X_t, Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X
#     )  # (bs, n, dx)
#     prob_E = compute_posterior_distribution(
#         M=E, M_t=E_t, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E
#     )  # (bs, n * n, de)
#
#     return PlaceHolder(X=prob_X, E=prob_E, y=y_t)
#


@typed
def posterior_distribution(
    g: GraphDistribution,
    g_t: GraphDistribution,
    transition_model: TransitionModel,
    t: Int[Array, "b"],
):
    q_t = transition_model.qs[t]
    q_t_bar = transition_model.q_bars[t]
    q_s_bar = transition_model.q_bars[t - 1]

    prob_x = __compute_posterior_distribution_nodes(
        nodes=g.nodes,
        nodes_t=g_t.nodes,
        q_t=q_t.nodes,
        q_s_bar=q_s_bar.nodes,
        q_t_bar=q_t_bar.nodes,
    )
    prob_e = __compute_posterior_distribution_edges(
        edges=g.edges,
        edges_t=g_t.edges,
        q_t=q_t.edges,
        q_s_bar=q_s_bar.edges,
        q_t_bar=q_t_bar.edges,
    )
    prob_x = prob_x / prob_x.sum(-1, keepdims=True)
    prob_e = prob_e / prob_e.sum(-1, keepdims=True)
    return GraphDistribution.create(
        nodes=prob_x,
        edges=prob_e,
        edges_counts=g.edges_counts,
        nodes_counts=g.nodes_counts,
    )


# get_probability: GetProbabilityType,
#     g: GraphDistribution,
#     transition_model: TransitionModel,
#     rng: Key,


@typed
def compute_lt(
    get_probability: GetProbabilityType,
    g: GraphDistribution,
    transition_model: TransitionModel,
    rng: Key,
):
    t = random.randint(rng, (g.batch_size,), 1, transition_model.diffusion_steps + 1)
    g_t: GraphDistribution = g @ transition_model.q_bars[t]
    g_pred: GraphDistribution = get_probability(g_t, t)
    # Compute distributions to compare with KL
    # bs, n, d = X.shape
    prob_true = posterior_distribution(
        g=g,
        g_t=g_t,
        transition_model=transition_model,
        t=t,
    )
    prob_pred = posterior_distribution(
        g_pred,
        g_t,
        transition_model=transition_model,
        t=t,
    )
    # Reshape and filter masked rows
    return transition_model.diffusion_steps * gd.kl_div(prob_true, prob_pred)


@typed
def sample_batch(
    rng_key: Key,
    get_probability: GetProbabilityType,
    batch_size: SInt,
    n: SInt,
    node_embedding_size: SInt,
    edge_embedding_size: SInt,
    diffusion_steps: SInt,
) -> GraphDistribution:
    random_batch = GraphDistribution.sample_from_uniform(
        key=rng_key,
        batch_size=batch_size,
        n=n,
        node_embedding_size=node_embedding_size,
        edge_embedding_size=edge_embedding_size,
    )
    for t in range(1, diffusion_steps + 1):
        t = np.ones(batch_size, dtype=int) * t
        random_batch = (get_probability(random_batch, t)).sample_one_hot(rng_key)
    return random_batch


@typed
def reconstruction_logp(
    *,
    rng_key: Key,
    get_probability: GetProbabilityType,
    g: GraphDistribution,
    transition_model: TransitionModel,
    n_samples: SInt,
    base: SFloat = 2,
):
    t = np.zeros(g.batch_size, dtype=int)
    q_t = transition_model.qs[t]
    g_t = (g @ q_t).sample_one_hot(rng_key)
    g_t_probs = get_probability(g_t, t)
    result = g_t_probs.logprobs_at(g)
    return result / np.log(base)


# @typed
# def compute_lt(
#     get_probability: GetProbabilityType,
#     g: GraphDistribution,
#     diffusion_steps: SInt,
#     transition_model: TransitionModel,
#     rng: Key,
# ) -> Float[Array, "batch_size"]:
#     t = random.randint(rng, (g.batch_size,), 1, diffusion_steps + 1)
#     q = transition_model.qs
#     q_bar = transition_model.q_bars
#
#     # q_t = transition_model.qs[t]
#     # q_s_bar = transition_model.q_bars[t - 1]
#     g_t = (g @ q_bar[t]).sample_one_hot(rng)
#     p = get_probability(g_t, t)
#     # f(G_{t-1}) = q(G_{t-1} | G, G_t) = q(G_t | G, G_{t-1})q(G_{t-1} | G) / q(G_t | G)
#     # q_t_bar = transition_model.q_bars[t]
#
#     # use bayes rule to compute q(z_s | z_t, g)
#     # __unsafe=True because at that multiplication, the graph is not a distribution.
#     # but it will be after the division.
#     q_num = (g @ q[t]).__mul__(g @ q_bar[t - 1], _safe=False)
#     denom = g @ q_bar[t]
#     # q_denom_nodes, q_denom_edges = denom.nodes.sum(-1), denom.edges.sum(-1)
#     # q = GraphDistribution.create(
#     #     q_num.nodes / q_num.nodes.sum(-1)[..., None],  # q_denom_nodes[..., None],
#     #     q_num.edges / q_num.edges.sum(-1)[..., None],  # q_denom_edges[..., None],
#     #     edges_counts=q_num.edges_counts,
#     #     nodes_counts=q_num.nodes_counts,
#     # )
#     q = q_num / denom
#     result = gd.kl_div(q, p)
#     return result * diffusion_steps
#


def check_is_dist(p):
    assert (
        np.allclose(p.sum(axis=-1), 1)
        and (0 <= p.min(axis=-1)).all()
        and (p.max(axis=-1) <= 1).all()
    )


@typed
def apply_random_noise(
    *,
    rng: Key,
    graph: GraphDistribution,
    diffusion_steps: SInt,
    transition_model: TransitionModel,
    test: SBool,
) -> tuple[Int[Array, "batch_size"], GraphDistribution]:
    lowest_t = np.array(test).astype(np.int32)
    bs = graph.nodes.shape[0]
    t = jax.random.randint(  # sample t_int from U[lowest_t, T]
        rng, (bs,), lowest_t, diffusion_steps + 1
    )
    return t, apply_noise(
        rng=rng,
        graph=graph,
        transition_model=transition_model,
        t=t,
    )


@typed
def apply_noise(
    *,
    rng: Key,
    graph: GraphDistribution,
    transition_model: TransitionModel,
    t: Int[Array, "batch_size"],
) -> GraphDistribution:
    """Sample noise and apply it to the data."""

    # get the transition matrix
    Qt_bar = transition_model.q_bars[t]

    # Compute transition probabilities
    prob_graph = graph @ Qt_bar

    # sampling returns an 1-hot encoded graph, so still a disctrete distribution
    return prob_graph.sample_one_hot(rng_key=rng)


@typed
def kl_prior(
    *,
    target: GraphDistribution,
    transition_model: TransitionModel,
    bits_per_edge: SBool,
) -> Float[Array, "batch_size"]:
    """Computes the KL between q(z1 | x) and the prior p(z1) (extracted from the data)

    This is essentially a lot of work for something that is in practice negligible in the loss. However, you
    compute it so that you see it when you've made a mistake in your noise schedule.
    """
    # Compute the last alpha value, alpha_T.
    qt_bar = transition_model.q_bars[np.array(-1)[None]]
    prior = transition_model.prior

    # Compute transition probabilities
    transition_probs = target @ qt_bar

    # turn the prior into a graph distribution
    bs, n, _ = transition_probs.nodes.shape

    def check_is_dist(vals):
        return (
            np.allclose(np.sum(vals, axis=-1), 1.0)
            and np.all(vals >= 0.0)
            and np.all(vals <= 1.0)
        )

    base = jax.lax.select(bits_per_edge, 2.0, np.e)
    limit_dist = transition_model.limit_dist.repeat(bs)
    return gd.kl_div(transition_probs, limit_dist, base=base)
