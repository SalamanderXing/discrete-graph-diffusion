"""
A lot of functions related to diffusion models. But which do not depend on the on the model itself.
"""
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

from ...shared.graph_distribution import GraphDistribution
from .types import TransitionModel

GetProbabilityType = Callable[
    [GraphDistribution, Int[Array, "batch_size"]], GraphDistribution
]
check = lambda _, __: None


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
    t = np.ones(g.batch_size, dtype=int)
    q_t = transition_model.qs[t]
    g_t = (g @ q_t).sample_one_hot(rng_key)
    g_t_probs = get_probability(g_t, t)
    result = g_t_probs.logprobs_at(g)
    return result / np.log(base)


@typed
def compute_lt(
    get_probability: GetProbabilityType,
    g: GraphDistribution,
    diffusion_steps: SInt,
    transition_model: TransitionModel,
    rng: Key,
) -> Float[Array, "batch_size"]:
    t = random.randint(rng, (g.batch_size,), 1, diffusion_steps + 1)
    q_t = transition_model.qs[t]
    q_s_bar = transition_model.q_bars[t - 1]
    g_q_t_bar = g @ transition_model.q_bars[t]

    g_t = g_q_t_bar.sample_one_hot(rng)
    p = get_probability(g_t, t)
    left_num = g @ q_t
    q_t_bar = transition_model.q_bars[t]

    # use bayes rule to compute q(z_s | z_t, g)
    mult = (g @ q_s_bar) / (g @ q_t_bar)
    # mult_x = mult.x[:, 0][:, None]
    # mult_e = mult.e[:, 0][:, None, None]
    q = left_num * mult
    # q = GraphDistribution.create(
    #     x=left_num.x * mult_x,
    #     e=left_num.e * mult_e,
    #     nodes_counts=left_num.nodes_counts,
    #     edges_counts=left_num.edges_counts,
    # )
    # result = graph_dist_kl_div(left_term, g_s_probs)
    # check_is_dist(q.x)
    # check_is_dist(q.e)
    mask_x, mask_e = p.masks()
    nodes_kl = (kl_div(q.nodes, p.nodes) * mask_x).sum(axis=1)
    edges_kl = (kl_div(q.edges, p.edges) * mask_e).sum(axis=(1, 2))
    result = nodes_kl + edges_kl
    return result * diffusion_steps


@typed
def kl_div(
    p: Array,
    q: Array,
    base: SFloat = np.e,
    eps: SFloat = 1**-17,
) -> Array:
    """Calculates the Kullback-Leibler divergence between arrays p and q."""
    p += eps
    q += eps
    return np.sum(p * np.log(p / q) / np.log(base), axis=-1)


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
    limit_X = np.broadcast_to(
        np.expand_dims(prior.x, (0, 1)), (bs, n, prior.x.shape[-1])
    )
    limit_X /= limit_X.sum(-1, keepdims=True)
    limit_E = np.broadcast_to(
        np.expand_dims(prior.e, (0, 1, 2)), (bs, n, n, prior.e.shape[-1])
    )

    def check_is_dist(vals):
        return (
            np.allclose(np.sum(vals, axis=-1), 1.0)
            and np.all(vals >= 0.0)
            and np.all(vals <= 1.0)
        )

    # assert check_is_dist(limit_X), ipdb.set_trace()
    # assert check_is_dist(limit_E), ipdb.set_trace()
    # assert check_is_dist(transition_probs.x), ipdb.set_trace()
    # assert check_is_dist(transition_probs.e), ipdb.set_trace()
    base = jax.lax.select(bits_per_edge, 2.0, np.e)
    mask_x, mask_e = target.masks()
    kl_distance_X = (kl_div(transition_probs.nodes, limit_X, base=base) * mask_x).sum(1)
    kl_distance_E = (kl_div(transition_probs.edges, limit_E, base=base) * mask_e).sum(
        (1, 2)
    )
    return kl_distance_X + kl_distance_E
