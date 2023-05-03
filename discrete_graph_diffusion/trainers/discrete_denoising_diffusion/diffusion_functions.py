"""
A lot of functions related to diffusion models. But which do not depend on the on the model itself.
"""
from typing import Callable
import jax
import ipdb
from jax import Array, random
import jax
from jax import numpy as np
from jax.experimental.checkify import check
from flax import linen as nn
from flax.training.train_state import TrainState
import mate as m
from mate.jax import SInt, SFloat, Key, typed, jit, SBool
from flax.training.train_state import TrainState
from jaxtyping import Int, Float, Bool

# from extra_features.cycle_features import batch_trace
from .diffusion_types import (
    GraphDistribution,
    XDistType,
    EDistType,
    MaskType,
    # Forward,
)
from .diffusion_types import TransitionModel
from .utils import softmax_kl_div

GetProbabilityType = Callable[
    [GraphDistribution, Int[Array, "batch_size"]], GraphDistribution
]
check = lambda x, y: None


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
):
    t = np.ones(g.batch_size, dtype=int)
    q_t = transition_model.qs[t]
    prob_acc = 0
    for _ in range(n_samples):
        g_t = (g @ q_t).sample_one_hot(rng_key)
        g_t_probs = get_probability(g_t, t)
        prob_acc += g_t_probs.probs_at(g)
    return np.log(prob_acc / n_samples)


@typed
def compute_lt(
    get_probability: GetProbabilityType,
    g: GraphDistribution,
    n_t_samples: SInt,
    n_g_samples: SInt,
    diffusion_steps: SInt,
    transition_model: TransitionModel,
    rng: Key,
) -> Float[Array, "batch_size"]:
    t_acc: Float[Array, "batch_size"] = np.zeros(g.batch_size)
    for _ in range(n_t_samples):
        g_acc: Float[Array, "batch_size"] = np.zeros(g.batch_size)
        t = random.randint(  # sample t_int from U[lowest_t, T]
            rng, (g.batch_size,), 1, diffusion_steps + 1
        )
        for _ in range(n_g_samples):
            q_t = transition_model.qs[t]
            q_s_bar = transition_model.q_bars[t - 1]
            g_q_t_bar = g @ transition_model.q_bars[t]
            g_t = g_q_t_bar.sample_one_hot(rng)
            g_s_probs = get_probability(g_t, t)
            left_term = ((g @ q_t) * (g @ q_s_bar)) / (g_q_t_bar)
            g_acc += graph_dist_kl_div(left_term, g_s_probs)
        t_acc += g_acc / n_g_samples
    return diffusion_steps * (t_acc / n_t_samples)


@typed
def kl_div(p: Array, q: Array, eps: SFloat = 2**-17) -> Array:
    """Calculates the Kullback-Leibler divergence between arrays p and q."""
    p += eps
    q += eps
    return np.sum(p * np.log(p / q), axis=-1)


def softmax_kl_div(tensor1, tensor2, reduction="batchsum"):
    # Subtract maximum value for numerical stability
    tensor1_max, tensor2_max = (
        tensor1.max(axis=-1, keepdims=True),
        tensor2.max(axis=-1, keepdims=True),
    )
    tensor1_stable, tensor2_stable = tensor1 - tensor1_max, tensor2 - tensor2_max

    # Compute log-sum-exp for both tensors
    log_sum_exp1 = jax.scipy.special.logsumexp(tensor1_stable, axis=-1, keepdims=True)
    log_sum_exp2 = jax.scipy.special.logsumexp(tensor2_stable, axis=-1, keepdims=True)

    # Compute the difference between input tensors and their log-sum-exp values
    tensor1_diff = tensor1_stable - log_sum_exp1
    tensor2_diff = tensor2_stable - log_sum_exp2

    # Calculate the proportional softmax values for tensor1
    proportional_softmax1 = np.exp(tensor1_diff)

    # Normalize the softmax values by dividing by the sum along the last dimension
    normalized_softmax1 = proportional_softmax1 / proportional_softmax1.sum(
        axis=-1, keepdims=True
    )

    # Calculate the KL divergence without explicitly computing the softmax values
    kl_div = normalized_softmax1 * (tensor1_diff - tensor2_diff)

    if reduction == "batchmean":
        kl_div = kl_div.sum(axis=-1).mean()
    elif reduction == "batchsum":
        kl_div = kl_div.sum(axis=-1)
    elif reduction == "none":
        pass  # Keep the element-wise KL divergence values as is
    elif reduction == "mean":
        kl_div = kl_div.mean()
    else:
        raise ValueError(
            f"Invalid reduction mode. Got {reduction}. Choose from ['batchmean', 'batchsum', 'none']"
        )
    return kl_div


@typed
def graph_dist_kl_div(p: GraphDistribution, q: GraphDistribution) -> Float[Array, "b"]:
    """Calculates the Kullback-Leibler divergence between graph distributions p and q."""
    mxp, mep, mxq, meq = mask_distributions(p.x, p.e, q.x, q.e, p.mask)
    mxp_reshaped = mxp.reshape((mxp.shape[0] * mxp.shape[1], -1))
    mxq_reshaped = mxq.reshape((mxq.shape[0] * mxq.shape[1], -1))
    mep_reshaped = mep.reshape((mep.shape[0] * mep.shape[1] * mep.shape[2], -1))
    meq_reshaped = meq.reshape((meq.shape[0] * meq.shape[1] * meq.shape[2], -1))
    a = (
        kl_div(jax.nn.softmax(mxp_reshaped), jax.nn.softmax(mxq_reshaped))
        .reshape((mxp.shape[0], mxp.shape[1]))
        .mean(axis=-1)
    )
    b = (
        kl_div(jax.nn.softmax(mep_reshaped), jax.nn.softmax(meq_reshaped))
        .reshape((mep.shape[0], mep.shape[1] * meq.shape[2]))
        .mean(axis=-1)
    )
    return a + b


@typed
def mask_distributions(
    true_X: XDistType,
    true_e: EDistType,
    pred_x: XDistType,
    pred_e: EDistType,
    node_mask: MaskType,
) -> tuple[XDistType, EDistType, XDistType, EDistType]:
    # Add a small value everywhere to avoid nans
    pred_x += 1e-7
    pred_x = pred_x / np.sum(pred_x, axis=-1, keepdims=True)

    pred_e += 1e-7
    pred_e = pred_e / np.sum(pred_e, axis=-1, keepdims=True)

    # Set masked rows to arbitrary distributions, so it doesn't contribute to loss
    row_X = np.zeros(true_X.shape[-1])
    row_X = row_X.at[0].set(1.0)
    row_E = np.zeros(true_e.shape[-1])
    row_E = row_E.at[0].set(1.0)

    diag_mask = ~jax.numpy.eye(node_mask.shape[1], dtype=bool)[None]
    # true_X = true_X.at[~node_mask].set(row_X)
    true_x = np.where(node_mask[..., None], true_X, row_X)
    mask = ~(node_mask[..., None] * node_mask[:, :, None] * diag_mask)[..., None]
    # true_E = true_E.at[mask, :].set(row_E)
    true_e = np.where(mask, true_e, row_E)
    # pred_X = pred_X.at[~node_mask].set(row_X)
    pred_x = np.where(node_mask[..., None], pred_x, row_X)
    # pred_E = pred_E.at[mask,].set(row_E)
    pred_e = np.where(mask, pred_e, row_E)
    return true_x, true_e, pred_x, pred_e


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
    bs = graph.x.shape[0]
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
    diffusion_steps: SInt,
    transition_model: TransitionModel,
) -> Float[Array, "batch_size"]:
    """Computes the KL between q(z1 | x) and the prior p(z1) (extracted from the data)

    This is essentially a lot of work for something that is in practice negligible in the loss. However, you
    compute it so that you see it when you've made a mistake in your noise schedule.
    """
    # Compute the last alpha value, alpha_T.
    timesteps = diffusion_steps * np.ones(target.x.shape[0], int)
    qt_bar = transition_model.q_bars[timesteps]
    prior = transition_model.prior

    # Compute transition probabilities
    transition_probs = target @ qt_bar
    bs, n, _ = transition_probs.x.shape
    limit_X = np.broadcast_to(
        np.expand_dims(prior.x, (0, 1)), (bs, n, prior.x.shape[-1])
    )
    limit_E = np.broadcast_to(
        np.expand_dims(prior.e, (0, 1, 2)), (bs, n, n, prior.e.shape[-1])
    )
    # Make sure that masked rows do not contribute to the loss
    limit_dist_X, limit_dist_E, probX, probE = mask_distributions(
        true_X=limit_X,
        true_e=limit_E,
        pred_x=transition_probs.x,
        pred_e=transition_probs.e,
        node_mask=target.mask,
    )
    kl_distance_X = kl_div(probX, limit_dist_X).mean(1)
    kl_distance_E = kl_div(probE, limit_dist_E).mean((1, 2))
    return kl_distance_X + kl_distance_E
