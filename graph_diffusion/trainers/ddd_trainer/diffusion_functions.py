"""
A lot of functions related to diffusion models. But which do not depend on the on the model itself.
"""
from typing import Callable
import jax
import ipdb
from jax import Array, random
import jax
from jax.debug import print as jprint  # type: ignore
from jax import numpy as np
from jax.experimental.checkify import check
from flax import linen as nn
from flax.training.train_state import TrainState
import mate as m
from mate.jax import SInt, SFloat, Key, typed, jit, SBool
from flax.training.train_state import TrainState
from jaxtyping import Int, Float, Bool

# from extra_features.cycle_features import batch_trace
from .types import GraphDistribution, XDistType, EDistType, MaskType, TransitionModel
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
    base: SFloat = 2,
):
    t = np.ones(g.batch_size, dtype=int)
    q_t = transition_model.qs[t]
    prob_acc = 0
    for _ in range(n_samples):
        g_t = (g @ q_t).sample_one_hot(rng_key)
        g_t_probs = get_probability(g_t, t)
        prob_acc += g_t_probs.probs_at(g)
    return np.log(prob_acc / n_samples) / np.log(base)


@typed
def compute_lt_meat(g, q_t, q_s_bar, g_q_t_bar, t, get_probability, rng):
    g_t = g_q_t_bar.sample_one_hot(rng)
    g_s_probs = get_probability(g_t, t)
    left_num = g @ q_t
    right_num = g @ q_s_bar
    left_term = ((left_num) * (right_num)) / (g_q_t_bar)
    ciao = graph_dist_kl_div(left_term, g_s_probs, g)
    return ciao


@typed
def compute_lt(
    get_probability: GetProbabilityType,
    g: GraphDistribution,
    n_t_samples: SInt,
    diffusion_steps: SInt,
    transition_model: TransitionModel,
    rng: Key,
) -> Float[Array, "batch_size"]:
    t_acc: Float[Array, "batch_size"] = np.zeros(g.batch_size)
    expanded_steps = np.arange(1, (diffusion_steps + 1))[:, None].repeat(
        g.batch_size, axis=1
    )
    ts = (
        random.randint(  # sample t_int from U[lowest_t, T]
            rng,
            (
                n_t_samples,
                g.batch_size,
            ),
            1,
            diffusion_steps + 1,  # type: ignore
        )
        if n_t_samples > 0
        else expanded_steps
    )
    for i in range(n_t_samples if n_t_samples > 0 else diffusion_steps):
        t = ts[i]
        q_t = transition_model.qs[t]
        q_s_bar = transition_model.q_bars[t - 1]
        g_q_t_bar = g @ transition_model.q_bars[t]

        # g_t = g_q_t_bar.sample_one_hot(rng)
        # g_s_probs = get_probability(g_t, t)
        # left_term = ((g @ q_t) * (g @ q_s_bar)) / (g_q_t_bar)
        # t_acc += graph_dist_kl_div(left_term, g_s_probs)
        t_cur = compute_lt_meat(g, q_t, q_s_bar, g_q_t_bar, t, get_probability, rng)
        t_acc += t_cur
    # if (t_acc != 0).any():
    value = jax.lax.select(
        n_t_samples > 0,
        (diffusion_steps * (t_acc / n_t_samples)),
        t_acc,
    )
    return value


@typed
def kl_div(p: Array, q: Array, eps: SFloat = 2**-17, base=2) -> Array:
    """Calculates the Kullback-Leibler divergence between arrays p and q."""
    p += eps
    q += eps
    # return np.sum(p * np.log(p / q), axis=-1)
    return np.sum(p * np.log(p / q) / np.log(base), axis=-1)


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


def mask_x(x, mask) -> Array:
    arbitrary_x = np.zeros_like(x[0, 0])
    arbitrary_x = arbitrary_x.at[0].set(1)
    x = x.at[~mask, :].set(arbitrary_x)
    return x


def mask_e(e, mask) -> Array:
    arbitrary_e = np.zeros_like(e[0, 0, 0])
    arbitrary_e = arbitrary_e.at[0].set(1)
    e = e.at[~mask, :, :].set(arbitrary_e)
    e = np.swapaxes(np.swapaxes(e, 1, 2).at[~mask, :, :].set(arbitrary_e), 1, 2)
    return e


def check_is_dist(p):
    assert (
        np.allclose(p.sum(axis=-1), 1)
        and (0 <= p.min(axis=-1)).all()
        and (p.max(axis=-1) <= 1).all()
    )


@typed
def graph_dist_kl_div(
    p: GraphDistribution, q: GraphDistribution, g: GraphDistribution
) -> Float[Array, "b"]:
    """Calculates the Kullback-Leibler divergence between graph distributions p and q."""

    # mask = g.node_counts
    ranges = np.arange(p.x.shape[1])[None].repeat(p.x.shape[0], axis=0)
    mask = ranges < g.nodes_counts[:, None]
    # mxp, mep, mxq, meq = mask_distributions(p.x, p.e, q.x, q.e, p.mask)
    mxp, mep, mxq, meq = p.x, p.e, q.x, q.e
    mxp = mask_x(mxp, mask)
    mep = mask_e(mep, mask)
    mxq = mask_x(mxq, mask)
    meq = mask_e(meq, mask)
    # arbitrary_e = np.zeros_like(mxp[0])
    # arbitrary_e = arbitrary_e.at[0].set(1)
    # mxp = np.where(
    #     p.mask,
    #     mxp,
    # )
    # mxp = mask_x(mxp, p.mask)
    # mep = mask_e(mep, p.mask)
    # mxq = mask_x(mxq, p.mask)
    # meq = mask_e(meq, p.mask)
    mxp_reshaped = mxp.reshape((mxp.shape[0] * mxp.shape[1], -1))
    mxq_reshaped = mxq.reshape((mxq.shape[0] * mxq.shape[1], -1))
    mep_reshaped = mep.reshape((mep.shape[0] * mep.shape[1] * mep.shape[2], -1))
    meq_reshaped = meq.reshape((meq.shape[0] * meq.shape[1] * meq.shape[2], -1))

    mxp_reshaped /= mxp_reshaped.sum(axis=-1, keepdims=True)
    # mxq_reshaped /= mxq_reshaped.sum(axis=-1, keepdims=True)
    mep_reshaped /= mep_reshaped.sum(axis=-1, keepdims=True)
    # meq_reshaped /= meq_reshaped.sum(axis=-1, keepdims=True)

    # mxp_reshaped_soft = jax.nn.softmax(mxp_reshaped, -1)
    # mxp_reshaped_soft_soft = jax.nn.softmax(mxp_reshaped, -1)
    # mep_reshaped_soft = jax.nn.softmax(mep_reshaped, -1)
    # ipdb.set_trace()

    check_is_dist(mxp_reshaped)
    check_is_dist(mxq_reshaped)
    check_is_dist(mep_reshaped)
    check_is_dist(meq_reshaped)
    # assert ((mxp_reshaped.sum(axis=-1) == 0) == (mxq_reshaped.sum(axis=-1) == 0)).all()
    # uba = np.where((mep_reshaped.sum(axis=-1) == 0) == (meq_reshaped.sum(axis=-1) == 0))
    # assert ((mep_reshaped.sum(axis=-1) == 0) == (meq_reshaped.sum(axis=-1) == 0)).all()
    # checks that all of the above are distributions
    a = (
        kl_div(mxp_reshaped, mxq_reshaped)
        .reshape((mxp.shape[0], mxp.shape[1]))
        .sum(axis=-1)
    )
    b = (
        kl_div(mep_reshaped, meq_reshaped)
        .reshape((mep.shape[0], mep.shape[1] * meq.shape[2]))
        .sum(axis=-1)
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
    # # Make sure that masked rows do not contribute to the loss
    # limit_dist_X, limit_dist_E, probX, probE = mask_distributions(
    #     true_X=limit_X,
    #     true_e=limit_E,
    #     pred_x=transition_probs.x,
    #     pred_e=transition_probs.e,
    #     node_mask=target.mask,
    # )
    kl_distance_X = kl_div(transition_probs.x, limit_X).sum(1)
    kl_distance_E = kl_div(transition_probs.e, limit_E).sum((1, 2))
    return kl_distance_X + kl_distance_E
