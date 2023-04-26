"""
A lot of functions related to diffusion models. But which do not depend on the on the model itself.
"""
from typing import Callable
import jax
import ipdb
from rich import print
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
from .diffusion_types import (
    GraphDistribution,
    XDistType,
    EDistType,
    YDistType,
    MaskType,
    # Forward,
)
from .diffusion_types import TransitionModel
from .utils import softmax_kl_div
from .extra_features import extra_features

check = lambda x, y: None


@typed
def reconstruction_logp(
    *,
    rng_key: Key,
    state: TrainState,
    noise_schedule,  # : NoiseSchedule,
    t: SInt,
    graph: GraphDistribution,
    transition_model: TransitionModel,
    get_extra_features,  #: Callable[[NoisyData], GraphDistribution],
    get_domain_features,  #: Callable[[NoisyData], GraphDistribution],
):
    e = graph.e
    x = graph.x
    mask = graph.mask
    # Compute noise values for t = 0.
    t_zeros = np.zeros_like(t)
    beta_0 = noise_schedule(t_zeros)
    Q0 = transition_model.get_Qt(beta_t=beta_0)

    probX0 = x @ Q0.x  # (bs, n, dx_out)
    probE0 = np.matmul(e, np.expand_dims(Q0.e, 1))  # (bs, n, n, de_out)

    sampled0 = sample_discrete_features(
        rng_key=rng_key, probX=probX0, probE=probE0, node_mask=mask
    )

    X0 = np.eye(x.shape[-1])[sampled0.x]
    E0 = np.eye(e.shape[-1])[sampled0.e]
    y0 = sampled0.y
    check((x.shape == X0.shape) and (e.shape == E0.shape), "Whoops")

    sampled_0 = GraphDistribution(x=X0, e=E0, y=y0, mask=mask)

    noisy_data = NoisyData(
        embedded_graph=sampled_0,
        t=np.zeros((X0.shape[0], 1), dtype=y0.dtype),
    )
    extra_data = extra_features(
        noisy_data=noisy_data
    )  # compute_extra_data(noisy_data, get_extra_features, get_domain_features)
    # pred0 = model.apply(noisy_data, extra_data, node_mask)
    # passes the parameters to the model and calls the apply
    x, e, y = state.apply_fn(state.params, noisy_data, extra_data, mask)
    pred0 = GraphDistribution(x=x, e=e, y=y, mask=mask)

    # Normalize predictions
    probX0 = jax.nn.softmax(pred0.x, axis=-1)
    probE0 = jax.nn.softmax(pred0.e, axis=-1)
    proby0 = jax.nn.softmax(pred0.y, axis=-1)

    # Set masked rows to arbitrary values that don't contribute to loss
    probX0 = probX0.at[np.logical_not(mask)].set(np.ones(x.shape[-1]))
    probE0 = probE0.at[
        np.logical_not(np.expand_dims(mask, 1) * np.expand_dims(mask, 2))
    ].set(np.ones(e.shape[-1]))

    diag_mask = np.eye(probE0.shape[1], dtype=bool)
    diag_mask = np.expand_dims(diag_mask, 0).repeat(probE0.shape[0], axis=0)
    probE0 = probE0.at[diag_mask].set(np.ones(e.shape[-1]))

    return GraphDistribution(x=probX0, e=probE0, y=proby0)


def compute_lt(
    get_probability: Callable[[GraphDistribution], GraphDistribution],
    g: GraphDistribution,
    n_t_samples: SInt,
    n_g_samples: SInt,
    diffusion_steps: SInt,
    transition_model: TransitionModel,
    rng: Key,
) -> SFloat:
    t_acc = 0
    for _ in range(n_t_samples):
        g_acc = 0
        t = random.randint(  # sample t_int from U[lowest_t, T]
            rng, (g.batch_size,), 1, diffusion_steps + 1
        )
        for _ in range(n_g_samples):
            q_t = transition_model.qs[t]
            q_t_bar = transition_model.q_bars[t]
            q_s_bar = transition_model.q_bars[t - 1]
            g_t = (g @ transition_model.q_bars[t]).sample_one_hot(rng)
            g_s_probs = get_probability(g_t)
            g_s = g_s_probs.sample_one_hot(rng)
            left_term = ((g_s @ q_t) * (g @ q_s_bar)) / (g @ q_t_bar)
            g_acc += graph_dist_kl_div(left_term, g_s_probs)
        t_acc += g_acc / n_g_samples
    return t_acc / n_t_samples


@typed
def posterior_distribution(
    t: SInt,
    original_embedded_graph: GraphDistribution,
    transition_model: TransitionModel,
) -> SFloat:
    q_t = transition_model.qs[t]
    q_t_bar = transition_model.q_bars[t]
    q_s_bar = transition_model.q_bars[t - 1]
    # dist_s_given_original = original_embedded_graph


# @typed
# def compute_posterior_distribution(
#     M: Array, M_t: Array, Qt_M: Array, Qsb_M: Array, Qtb_M: Array
# ) -> Array:
#     """M: X or E
#     Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T
#     """
#     # Flatten feature tensors
#     M = np.reshape(M, (M.shape[0], -1, M.shape[-1])).astype(
#         np.float32
#     )  # (bs, N, d) with N = n or n * n
#     M_t = np.reshape(M_t, (M_t.shape[0], -1, M_t.shape[-1])).astype(np.float32)  # same

#     Qt_M_T = np.transpose(Qt_M, (0, 2, 1))  # (bs, d, d)

#     left_term = M_t @ Qt_M_T  # (bs, N, d)
#     right_term = M @ Qsb_M  # (bs, N, d)
#     product = left_term * right_term  # (bs, N, d)

#     denom = M @ Qtb_M  # (bs, N, d) @ (bs, d, d) = (bs, N, d)
#     denom = (denom * M_t).sum(axis=-1)  # (bs, N, d) * (bs, N, d) + sum = (bs, N)

#     prob = product / denom[..., None]  # (bs, N, d)

#     return prob


# @typed
# def posterior_distributions(
#     graph: GraphDistribution, graph_t: GraphDistribution, Qt, Qsb, Qtb
# ) -> GraphDistribution:
#     return GraphDistribution.with_trivial_mask(
#         x=compute_posterior_distribution(
#             M=graph.x, M_t=graph_t.x, Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X
#         ),  # (bs, n, dx),
#         e=compute_posterior_distribution(
#             M=graph.e, M_t=graph_t.e, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E
#         ).reshape((graph.e.shape[0], graph.e.shape[1], graph.e.shape[1], -1)),
#         y=graph_t.y,
#     )


@typed
def kl_div(p: Array, q: Array, eps: SFloat = 2**-17) -> Array:
    """Calculates the Kullback-Leibler divergence between arrays p and q."""
    p += eps
    q += eps
    return np.sum(p * np.log(p / q), axis=-1)


@typed
def graph_dist_kl_div(p: GraphDistribution, q: GraphDistribution) -> SFloat:
    """Calculates the Kullback-Leibler divergence between arrays p and q."""
    return kl_div(p.x, q.x).mean() + kl_div(p.e, q.e).mean()  # + kl_div(p.y, q.y)


@typed
def mask_distributions(
    true_X: XDistType,
    true_E: EDistType,
    pred_X: XDistType,
    pred_E: EDistType,
    node_mask: MaskType,
) -> tuple[XDistType, EDistType, XDistType, EDistType]:
    # Add a small value everywhere to avoid nans
    pred_X += 1e-7
    pred_X = pred_X / np.sum(pred_X, axis=-1, keepdims=True)

    pred_E += 1e-7
    pred_E = pred_E / np.sum(pred_E, axis=-1, keepdims=True)

    # Set masked rows to arbitrary distributions, so it doesn't contribute to loss
    row_X = np.zeros(true_X.shape[-1])
    row_X = row_X.at[0].set(1.0)
    row_E = np.zeros(true_E.shape[-1])
    row_E = row_E.at[0].set(1.0)

    diag_mask = ~jax.numpy.eye(node_mask.shape[1], dtype=bool)[None]
    true_X = true_X.at[~node_mask].set(row_X)
    mask = ~(node_mask[:, None] * node_mask[:, :, None] * diag_mask)
    true_E = true_E.at[mask, :].set(row_E)
    pred_X = pred_X.at[~node_mask].set(row_X)
    pred_E = pred_E.at[mask,].set(row_E)

    return true_X, true_E, pred_X, pred_E


@typed
def apply_random_noise(
    *,
    rng: Key,
    graph: GraphDistribution,
    diffusion_steps: SInt,
    transition_model: TransitionModel,
    test: SBool,
) -> GraphDistribution:
    lowest_t = np.array(test).astype(np.int32)
    bs = graph.x.shape[0]
    t = jax.random.randint(  # sample t_int from U[lowest_t, T]
        rng, (bs,), lowest_t, diffusion_steps + 1
    )
    return apply_noise(
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
) -> SFloat:
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
        true_E=limit_E,
        pred_X=transition_probs.x,
        pred_E=transition_probs.e,
        node_mask=target.mask,
    )
    kl_distance_X = kl_div(probX, limit_dist_X).sum()
    kl_distance_E = kl_div(probE, limit_dist_E).sum()
    return kl_distance_X + kl_distance_E
