"""
A lot of functions related to diffusion models. But which do not depend on the on the model itself.
"""
from typing import Callable
import jax
import ipdb
from rich import print
from jax import Array
import jax
from jax import numpy as np
from flax import linen as nn
from flax.training.train_state import TrainState
import mate as m
from mate.jax import typed
from mate.jax import SInt, SFloat, Key
from jaxtyping import Int, Float, Bool

from .diffusion_types import TransitionModel
from .sample import sample_discrete_features
from .utils import softmax_kl_div
from .diffusion_types import GraphDistribution, NoisyData, Distribution, NoiseSchedule
from .extra_features import extra_features


@typed
def reconstruction_logp(
    *,
    rng_key: Key,
    state: TrainState,
    noise_schedule: NoiseSchedule,
    t: SInt,
    graph: GraphDistribution,
    transition_model: TransitionModel,
    get_extra_features: Callable[[NoisyData], GraphDistribution],
    get_domain_features: Callable[[NoisyData], GraphDistribution],
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
    assert (x.shape == X0.shape) and (e.shape == E0.shape)

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


@typed
def compute_Lt(
    *,
    target: GraphDistribution,
    pred: GraphDistribution,
    noisy_data: NoisyData,
    T: SFloat,
    transition_model: TransitionModel,
    t: SInt,
) -> Array:
    pred_probs = GraphDistribution.with_trivial_mask(
        x=jax.nn.softmax(pred.x, axis=-1),
        e=jax.nn.softmax(pred.e, axis=-1),
        y=jax.nn.softmax(pred.y, axis=-1),
    )

    Qt_bar = transition_model.get_Qt_bar(
        noisy_data.alpha_t_bar
    )  # this is the transitin matrix up to time t
    Qs_bar = transition_model.get_Qt_bar(
        noisy_data.alpha_s_bar
    )  # this is the transitin matrix up to time t - 1
    Qt = transition_model.get_Qt(noisy_data.beta_t)

    # Compute distributions to compare with KL
    prob_true = posterior_distributions(
        target,
        noisy_data.embedded_graph,
        Qt=Qt,
        Qsb=Qs_bar,
        Qtb=Qt_bar,
    )
    prob_pred_unmasked = posterior_distributions(
        pred_probs,
        noisy_data.embedded_graph,
        Qt=Qt,
        Qsb=Qs_bar,
        Qtb=Qt_bar,
    )
    # Reshape and filter masked rows
    (
        _,
        _,
        prob_pred_x,
        prob_pred_e,
    ) = mask_distributions(
        true_X=prob_true.x,
        true_E=prob_true.e,
        pred_X=prob_pred_unmasked.x,
        pred_E=prob_pred_unmasked.e,
        node_mask=target.mask,
    )
    kl_x = softmax_kl_div(prob_true.x, prob_pred_x)
    kl_e = softmax_kl_div(prob_true.e, prob_pred_e)
    return T * (kl_x + kl_e)


@typed
def posterior_distribution(
    t: SInt,
    original_embedded_graph: GraphDistribution,
    transition_model: TransitionModel,
) -> SFloat:
    q_t = transition_model.qs[t]
    q_t_bar = transition_model.q_bars[t]
    q_s_bar = transition_model.q_bars[t - 1]
    #dist_s_given_original = original_embedded_graph




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


def kl_div(p: Array, q: Array, eps: float = 2**-17) -> Array:
    """Calculates the Kullback-Leibler divergence between arrays p and q."""
    return p.dot(np.log(p + eps) - np.log(q + eps))


@typed
def mask_distributions(
    true_X: Array, true_E: Array, pred_X: Array, pred_E: Array, node_mask: Array
) -> tuple[Array, Array, Array, Array]:
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

    diag_mask = ~jax.numpy.eye(node_mask.shape[1], dtype=bool)[None, ...]
    true_X = true_X.at[~node_mask].set(row_X)
    true_E = true_E.at[
        ~(node_mask[:, None, None] & node_mask[:, None] & diag_mask), :
    ].set(row_E)
    pred_X = pred_X.at[~node_mask].set(row_X)
    pred_E = pred_E.at[
        ~(node_mask[:, None, None] & node_mask[:, None] & diag_mask), :
    ].set(row_E)

    return true_X, true_E, pred_X, pred_E


@typed
def apply_random_noise(
    *,
    rng: Key,
    graph: GraphDistribution,
    T: SInt,
    noise_schedule: NoiseSchedule,
    transition_model: TransitionModel,
    training: Bool[Array, ""],
) -> NoisyData:
    lowest_t = 0 if training else 1
    t_int = jax.random.randint(  # sample t_int from U[lowest_t, T]
        rng, (graph.x.shape[0], 1), lowest_t, T + 1
    )
    return apply_noise(
        rng=rng,
        graph=graph,
        T=T,
        noise_schedule=noise_schedule,
        transition_model=transition_model,
        t_int=t_int,
    )


@typed
def apply_noise(
    *,
    rng: Key,
    graph: GraphDistribution,
    T: SInt,
    noise_schedule: NoiseSchedule,
    transition_model: TransitionModel,
    t_int: Int[Array, "batch_size 1"],
) -> NoisyData:
    """Sample noise and apply it to the data."""

    s_int = t_int - 1

    t_float = t_int / T  # current normalized timestep

    # beta_t and alpha_s_bar are used for denoising/loss computation
    beta_t = noise_schedule.betas[t_int]  # (bs, 1) beta_t is the noise level at t
    alpha_s_bar = noise_schedule.alphas_bar[s_int]  # (bs, 1)
    alpha_t_bar = noise_schedule.alphas_bar[s_int]  # (bs, 1)

    Qt_bar = transition_model.q_bars[t_int]
    # .get_Qt_bar(
    #    alpha_t_bar
    # )  # (bs, dx_in, dx_out), (bs, de_in, de_out)
    assert (abs(Qt_bar.x.sum(axis=2) - 1.0) < 1e-4).all(), Qt_bar.x.sum(axis=2) - 1
    assert (abs(Qt_bar.e.sum(axis=2) - 1.0) < 1e-4).all()
    # Compute transition probabilities
    #probX = graph.x @ Qt_bar.x  # (bs, n, dx_out)
    #probE = graph.e @ Qt_bar.e[:, None]  # (bs, n, n, de_out)
    prob_graph = graph @ Qt_bar

    sampled_graph_t = sample_discrete_features(
        probX=probX, probE=probE, node_mask=graph.mask, rng_key=rng
    )

    X_t = jax.nn.one_hot(sampled_graph_t.x, num_classes=probX.shape[-1])
    E_t = jax.nn.one_hot(sampled_graph_t.e, num_classes=probE.shape[-1])
    assert (graph.x.shape == X_t.shape) and (graph.e.shape == E_t.shape)

    z_t = GraphDistribution(x=X_t, e=E_t, y=graph.y, mask=graph.mask).type_as(X_t)

    vals = (E_t.reshape(E_t.shape[0], -1) == 0).all(axis=-1)
    if vals.any():
        ipdb.set_trace()

    return NoisyData(
        t_int=t_int,  # t_int is the timestep at the current timestep
        t=t_float,
        beta_t=beta_t,
        alpha_s_bar=alpha_s_bar,  # alpha_s_bar is the alpha value at the previous timestep
        alpha_t_bar=alpha_t_bar,  # alpha_t_bar is the alpha value at the current timestep
        embedded_graph=z_t,
    )


@typed
def kl_prior(
    *,
    target: GraphDistribution,
    T: SInt,
    noise_schedule: NoiseSchedule,
    transition_model: TransitionModel,
    prior_dist: Distribution,
) -> SFloat:
    """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

    This is essentially a lot of work for something that is in practice negligible in the loss. However, you
    compute it so that you see it when you've made a mistake in your noise schedule.
    """
    # Compute the last alpha value, alpha_T.
    ones = np.ones((target.x.shape[0], 1))
    Ts = T * ones
    alpha_t_bar = noise_schedule.alphas_bar[Ts]  # (bs, 1)

    Qtb = transition_model.get_Qt_bar(alpha_t_bar)

    # Compute transition probabilities
    probX = target.x @ Qtb.x  # (bs, n, dx_out)
    probE = target.e @ np.expand_dims(Qtb.e, 1)  # (bs, n, n, de_out)
    assert probX.shape == target.x.shape

    bs, n, _ = probX.shape

    limit_X = np.broadcast_to(np.expand_dims(prior_dist.x, (0, 1)), (bs, n, -1))
    limit_E = np.broadcast_to(np.expand_dims(prior_dist.e, (0, 1, 2)), (bs, n, n, -1))

    # Make sure that masked rows do not contribute to the loss
    limit_dist_X, limit_dist_E, probX, probE = mask_distributions(
        true_X=np.copy(limit_X),
        true_E=np.copy(limit_E),
        pred_X=probX,
        pred_E=probE,
        node_mask=target.mask,
    )

    kl_distance_X = kl_div(np.log(probX), limit_dist_X).sum(1)
    kl_distance_E = kl_div(np.log(probE), limit_dist_E).sum(1)

    return kl_distance_X + kl_distance_E
