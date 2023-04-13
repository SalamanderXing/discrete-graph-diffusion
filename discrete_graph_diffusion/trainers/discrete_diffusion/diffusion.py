from typing import Callable
from jax.random import PRNGKey, PRNGKeyArray
import jax
import optax
import ipdb
from rich import print
from jax import Array
import jax
from jax import numpy as np
from flax import linen as nn
from flax.training.train_state import TrainState
from dataclasses import dataclass

from .transition_model import TransitionModel
from .noise_schedule import PredefinedNoiseScheduleDiscrete
from .sample import sample_discrete_features
from .utils import Graph, NoisyData, softmax_kl_div


def reconstruction_logp(
    *,
    rng_key: PRNGKeyArray,
    model: nn.Module,
    state: TrainState,
    noise_schedule: PredefinedNoiseScheduleDiscrete,
    t: int,
    X: Array,
    E: Array,
    node_mask: Array,
    transition_model: TransitionModel,
    get_extra_features: Callable[[NoisyData], Graph],
    get_domain_features: Callable[[NoisyData], Graph],
):
    # Compute noise values for t = 0.
    t_zeros = np.zeros_like(t)
    beta_0 = noise_schedule(t_zeros)
    Q0 = transition_model.get_Qt(beta_t=beta_0)

    probX0 = X @ Q0.X  # (bs, n, dx_out)
    probE0 = np.matmul(E, np.expand_dims(Q0.E, 1))  # (bs, n, n, de_out)

    sampled0 = sample_discrete_features(
        rng_key=rng_key, probX=probX0, probE=probE0, node_mask=node_mask
    )

    X0 = np.eye(X.shape[-1])[sampled0.X].astype(np.float32)
    E0 = np.eye(E.shape[-1])[sampled0.E].astype(np.float32)
    y0 = sampled0.y
    assert (X.shape == X0.shape) and (E.shape == E0.shape)

    sampled_0 = Graph(x=X0, e=E0, y=y0, mask=node_mask)

    noisy_data = NoisyData(
        graph=sampled_0,
        t=np.zeros((X0.shape[0], 1), dtype=y0.dtype),
    )
    extra_data = compute_extra_data(noisy_data, get_extra_features, get_domain_features)
    # pred0 = model.apply(noisy_data, extra_data, node_mask)
    # passes the parameters to the model and calls the apply
    x, e, y = model.apply(state.params, noisy_data, extra_data, node_mask)
    pred0 = Graph(x=x, e=e, y=y, mask=node_mask)

    # Normalize predictions
    probX0 = jax.nn.softmax(pred0.X, axis=-1)
    probE0 = jax.nn.softmax(pred0.E, axis=-1)
    proby0 = jax.nn.softmax(pred0.y, axis=-1)

    # Set masked rows to arbitrary values that don't contribute to loss
    probX0 = probX0.at[np.logical_not(node_mask)].set(np.ones(X.shape[-1]))
    probE0 = probE0.at[
        np.logical_not(np.expand_dims(node_mask, 1) * np.expand_dims(node_mask, 2))
    ].set(np.ones(E.shape[-1]))

    diag_mask = np.eye(probE0.shape[1], dtype=bool)
    diag_mask = np.expand_dims(diag_mask, 0).repeat(probE0.shape[0], axis=0)
    probE0 = probE0.at[diag_mask].set(np.ones(E.shape[-1]))

    return Graph(x=probX0, e=probE0, y=proby0)


from typing import Callable


def compute_extra_data(
    noisy_data: NoisyData,
    get_extra_features: Callable[[NoisyData], Graph],
    get_domain_features: Callable[[NoisyData], Graph],
):
    """At every training step (after adding noise) and step in sampling, compute extra information and append to
    the network input."""

    extra_features = get_extra_features(noisy_data)
    extra_molecular_features = get_domain_features(noisy_data)

    extra_X = np.concatenate((extra_features.x, extra_molecular_features.x), axis=-1)
    extra_E = np.concatenate((extra_features.e, extra_molecular_features.e), axis=-1)
    extra_y = np.concatenate((extra_features.y, extra_molecular_features.y), axis=-1)

    t = noisy_data.t
    extra_y = np.concatenate((extra_y, t), axis=1)

    return Graph(x=extra_X, e=extra_E, y=extra_y)


def compute_Lt(
    *,
    target: Graph,
    pred: Graph,
    noisy_data: NoisyData,
    T: float,
    transition_model: TransitionModel,
) -> Array:
    pred_probs = Graph(
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
        noisy_data.graph,
        Qt=Qt,
        Qsb=Qs_bar,
        Qtb=Qt_bar,
    )
    prob_pred_unmasked = posterior_distributions(
        pred_probs,
        noisy_data.graph,
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


def compute_posterior_distribution(
    M: Array, M_t: Array, Qt_M: Array, Qsb_M: Array, Qtb_M: Array
) -> Array:
    """M: X or E
    Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T
    """
    # Flatten feature tensors
    M = np.reshape(M, (M.shape[0], -1, M.shape[-1])).astype(
        np.float32
    )  # (bs, N, d) with N = n or n * n
    M_t = np.reshape(M_t, (M_t.shape[0], -1, M_t.shape[-1])).astype(np.float32)  # same

    Qt_M_T = np.transpose(Qt_M, (0, 2, 1))  # (bs, d, d)

    left_term = M_t @ Qt_M_T  # (bs, N, d)
    right_term = M @ Qsb_M  # (bs, N, d)
    product = left_term * right_term  # (bs, N, d)

    denom = M @ Qtb_M  # (bs, N, d) @ (bs, d, d) = (bs, N, d)
    denom = (denom * M_t).sum(axis=-1)  # (bs, N, d) * (bs, N, d) + sum = (bs, N)

    prob = product / denom[..., None]  # (bs, N, d)

    return prob


def posterior_distributions(graph: Graph, graph_t: Graph, Qt, Qsb, Qtb) -> Graph:
    return Graph(
        x=compute_posterior_distribution(
            M=graph.x, M_t=graph_t.x, Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X
        ),  # (bs, n, dx),
        e=compute_posterior_distribution(
            M=graph.e, M_t=graph_t.e, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E
        ).reshape((graph.e.shape[0], graph.e.shape[1], graph.e.shape[1], -1)),
        y=graph_t.y,
    )


def kl_div(p: Array, q: Array, eps: float = 2**-17) -> Array:
    """Calculates the Kullback-Leibler divergence between arrays p and q."""
    return p.dot(np.log(p + eps) - np.log(q + eps))


def mask_distributions(
    true_X: Array, true_E: Array, pred_X: Array, pred_E: Array, node_mask: Array
) -> tuple[Array, Array, Array, Array]:
    # Add a small value everywhere to avoid nans
    pred_X = pred_X.at[:, :, :].add(1e-7)
    pred_X = pred_X / np.sum(pred_X, axis=-1, keepdims=True)

    pred_E = pred_E.at[:, :, :, :].add(1e-7)
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


def apply_noise(
    *,
    rng: PRNGKeyArray,
    x: Array,
    e: Array,
    y: Array,
    node_mask: Array,
    T: int,
    noise_schedule: PredefinedNoiseScheduleDiscrete,
    transition_model: TransitionModel,
    training: bool,
) -> NoisyData:
    """Sample noise and apply it to the data."""

    # Sample a timestep t.
    # When evaluating, the loss for t=0 is computed separately
    lowest_t = 0 if training else 1
    t_int = jax.random.randint(  # sample t_int from U[lowest_t, T]
        rng, (x.shape[0], 1), lowest_t, T + 1
    )  # why as type float?
    s_int = t_int - 1

    t_float = t_int / T  # current normalized timestep
    s_float = s_int / T  # previous normalized timestep

    # beta_t and alpha_s_bar are used for denoising/loss computation
    beta_t = noise_schedule(
        t_normalized=t_float
    )  # (bs, 1) beta_t is the noise level at t
    alpha_s_bar = noise_schedule.get_alpha_bar(t_normalized=s_float)  # (bs, 1)
    alpha_t_bar = noise_schedule.get_alpha_bar(t_normalized=t_float)  # (bs, 1)

    Qt_bar = transition_model.get_Qt_bar(
        alpha_t_bar
    )  # (bs, dx_in, dx_out), (bs, de_in, de_out)
    assert (abs(Qt_bar.x.sum(axis=2) - 1.0) < 1e-4).all(), Qt_bar.x.sum(axis=2) - 1
    assert (abs(Qt_bar.e.sum(axis=2) - 1.0) < 1e-4).all()
    # Compute transition probabilities
    probX = x @ Qt_bar.x  # (bs, n, dx_out)
    probE = e @ Qt_bar.e[:, None]  # (bs, n, n, de_out)

    sampled_graph_t = sample_discrete_features(
        probX=probX, probE=probE, node_mask=node_mask, rng_key=rng
    )

    X_t = jax.nn.one_hot(sampled_graph_t.x, num_classes=probX.shape[-1])
    E_t = jax.nn.one_hot(sampled_graph_t.e, num_classes=probE.shape[-1])
    assert (x.shape == X_t.shape) and (e.shape == E_t.shape)

    z_t = Graph(x=X_t, e=E_t, y=y, mask=node_mask).type_as(X_t)

    return NoisyData(
        t_int=t_int,  # t_int is the timestep at the current timestep
        t=t_float,
        beta_t=beta_t,
        alpha_s_bar=alpha_s_bar,  # alpha_s_bar is the alpha value at the previous timestep
        alpha_t_bar=alpha_t_bar,  # alpha_t_bar is the alpha value at the current timestep
        graph=z_t,
    )


def kl_prior(
    *,
    X: Array,
    E: Array,
    node_mask: Array,
    T: float,
    noise_schedule,
    transition_model,
    limit_dist,
) -> Array:
    """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

    This is essentially a lot of work for something that is in practice negligible in the loss. However, you
    compute it so that you see it when you've made a mistake in your noise schedule.
    """
    # Compute the last alpha value, alpha_T.
    ones = np.ones((X.shape[0], 1))
    Ts = T * ones
    alpha_t_bar = noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

    Qtb = transition_model.get_Qt_bar(alpha_t_bar)

    # Compute transition probabilities
    probX = X @ Qtb.X  # (bs, n, dx_out)
    probE = E @ np.expand_dims(Qtb.E, 1)  # (bs, n, n, de_out)
    assert probX.shape == X.shape

    bs, n, _ = probX.shape

    limit_X = np.broadcast_to(np.expand_dims(limit_dist.X, (0, 1)), (bs, n, -1))
    limit_E = np.broadcast_to(np.expand_dims(limit_dist.E, (0, 1, 2)), (bs, n, n, -1))

    # Make sure that masked rows do not contribute to the loss
    limit_dist_X, limit_dist_E, probX, probE = mask_distributions(
        true_X=np.copy(limit_X),
        true_E=np.copy(limit_E),
        pred_X=probX,
        pred_E=probE,
        node_mask=node_mask,
    )

    kl_distance_X = kl_div(np.log(probX), limit_dist_X).sum(1)
    kl_distance_E = kl_div(np.log(probE), limit_dist_E).sum(1)

    return kl_distance_X + kl_distance_E
