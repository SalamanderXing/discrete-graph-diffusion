from jax.random import PRNGKeyArray
import jax
import optax
import ipdb
from rich import print
from jax import Array
import jax
from jax import numpy as np
from dataclasses import dataclass

from .config import Dimensions
from .transition_model import TransitionModel
from .noise_schedule import PredefinedNoiseScheduleDiscrete
from .sample import sample_discrete_features
from .utils.placeholder import PlaceHolder
from .metrics.sum_except_batch_kl import SumExceptBatchKL


@dataclass(frozen=True)
class NoisyData:
    t_int: Array
    t: Array
    beta_t: Array
    alpha_s_bar: Array
    alpha_t_bar: Array
    X_t: Array
    E_t: Array
    y_t: Array
    node_mask: Array


def compute_Lt(
    X: Array,
    E: Array,
    y: Array,
    pred,
    noisy_data: NoisyData,
    node_mask: Array,
    test: bool,
    T: float,
    transition_model: TransitionModel,
    test_X_kl: SumExceptBatchKL,
    val_X_kl: SumExceptBatchKL,
    test_E_kl: SumExceptBatchKL,
    val_E_kl: SumExceptBatchKL,
) -> float:
    pred_probs_X = jax.nn.softmax(pred.X, axis=-1)
    pred_probs_E = jax.nn.softmax(pred.E, axis=-1)
    pred_probs_y = jax.nn.softmax(pred.y, axis=-1)

    Qtb = transition_model.get_Qt_bar(noisy_data.alpha_t_bar)
    Qsb = transition_model.get_Qt_bar(noisy_data.alpha_s_bar)
    Qt = transition_model.get_Qt(noisy_data.beta_t)

    # Compute distributions to compare with KL
    bs, n, d = X.shape
    prob_true = posterior_distributions(
        X=X,
        E=E,
        y=y,
        X_t=noisy_data.X_t,
        E_t=noisy_data.E_t,
        y_t=noisy_data.y_t,
        Qt=Qt,
        Qsb=Qsb,
        Qtb=Qtb,
    )
    prob_true.e = prob_true.e.reshape((bs, n, n, -1))
    prob_pred = posterior_distributions(
        X=pred_probs_X,
        E=pred_probs_E,
        y=pred_probs_y,
        X_t=noisy_data.X_t,
        E_t=noisy_data.E_t,
        y_t=noisy_data.y_t,
        Qt=Qt,
        Qsb=Qsb,
        Qtb=Qtb,
    )
    prob_pred.e = prob_pred.e.reshape((bs, n, n, -1))

    # Reshape and filter masked rows
    (
        prob_true_X,
        prob_true_E,
        prob_pred.x,
        prob_pred.e,
    ) = mask_distributions(
        true_X=prob_true.x,
        true_E=prob_true.e,
        pred_X=prob_pred.x,
        pred_E=prob_pred.e,
        node_mask=node_mask,
    )
    kl_x = (test_X_kl if test else val_X_kl)(prob_true.x, np.log(prob_pred.x))
    kl_e = (test_E_kl if test else val_E_kl)(prob_true.e, np.log(prob_pred.e))
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


def posterior_distributions(X, E, y, X_t, E_t, y_t, Qt, Qsb, Qtb):
    prob_X = compute_posterior_distribution(
        M=X, M_t=X_t, Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X
    )  # (bs, n, dx)
    prob_E = compute_posterior_distribution(
        M=E, M_t=E_t, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E
    )  # (bs, n * n, de)

    return PlaceHolder(x=prob_X, e=prob_E, y=y_t)


def sum_except_batch(x: Array):
    return x.reshape(x.shape[0], -1).sum(axis=-1)


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
    rng: PRNGKeyArray,
    X: Array,
    E: Array,
    y: Array,
    node_mask: Array,
    T: int,
    noise_schedule: PredefinedNoiseScheduleDiscrete,
    transition_model: TransitionModel,
    training: bool,
    output_dims: Dimensions,
) -> NoisyData:
    """Sample noise and apply it to the data."""

    # Sample a timestep t.
    # When evaluating, the loss for t=0 is computed separately
    lowest_t = 0 if training else 1
    # the above in jax is:
    t_int = jax.random.randint(
        rng, (X.shape[0], 1), lowest_t, T + 1
    )  # why as type float?
    s_int = t_int - 1

    t_float = t_int / T
    s_float = s_int / T

    # beta_t and alpha_s_bar are used for denoising/loss computation
    beta_t = noise_schedule(t_normalized=t_float)  # (bs, 1)
    alpha_s_bar = noise_schedule.get_alpha_bar(t_normalized=s_float)  # (bs, 1)
    alpha_t_bar = noise_schedule.get_alpha_bar(t_normalized=t_float)  # (bs, 1)

    Qtb = transition_model.get_Qt_bar(
        alpha_t_bar
    )  # (bs, dx_in, dx_out), (bs, de_in, de_out)
    assert (abs(Qtb.X.sum(dim=2) - 1.0) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
    assert (abs(Qtb.E.sum(dim=2) - 1.0) < 1e-4).all()

    # Compute transition probabilities
    probX = X @ Qtb.X  # (bs, n, dx_out)
    probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

    sampled_t = sample_discrete_features(
        probX=probX, probE=probE, node_mask=node_mask, rng_key=rng
    )

    X_t = jax.nn.one_hot(sampled_t.X, num_classes=output_dims.X)
    E_t = jax.nn.one_hot(sampled_t.E, num_classes=output_dims.E)
    assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

    z_t = PlaceHolder(x=X_t, e=E_t, y=y).type_as(X_t).mask(node_mask)

    return NoisyData(
        **{
            "t_int": t_int,
            "t": t_float,
            "beta_t": beta_t,
            "alpha_s_bar": alpha_s_bar,
            "alpha_t_bar": alpha_t_bar,
            "X_t": z_t.x,
            "E_t": z_t.e,
            "y_t": z_t.y,
            "node_mask": node_mask,
        }
    )


def kl_prior(
    X: Array,
    E: Array,
    node_mask: Array,
    T: float,
    noise_schedule,
    transition_model,
    limit_dist,
    device,
) -> Array:
    """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

    This is essentially a lot of work for something that is in practice negligible in the loss. However, you
    compute it so that you see it when you've made a mistake in your noise schedule.
    """
    # Compute the last alpha value, alpha_T.
    ones = np.ones((X.shape[0], 1))
    Ts = T * ones
    alpha_t_bar = noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

    Qtb = transition_model.get_Qt_bar(alpha_t_bar, device)

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

    kl_distance_X = kl_div(np.log(probX), limit_dist_X)
    kl_distance_E = kl_div(np.log(probE), limit_dist_E)

    return sum_except_batch(kl_distance_X) + sum_except_batch(kl_distance_E)
