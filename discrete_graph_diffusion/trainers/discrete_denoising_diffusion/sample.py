"""
Functions related to sampling.
"""
# TODO: organize this better
from jax import numpy as np
from flax import linen as nn
from jax import Array
from jax.random import PRNGKeyArray
import jax
from jax import random
from jax._src.random import PRNGKey
from jax.scipy.special import logit
from mate.jax import typed, jit, SInt, SFloat, SBool, Key
import ipdb
from .diffusion_types import Q, GraphDistribution
from .diffusion_types import TransitionModel


def sample_discrete_features(
    *, probX: Array, probE: Array, node_mask: Array, rng_key: Key
):
    """Sample features from multinomial distribution with given probabilities (probX, probE, proby)
    :param probX: bs, n, dx_out        node features
    :param probE: bs, n, n, de_out     edge features
    :param rng_key: random.PRNGKey     random key for JAX operations
    """
    bs, n, _ = probX.shape
    # Noise X
    # The masked rows should define probability distributions as well
    # probX = probX.at[~node_mask].set(1 / probX.shape[-1])  # , probX)
    probX = np.where(
        (~node_mask)[:, :, None],
        1 / probX.shape[-1],
        probX,
    )

    # Flatten the probability tensor to sample with categorical distribution
    probX = probX.reshape(bs * n, -1)  # (bs * n, dx_out)

    # Sample X
    rng_key, subkey = random.split(rng_key)
    X_t = random.categorical(subkey, logit(probX), axis=-1)  # (bs * n,)
    X_t = X_t.reshape(bs, n)  # (bs, n)

    # Noise E
    # The masked rows should define probability distributions as well
    inverse_edge_mask = ~(node_mask[:, None] * node_mask[:, :, None])
    diag_mask = np.eye(n)[None].astype(bool).repeat(bs, axis=0)
    probE = np.where(inverse_edge_mask[..., None], 1 / probE.shape[-1], probE)
    probE = np.where(diag_mask[..., None], 1 / probE.shape[-1], probE)

    probE = probE.reshape(bs * n * n, -1)  # (bs * n * n, de_out)

    # Sample E
    rng_key, subkey = random.split(rng_key)
    E_t = random.categorical(subkey, logit(probE), axis=-1)  # (bs * n * n,)
    E_t = E_t.reshape(bs, n, n)  # (bs, n, n)
    E_t = np.triu(E_t, k=1)
    E_t = E_t + np.transpose(E_t, (0, 2, 1))
    # hedsfs
    return Q(x=X_t, e=E_t, y=np.zeros((bs, 0), dtype=X_t.dtype))


def compute_batched_over0_posterior_distribution(X_t, Qt, Qsb, Qtb):
    """M: X or E
    Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
    X_t: bs, n, dt          or bs, n, n, dt
    Qt: bs, d_t-1, dt
    Qsb: bs, d0, d_t-1
    Qtb: bs, d0, dt.
    """
    # Flatten feature tensors
    # Careful with this line. It does nothing if X is a node feature. If X is an edge features it maps to
    # bs x (n ** 2) x d
    X_t = np.reshape(X_t, (X_t.shape[0], -1, X_t.shape[-1])).astype(
        np.float32
    )  # bs x N x dt

    Qt_T = np.swapaxes(Qt, -1, -2)  # bs, dt, d_t-1
    left_term = X_t @ Qt_T  # bs, N, d_t-1
    left_term = np.expand_dims(left_term, axis=2)  # bs, N, 1, d_t-1

    right_term = np.expand_dims(Qsb, axis=1)  # bs, 1, d0, d_t-1
    numerator = left_term * right_term  # bs, N, d0, d_t-1

    X_t_transposed = np.swapaxes(X_t, -1, -2)  # bs, dt, N

    prod = Qtb @ X_t_transposed  # bs, d0, N
    prod = np.swapaxes(prod, -1, -2)  # bs, N, d0
    denominator = np.expand_dims(prod, axis=-1)  # bs, N, d0, 1
    denominator = np.where(denominator == 0, 1e-6, denominator)

    out = numerator / denominator
    return out


def sample_discrete_feature_noise(*, limit_dist, node_mask):
    """Sample from the limit distribution of the diffusion process"""
    bs, n_max = node_mask.shape
    x_limit = np.broadcast_to(limit_dist.X[None, None, :], (bs, n_max, -1))
    e_limit = np.broadcast_to(limit_dist.E[None, None, None, :], (bs, n_max, n_max, -1))
    U_X = x_limit.reshape(bs * n_max, -1).argmax(axis=-1).reshape(bs, n_max)
    U_E = (
        e_limit.reshape(bs * n_max * n_max, -1)
        .argmax(axis=-1)
        .reshape(bs, n_max, n_max)
    )
    U_y = np.empty((bs, 0))

    long_mask = node_mask.astype(np.int32)
    U_X = U_X.astype(long_mask.dtype)
    U_E = U_E.astype(long_mask.dtype)
    U_y = U_y.astype(long_mask.dtype)

    U_X = jax.nn.one_hot(U_X, num_classes=x_limit.shape[-1]).astype(np.float32)
    U_E = jax.nn.one_hot(U_E, num_classes=e_limit.shape[-1]).astype(np.float32)

    # Get upper triangular part of edge noise, without main diagonal
    upper_triangular_mask = np.zeros_like(U_E)
    indices = np.triu_indices(n=U_E.shape[1], m=U_E.shape[2], k=1)
    upper_triangular_mask = upper_triangular_mask.at[:, indices[0], indices[1], :].set(
        1
    )

    U_E = U_E * upper_triangular_mask
    U_E = U_E + np.transpose(U_E, (0, 2, 1, 3))

    assert np.all(U_E == np.transpose(U_E, (0, 2, 1, 3)))

    return GraphDistribution(x=U_X, e=U_E, y=U_y, mask=node_mask)


def sample_batch(
    *,
    batch_size: int,
    keep_chain: int,
    number_chain_steps: int,
    model: nn.Module,
    T: int,
    node_dist,  #: NodesDistribution,
    num_nodes=None,
    limit_dist: GraphDistribution | None = None,
    noise_transition: TransitionModel,
    noise_schedule,  #: NoiseSchedule,
    extra_features: Array,
    domain_features: Array,
    batch_id: int = -1,  # used by the visualization tools
    save_final: int = -1,  # used by the visualization tools
    visualization_tools=None,
    rng_key: Array,
    Xdim_output: int,
    Edim_output: int,
):
    """
    :param batch_id: int
    :param batch_size: int
    :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
    :param save_final: int: number of predictions to save to file
    :param keep_chain: int: number of chains to save to file
    :param keep_chain_steps: number of timesteps to save for each chain
    :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
    """
    if num_nodes is None:
        n_nodes = node_dist.sample_n(batch_size)
    elif type(num_nodes) == int:
        n_nodes = num_nodes * np.ones(batch_size, dtype=np.int32)
    else:
        assert isinstance(num_nodes, Array)
        n_nodes = num_nodes
    n_max = np.max(n_nodes).item()
    # Build the masks
    arange = np.arange(n_max).reshape(1, -1).repeat(batch_size, axis=0)
    node_mask = arange < n_nodes.reshape(-1, 1)

    z_T = sample_discrete_feature_noise(limit_dist=limit_dist, node_mask=node_mask)
    X, E, y = z_T.x, z_T.e, z_T.y

    assert np.all(E == np.transpose(E, (0, 2, 1, 3)))
    assert number_chain_steps < T
    chain_X_size = (number_chain_steps, keep_chain, X.shape[1])
    chain_E_size = (number_chain_steps, keep_chain, E.shape[1], E.shape[2])

    chain_X = np.zeros(chain_X_size)
    chain_E = np.zeros(chain_E_size)

    assert T > 0
    # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
    for s_int in reversed(range(0, T)):
        s_array = s_int * np.ones((batch_size, 1)).astype(y.dtype)
        t_array = s_array + 1
        s_norm = s_array / T
        t_norm = t_array / T

        # Sample z_s
        sampled_s, discrete_sampled_s = sample_p_zs_given_zt(
            extra_features=extra_features,
            domain_features=domain_features,
            model=model,
            noise_schedule=noise_schedule,
            s=s_norm,
            t=t_norm,
            X_t=X,
            E_t=E,
            y_t=y,
            node_mask=node_mask,
            noise_transition=noise_transition,
            rng_key=rng_key,
            Xdim_output=Xdim_output,
            Edim_output=Edim_output,
        )
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Save the first keep_chain graphs
        write_index = (s_int * number_chain_steps) // T
        chain_X = chain_X.at[write_index].set(discrete_sampled_s.X[:keep_chain])
        chain_E = chain_E.at[write_index].set(discrete_sampled_s.E[:keep_chain])

    # Sample
    sampled_s = sampled_s.mask(node_mask, collapse=True)
    X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

    # Prepare the chain for saving
    if keep_chain > 0:
        final_X_chain = X[:keep_chain]
        final_E_chain = E[:keep_chain]

        chain_X = chain_X.at[0].set(
            final_X_chain
        )  # Overwrite last frame with the resulting X, E
        chain_E = chain_E.at[0].set(final_E_chain)

        chain_X = np.flip(chain_X)
        chain_E = np.flip(chain_E)

        # Repeat last frame to see final sample better
        chain_X = np.concatenate([chain_X, np.repeat(chain_X[-1:], 10, axis=0)], axis=0)
        chain_E = np.concatenate([chain_E, np.repeat(chain_E[-1:], 10, axis=0)], axis=0)

        assert chain_X.shape[0] == (number_chain_steps + 10)

    molecule_list = []
    for i in range(batch_size):
        n = n_nodes[i]
        atom_types = X[i, :n].copy()
        edge_types = E[i, :n, :n].copy()
        molecule_list.append([atom_types, edge_types])

    # TODO: add visualization if
    return molecule_list


def sample_p_zs_given_zt(
    *,
    s,
    t,
    X_t: Array,
    E_t: Array,
    y_t: Array,
    node_mask: Array,
    noise_schedule,  # : NoiseSchedule,
    noise_transition: TransitionModel,
    extra_features: Array,
    domain_features: Array,
    Xdim_output: int,
    Edim_output: int,
    model: nn.Module,
    rng_key: Array,
):
    """Samples from zs ~ p(zs | zt). Only used during sampling.
    if last_step, return the graph prediction as well"""
    bs, n, _ = X_t.shape
    beta_t = noise_schedule.betas[t]  # (bs, 1)
    alpha_s_bar = noise_schedule.get_alpha_bar(
        t_normalized=s
    )  # TODO: covert to updated TransitionModel
    alpha_t_bar = noise_schedule.get_alpha_bar(t_normalized=t)

    # Retrieve transitions matrix
    Qtb = noise_transition.get_Qt_bar(alpha_t_bar)
    Qsb = noise_transition.get_Qt_bar(alpha_s_bar)
    Qt = noise_transition.get_Qt(beta_t)

    # Neural net predictions
    noisy_data = {
        "X_t": X_t,
        "E_t": E_t,
        "y_t": y_t,
        "t": t,
        "node_mask": node_mask,
    }
    extra_data = compute_extra_data(
        noisy_data=noisy_data,
        extra_features=extra_features,
        domain_features=domain_features,
    )
    pred = model(noisy_data, extra_data, node_mask)

    # Normalize predictions
    pred_X = jax.nn.softmax(pred.X, axis=-1)  # bs, n, d0
    pred_E = jax.nn.softmax(pred.E, axis=-1)  # bs, n, n, d0

    p_s_and_t_given_0_X = compute_batched_over0_posterior_distribution(
        X_t=X_t, Qt=Qt.x, Qsb=Qsb.x, Qtb=Qtb.x
    )

    p_s_and_t_given_0_E = compute_batched_over0_posterior_distribution(
        X_t=E_t, Qt=Qt.e, Qsb=Qsb.e, Qtb=Qtb.e
    )
    # Dim of these two tensors: bs, N, d0, d_t-1
    weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X  # bs, n, d0, d_t-1
    unnormalized_prob_X = weighted_X.sum(dim=2)  # bs, n, d_t-1
    unnormalized_prob_X[np.sum(unnormalized_prob_X, axis=-1) == 0] = 1e-5
    prob_X = unnormalized_prob_X / np.sum(
        unnormalized_prob_X, axis=-1, keepdims=True
    )  # bs, n, d_t-1

    pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
    weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E  # bs, N, d0, d_t-1
    unnormalized_prob_E = weighted_E.sum(dim=-2)
    unnormalized_prob_E[np.sum(unnormalized_prob_E, axis=-1) == 0] = 1e-5
    prob_E = unnormalized_prob_E / np.sum(unnormalized_prob_E, axis=-1, keepdims=True)
    prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

    assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
    assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

    sampled_s = sample_discrete_features(
        prob_X, prob_E, node_mask=node_mask, rng_key=rng_key
    )

    X_s = jax.nn.one_hot(sampled_s.x, num_classes=Xdim_output).float()
    E_s = jax.nn.one_hot(sampled_s.e, num_classes=Edim_output).float()

    assert (E_s == np.swapaxes(E_s, 1, 2)).all()
    assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

    out_one_hot = GraphDistribution(x=X_s, e=E_s, y=np.zeros(y_t.shape[0], 0))
    out_discrete = GraphDistribution(x=X_s, e=E_s, y=np.zeros(y_t.shape[0], 0))

    return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(
        node_mask, collapse=True
    ).type_as(y_t)



