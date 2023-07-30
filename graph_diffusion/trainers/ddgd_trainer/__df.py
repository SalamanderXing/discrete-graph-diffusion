"""
A lot of functions related to diffusion models. But which do not depend on the on the model itself.
"""
from rich import print
from beartype import beartype
import ipdb
from jaxtyping import jaxtyped
from collections.abc import Callable
import jax
from jax import Array, random
import jax
from jax.debug import print as jprint  # type: ignore
from jax import numpy as np
from flax import linen as nn
import mate as m
from mate.jax import SInt, SFloat, Key, SBool
from jaxtyping import Int, Float, Bool, jaxtyped
import hashlib
from orbax.checkpoint.pytree_checkpoint_handler import Transform
from ...shared.graph import graph_distribution as gd
from .transition_model import TransitionModel
import ipdb
import einops as e

GraphDistribution = gd.GraphDistribution
EdgeDistribution = gd.EdgeDistribution
NodeDistribution = gd.NodeDistribution
Q = gd.Q

GetProbabilityType = Callable[
    [gd.OneHotGraph, Int[Array, "batch_size"]], gd.DenseGraphDistribution
]

# import sys
# sys.path.insert(0, "/home/bluesk/Documents/tmp/digress-copy/")
#
# # from src.diffusion import diffusion_utils
# from src import diffusion_model_discrete
# from src.diffusion import diffusion_utils
#
# digress = diffusion_model_discrete.DiscreteDenoisingDiffusion(
#     cfg=None,
#     dataset_infos=None,
#     train_metrics=None,
#     sampling_metrics=None,
#     visualization_tools=None,
#     extra_features=None,
#     domain_features=None,
#     reload=True,
# )
#


def enc(x: str):
    return int(hashlib.md5(x.encode()).hexdigest()[:8], base=16)


def summarize(name, node_mask, x):
    def _summarize(prefix, x: Array, tab=0):
        tabs = " " * tab
        print(f"{tabs}{prefix} mean: {np.mean(x):.3f} std: {np.std(x):.3f}")

    print(f"NAME: {name}")
    edge_mask = node_mask[:, None]
    xs = node_mask * x.nodes.argmax(-1)
    es = edge_mask * x.edges.argmax(-1)
    _summarize("nodes", xs, 1)
    _summarize("edges", es, 1)


def pseudo_assert(condition: SBool):
    """
    When debugging NaNs, the following
    function will raise an exception (even during jit)
    """
    return np.where(condition, 0, np.nan)


def __compute_posterior_distribution_edges(
    edges: EdgeDistribution,
    edges_t: EdgeDistribution,
    q_t: Float[Array, "bs e e"],
    q_s_bar: Float[Array, "bs e e"],
    q_t_bar: Float[Array, "bs e e"],
):
    q_t_e_transposed = e.repeat(
        q_t, "bs e1 e2 -> bs n1 n2 e2 e1", n1=edges.shape[1], n2=edges.shape[2]
    )
    left_term = e.einsum(
        edges_t, q_t_e_transposed, "bs n1 n2 de, bs n1 n2 de e1 -> bs n1 n2 e1"
    )
    right_term = e.einsum(edges, q_s_bar, "bs n1 n2 de, bs de e1 -> bs n1 n2 e1")
    product = left_term * right_term
    denom = e.einsum(edges, q_t_bar, "bs n1 n2 de, bs de e1 -> bs n1 n2 e1")
    denom = e.reduce(denom * edges_t, "bs n1 n2 e -> bs n1 n2", "sum")
    denom = np.where(denom == 0, 1, denom)
    prob = product / denom[..., None]
    return prob


def __compute_posterior_distribution_nodes(
    nodes: NodeDistribution,
    nodes_t: NodeDistribution,
    q_t: Float[Array, "bs e e"],
    q_s_bar: Float[Array, "bs e e"],
    q_t_bar: Float[Array, "bs e e"],
):
    q_t_T = e.repeat(q_t, "bs e1 e2 -> bs n e2 e1", n=nodes.shape[1])
    left_term = e.einsum(nodes_t, q_t_T, "bs n de, bs n de e1 -> bs n e1")
    right_term = nodes @ q_s_bar
    product = left_term * right_term

    denom = nodes @ q_t_bar
    denom = e.reduce(denom * nodes_t, "bs n e -> bs n", reduction="sum")
    denom = np.where(denom == 0, 1, denom)
    prob = product / denom[..., None]  # type:ignore
    pseudo_assert((prob >= 0).all())
    return prob


def posterior_distribution(
    g: gd.GraphDistribution,
    g_t: gd.OneHotGraph,
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
    pseudo_assert((prob_x >= 0).all())
    prob_e = __compute_posterior_distribution_edges(
        edges=g.edges,
        edges_t=g_t.edges,
        q_t=q_t.edges,
        q_s_bar=q_s_bar.edges,
        q_t_bar=q_t_bar.edges,
    )
    pseudo_assert((prob_e >= 0).all())
    return gd.DenseGraphDistribution.create(
        nodes=prob_x,
        edges=prob_e,
        nodes_mask=g.nodes_mask,
        edges_mask=g.edges_mask,
    )


def predict_from_random_timesteps(
    p: GetProbabilityType,
    g: gd.OneHotGraph,
    transition_model: TransitionModel,
    rng: Key,
):
    rng_t = jax.random.fold_in(rng, enc("t"))
    rng_z = jax.random.fold_in(rng, enc("z"))
    t = jax.random.randint(
        rng_t, (g.nodes.shape[0],), 1, transition_model.diffusion_steps + 1
    )
    q_t_b = transition_model.q_bars[t]
    q_t_bar_given_g = gd.matmul(g, q_t_b)
    g_t = gd.sample_one_hot(q_t_bar_given_g, rng_z)
    g_pred = p(g_t, t)
    return t, g_t, g_pred


def _compute_lt(
    t: Int[Array, "b"],
    g: gd.OneHotGraph,
    g_t: gd.OneHotGraph,
    raw_g_pred: gd.DenseGraphDistribution,
    transition_model: TransitionModel,
):
    raw_g_pred = gd.softmax(raw_g_pred)
    posterior_prob_true = posterior_distribution(
        g=g,
        g_t=g_t,
        transition_model=transition_model,
        t=t,
    )
    posterior_prob_pred = posterior_distribution(
        g=raw_g_pred,
        g_t=g_t,
        transition_model=transition_model,
        t=t,
    )
    kl_div = gd.kl_div(posterior_prob_true, posterior_prob_pred)
    result = transition_model.diffusion_steps * kl_div
    return result


def compute_reconstruction_logp(
    *,
    rng_key: Key,
    p: GetProbabilityType,
    g: gd.OneHotGraph,
    transition_model: TransitionModel,
):
    rng_z = jax.random.fold_in(rng_key, enc("z"))
    t_0 = np.zeros(g.nodes.shape[0], dtype=int)
    q_0 = transition_model.qs[t_0]
    q_0_given_g = gd.matmul(g, q_0)
    z_1 = gd.sample_one_hot(q_0_given_g, rng_z)
    g_pred = gd.softmax(p(z_1, t_0))
    return _compute_reconstruction_logp(
        g=g,
        g_pred=g_pred,
    )


def _compute_reconstruction_logp(
    g: gd.OneHotGraph,
    g_pred: gd.DenseGraphDistribution,
) -> Float[Array, "batch_size"]:
    return gd.logprobs_at(g, g_pred)  # gd.logprobs_at(g_pred, g)


def compute_kl_prior(
    *,
    target: gd.OneHotGraph,
    transition_model: TransitionModel,
) -> Float[Array, "batch_size"]:
    """Computes the KL between q(z1 | x) and the prior p(z1) (extracted from the data)

    Zero in our case
    """
    # Compute the last alpha value, alpha_T.
    qt_bar_T = transition_model.q_bars[np.array([-1])]

    # Compute transition probabilities
    transition_probs = gd.matmul(target, qt_bar_T)

    limit_dist = transition_model.limit_dist.repeat(target.nodes.shape[0])
    return gd.kl_div(transition_probs, limit_dist)


def compute_train_loss(
    *,
    target: gd.OneHotGraph,
    transition_model: TransitionModel,
    get_probability: GetProbabilityType,
    rng_key: Key,
):  # -> Float[Array, "batch_size"]:
    rng_sample = jax.random.fold_in(rng_key, enc("sample"))
    rng_reclogp = jax.random.fold_in(rng_key, enc("train_reclogp"))
    t, g_t, g_pred = predict_from_random_timesteps(
        get_probability, target, transition_model, rng_sample
    )
    loss_type = "digress"
    if "ce" in loss_type:
        ce_term = gd.softmax_cross_entropy(g_pred, target).mean()
    if "elbo" in loss_type or loss_type == "ce_minimal":
        loss_all_t = _compute_lt(
            t=t, g=target, g_t=g_t, raw_g_pred=g_pred, transition_model=transition_model
        ).mean()
        if loss_type == "ce_minimal":
            tot_loss = loss_all_t + ce_term
        if "elbo" in loss_type:
            reconstruction_logp = compute_reconstruction_logp(
                rng_key=rng_reclogp,
                p=get_probability,
                transition_model=transition_model,
                g=target,
            ).mean()
            if loss_type == "elbo_ce":
                tot_loss = loss_all_t - reconstruction_logp + ce_term
            elif loss_type == "elbo":
                tot_loss = loss_all_t - reconstruction_logp
    elif loss_type == "ce":
        tot_loss = ce_term
    elif loss_type == "digress":
        tot_loss = gd.softmax_cross_entropy(g_pred, target, np.array([1.0, 5.0]))
    return tot_loss


def compute_val_loss(
    *,
    target: gd.OneHotGraph,
    transition_model: TransitionModel,
    p: GetProbabilityType,
    nodes_dist: Float[Array, "node_types"],
    rng_key: Key,
) -> dict[str, Float[Array, "batch_size"]]:
    rng_lt = jax.random.fold_in(rng_key, enc("val_lt"))
    rng_reclogp = jax.random.fold_in(rng_key, enc("val_reclogp"))

    # 1.  log_prob of the target graph under the nodes distribution (based on # of nodes)
    log_pn = np.log(nodes_dist[target.nodes_mask.sum(-1)])  # / np.log(base)

    # # 2. The KL between q(z_T | x) and p(z_T) = (simply an Empirical prior). Should be close to zer
    kl_prior = 0.0
    t, g_t, g_pred = predict_from_random_timesteps(p, target, transition_model, rng_lt)
    loss_all_t = _compute_lt(
        t=t, g=target, g_t=g_t, raw_g_pred=g_pred, transition_model=transition_model
    )
    # 4. Reconstruction loss
    # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
    reconstruction_logp = compute_reconstruction_logp(
        rng_key=rng_reclogp,
        g=target,
        transition_model=transition_model,
        p=p,
    ).mean()
    tot_loss = -log_pn + kl_prior + loss_all_t - reconstruction_logp
    return {
        "log_pn": log_pn,
        "kl_prior": kl_prior,
        "diffusion_loss": loss_all_t,
        "rec_logp": reconstruction_logp,
        "nll": tot_loss,
    }


def compute_posterior_for_all_features(
    g_t: gd.OneHotGraph, transition_model: TransitionModel, t
):
    """
    Computes the posterior distribution for every feature of the graph
    """
    q_t = transition_model.qs[t]
    q_s_b = transition_model.q_bars[t - 1]
    q_t_b = transition_model.q_bars[t]
    nodes = __compute_batched_over0_posterior_distribution_nodes(
        g_t.nodes, q_t.nodes, q_s_b.nodes, q_t_b.nodes
    )
    edges = __compute_batched_over0_posterior_distribution_edges(
        g_t.edges, q_t.edges, q_s_b.edges, q_t_b.edges
    )
    return (nodes, edges)


def __compute_batched_over0_posterior_distribution_nodes(nodes_t, q_t, q_s_b, q_t_b):
    """M: X or E
    Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
    X_t: bs, n, dt          or bs, n, n, dt
    Qt: bs, d_t-1, dt
    Qsb: bs, d0, d_t-1
    Qtb: bs, d0, dt.
    (x_t @ q_t.T * qsb)/(qtb @ x.T)
    """
    q_t_T = e.rearrange(q_t, "bs ne1 ne2 -> bs ne2 ne1")  # bs, dt, d_t-1
    left_term = nodes_t @ q_t_T  # bs, N, d_t-1
    left_term = e.rearrange(left_term, "bs n ne -> bs n 1 ne")
    right_term = e.rearrange(q_s_b, "bs ne1 ne2 -> bs 1 ne1 ne2")
    numerator = left_term * right_term

    nodes_t_T = e.rearrange(nodes_t, "bs n ne -> bs ne n")
    denominator = e.rearrange(q_t_b @ nodes_t_T, "bs ne n -> bs n ne 1")
    denominator = np.where(denominator != 0, denominator, 1e-6)

    out = numerator / denominator
    return out


def __compute_batched_over0_posterior_distribution_edges(edges_t, q_t, q_s_b, q_t_b):
    """M: X or E
    Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
    X_t: bs, n, dt          or bs, n, n, dt
    Qt: bs, d_t-1, dt
    Qsb: bs, d0, d_t-1
    Qtb: bs, d0, dt.
    """
    n = edges_t.shape[1]
    edges_t_reformat = e.rearrange(edges_t, "b n1 n2 ee -> b (n1 n2) ee")
    q_t_T = e.rearrange(q_t, "b n1 n2 -> b n2 n1")
    left_term = edges_t_reformat @ q_t_T
    left_term = e.rearrange(left_term, "bs n2 n1 -> bs n2 1 n1")
    right_term = e.rearrange(q_s_b, "bs n1 n2 -> bs 1 n1 n2")
    numerator = left_term * right_term

    edges_t_T = e.rearrange(edges_t_reformat, "bs m ee -> bs ee m")
    prod = q_t_b @ edges_t_T  # bs, d0, N
    prod = e.rearrange(prod, "bs n1 n2 -> bs n2 n1")
    denominator = e.rearrange(prod, "bs n ee -> bs n ee 1")
    denominator = np.where(denominator != 0, denominator, 1e-6)
    out = numerator / denominator
    out = e.rearrange(out, "bs (n1 n2) ee1 ee2 -> bs n1 n2 ee1 ee2", n1=n, n2=n)
    return out


# @jaxtyped
# @beartype


@jax.jit
def sample_p_zs_given_zt(
    p: GetProbabilityType,
    t: Int[Array, "batch"],
    g_t: gd.OneHotGraph,
    transition_model: TransitionModel,
    rng,
):
    """Samples from zs ~ p(zs | zt). Only used during sampling.
    if last_step, return the graph prediction as well"""

    pred = gd.softmax(p(g_t, t))

    (
        p_s_and_t_for_all_features_nodes,
        p_s_and_t_given_0_edges,
    ) = compute_posterior_for_all_features(
        g_t=g_t, t=t, transition_model=transition_model
    )

    # Dim of these two tensors: bs, N, d0, d_t-1
    weighted_nodes = (
        e.rearrange(pred.nodes, "bs n ne -> bs n ne 1")
        * p_s_and_t_for_all_features_nodes
    )  # bs, n, d0, d_t-1
    unnormalized_prob_nodes = weighted_nodes.sum(axis=2)  # bs, n, d_t-1
    unnormalized_prob_nodes = np.where(
        g_t.nodes_mask[..., None],
        unnormalized_prob_nodes,
        1e-5,
    )
    prob_nodes = unnormalized_prob_nodes / np.sum(
        unnormalized_prob_nodes, axis=-1, keepdims=True
    )  # bs, n, d_t-1

    # pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
    weighted_edges = (
        e.rearrange(pred.edges, "bs n1 n2 ee -> bs n1 n2 ee 1")
        * p_s_and_t_given_0_edges
    )  # bs, N, d0, d_t-1
    unnormalized_prob_edges = weighted_edges.sum(axis=-2)
    unnormalized_prob_edges = np.where(
        g_t.edges_mask[..., None],
        unnormalized_prob_edges,
        1e-5,
    )
    prob_edges = unnormalized_prob_edges / np.sum(
        unnormalized_prob_edges, axis=-1, keepdims=True
    )
    sampled_s = gd.sample_one_hot(
        gd.DenseGraphDistribution.create(
            nodes=prob_nodes,
            edges=prob_edges,
            nodes_mask=g_t.nodes_mask,
            edges_mask=g_t.edges_mask,
        ),
        rng_key=jax.random.fold_in(rng, enc("sample_p_zs_given_zt")),
    )

    return sampled_s


sample_step = sample_p_zs_given_zt
