"""
A lot of functions related to diffusion models. But which do not depend on the on the model itself.
"""
from rich import print
from beartype import beartype
import ipdb
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
from einop import einop
import hashlib
from orbax.checkpoint.pytree_checkpoint_handler import Transform
from ...shared.graph import graph_distribution as gd
from .types import TransitionModel
from einop import einop
import ipdb

GraphDistribution = gd.GraphDistribution
EdgeDistribution = gd.EdgeDistribution
NodeDistribution = gd.NodeDistribution
Q = gd.Q

GetProbabilityType = Callable[
    [gd.OneHotGraph, Int[Array, "batch_size"]], gd.DenseGraphDistribution
]

import sys

sys.path.insert(0, "/home/bluesk/Documents/tmp/digress-copy/")

# from src.diffusion import diffusion_utils
from src import diffusion_model_discrete
from src.diffusion import diffusion_utils

digress = diffusion_model_discrete.DiscreteDenoisingDiffusion(
    cfg=None,
    dataset_infos=None,
    train_metrics=None,
    sampling_metrics=None,
    visualization_tools=None,
    extra_features=None,
    domain_features=None,
    reload=True,
)


### TMP TORCH STUFF

import torch


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """Changes the device and dtype of X, E, y."""
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = -1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self


def compute_posterior_distribution(M, M_t, Qt_M, Qsb_M, Qtb_M):
    """M: X or E
    Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T
    """
    # Flatten feature tensors
    M = M.flatten(start_dim=1, end_dim=-2).to(
        torch.float32
    )  # (bs, N, d) with N = n or n * n
    M_t = M_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)  # same

    Qt_M_T = torch.transpose(Qt_M, -2, -1)  # (bs, d, d)

    left_term = M_t @ Qt_M_T  # (bs, N, d)
    right_term = M @ Qsb_M  # (bs, N, d)
    product = left_term * right_term  # (bs, N, d)

    denom = M @ Qtb_M  # (bs, N, d) @ (bs, d, d) = (bs, N, d)
    denom = (denom * M_t).sum(dim=-1)  # (bs, N, d) * (bs, N, d) + sum = (bs, N)
    # denom = product.sum(dim=-1)
    # denom[denom == 0.] = 1

    prob = product / denom.unsqueeze(-1)  # (bs, N, d)

    return prob


def posterior_distributions(X, E, y, X_t, E_t, y_t, Qt, Qsb, Qtb):
    prob_X = compute_posterior_distribution(
        M=X, M_t=X_t, Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X
    )  # (bs, n, dx)
    prob_E = compute_posterior_distribution(
        M=E, M_t=E_t, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E
    )  # (bs, n * n, de)

    return PlaceHolder(X=prob_X, E=prob_E, y=y_t)


def to_torch(x):
    return torch.tensor(x.tolist())


def tt(args):
    if isinstance(args, Array):
        return to_torch(args)
    elif isinstance(args, (tuple, list)):
        return tuple(tt(x) for x in args)
    elif isinstance(args, dict):
        return {key: tt(val) for key, val in args.items()}
    elif isinstance(args, gd.GraphDistribution):
        return PlaceHolder(
            X=tt(args.nodes), E=tt(args.edges), y=torch.zeros(args.edges.shape[0])
        )
    else:
        return args


def to_gd_one_hot(x, orig):
    nodes = np.array(x.X)
    edges = np.array(x.E)
    nodes = np.where(orig.nodes_mask, nodes, 0)
    edges = np.where(orig.edges_mask, edges, 0)
    nodes = jax.nn.one_hot(nodes, num_classes=orig.nodes.shape[-1])
    edges = jax.nn.one_hot(np.array(x.E), num_classes=orig.edges.shape[-1])
    nodes = np.where(orig.nodes_mask[..., None], nodes, 0)
    edges = np.where(orig.edges_mask[..., None], edges, 0)

    return gd.create_one_hot(
        nodes=nodes,
        edges=edges,
        nodes_mask=orig.nodes_mask,
        edges_mask=orig.edges_mask,
    )


def to_gd_dense(x, orig):
    nodes = np.array(x.X)
    nodes = np.where(np.isnan(nodes), 0, nodes)
    edges = np.array(x.E.reshape(orig.edges.shape))
    edges = np.where(np.isnan(edges), 0, edges)
    return gd.create_dense(
        nodes=nodes,
        edges=edges,
        nodes_mask=orig.nodes_mask,
        edges_mask=orig.edges_mask,
    )


def to_q(x):
    return gd.Q(
        nodes=np.array(x.X),
        edges=np.array(x.E),
    )


def q_to_torch(q):
    return PlaceHolder(
        X=to_torch(q.nodes),
        E=to_torch(q.edges),
        y=None,
    )


class DummyModel(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, X, E, y, node_mask):
        y = y.squeeze(-1)
        res = self.p(
            gd.create_one_hot_minimal(
                nodes=np.array(X), edges=np.array(E), nodes_mask=np.array(node_mask)
            ),
            np.array(y),
        )
        res_torch = tt(res)
        return res_torch


## END TMP TORCH STUFF


def enc(x: str):
    return int(hashlib.md5(x.encode()).hexdigest()[:8], base=16)


def typed(f):
    return jaxtyped(beartype(f))


@typed
def pseudo_assert(condition: SBool):
    """
    When debugging NaNs, the following function will raise an exception
    """
    return np.where(condition, 0, np.nan)


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
    q_t_T = einop(q_t, "bs e1 e2 -> bs n e2 e1", n=nodes.shape[1])
    left_term = einop(nodes_t, q_t_T, "bs n de, bs n de e1 -> bs n e1")
    right_term = nodes @ q_s_bar
    product = left_term * right_term

    denom = nodes @ q_t_bar
    denom = einop(denom * nodes_t, "bs n e -> bs n", reduction="sum")
    denom = np.where(denom == 0, 1, denom)
    prob = product / denom[..., None]  # type:ignore
    pseudo_assert((prob >= 0).all())
    return prob


@typed
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
    return gd.create_dense(
        nodes=prob_x,
        edges=prob_e,
        nodes_mask=g.nodes_mask,
        edges_mask=g.edges_mask,
    )


def rng_to_seed(rng):
    a, b = rng.tolist()
    return int(f"{a}{b}")


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


@typed
def _compute_lt(
    t: Int[Array, "b"],
    g: gd.OneHotGraph,
    g_t: gd.OneHotGraph,
    g_pred: gd.DenseGraphDistribution,
    transition_model: TransitionModel,
):
    g_pred = norm_graph(g_pred)
    posterior_prob_true = posterior_distribution(
        g=g,
        g_t=g_t,
        transition_model=transition_model,
        t=t,
    )
    posterior_prob_pred = posterior_distribution(
        g=g_pred,
        g_t=g_t,
        transition_model=transition_model,
        t=t,
    )
    kl_div = gd.kl_div(posterior_prob_true, posterior_prob_pred)
    result = transition_model.diffusion_steps * kl_div
    return result


def minmax_norm(x):
    num = x - x.min(-1, keepdims=True)
    denom = num.sum(-1, keepdims=True)
    denom = np.where(denom == 0, 1, denom)
    return (num / denom) + 1e-7


def norm_graph(g):
    return gd.softmax(g)


@typed
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
    g_pred = norm_graph(p(z_1, t_0))
    return _compute_reconstruction_logp(
        g=g,
        g_pred=g_pred,
    )


@typed
def _compute_reconstruction_logp(
    g: gd.OneHotGraph,
    g_pred: gd.DenseGraphDistribution,
) -> Float[Array, "batch_size"]:
    # g_pred = gd.softmax(g_pred)
    return gd.logprobs_at(g, g_pred)  # gd.logprobs_at(g_pred, g)


@typed
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

    limit_dist = gd.repeat_dense(transition_model.limit_dist, target.nodes.shape[0])
    return gd.kl_div(transition_probs, limit_dist)


@typed
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
    loss_type = "elbo_ce"
    if "ce" in loss_type:
        ce_term = gd.cross_entropy(g_pred, target).mean()
    if "elbo" in loss_type:
        loss_all_t = _compute_lt(
            t=t, g=target, g_t=g_t, g_pred=g_pred, transition_model=transition_model
        ).mean()
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
        tot_loss = gd.cross_entropy(g_pred, target, np.array([1.0, 5.0]))
    return tot_loss


from jaxtyping import jaxtyped


# @jax.jit
@jaxtyped
@beartype
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
    # # 3. Diffusion loss
    # loss_all_t, t = compute_lt(
    #     rng=rng_lt,
    #     g=target,
    #     transition_model=transition_model,
    #     get_probability=get_probability,
    # )
    t, g_t, g_pred = predict_from_random_timesteps(p, target, transition_model, rng_lt)
    loss_all_t = _compute_lt(
        t=t, g=target, g_t=g_t, g_pred=g_pred, transition_model=transition_model
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


def compute_batched_over0_posterior_distribution(
    g_t: gd.OneHotGraph, transition_model: TransitionModel, t
):
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
    # Flatten feature tensors
    # Careful with this line. It does nothing if X is a node feature. If X is an edge features it maps to
    # bs x (n ** 2) x d
    q_t_T = einop(q_t, "bs ne1 ne2 -> bs ne2 ne1")  # bs, dt, d_t-1
    left_term = nodes_t @ q_t_T  # bs, N, d_t-1
    left_term = einop(left_term, "bs n ne -> bs n 1 ne")
    right_term = einop(q_s_b, "bs ne1 ne2 -> bs 1 ne1 ne2")
    numerator = left_term * right_term

    nodes_t_T = einop(nodes_t, "bs n ne -> bs ne n")
    denominator = einop(q_t_b @ nodes_t_T, "bs ne n -> bs n ne 1")
    denominator = np.where(denominator != 0, denominator, 1e-6)

    out = numerator / denominator
    return out


from einops import rearrange


def __compute_batched_over0_posterior_distribution_edges(edges_t, Qt, Qsb, Qtb):
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
    n = edges_t.shape[1]
    edges_t_reformat = einop(edges_t, "b n1 n2 ee -> b (n1 n2) ee")
    # .flatten(start_dim=1, end_dim=-2).to(torch.float32)  # bs x N x dt

    Qt_T = einop(Qt, "b n1 n2 -> b n2 n1")
    left_term = edges_t_reformat @ Qt_T
    left_term = einop(left_term, "bs n2 n1 -> bs n2 1 n1")
    right_term = einop(Qsb, "bs n1 n2 -> bs 1 n1 n2")
    numerator = left_term * right_term

    X_t_transposed = einop(edges_t_reformat, "bs m ee -> bs ee m")
    prod = Qtb @ X_t_transposed  # bs, d0, N
    prod = einop(prod, "bs n1 n2 -> bs n2 n1")
    denominator = einop(prod, "bs n ee -> bs n ee 1")
    denominator = np.where(denominator != 0, denominator, 1e-6)
    out = numerator / denominator
    out = einop(out, "bs (n1 n2) ee1 ee2 -> bs n1 n2 ee1 ee2", n1=n, n2=n)
    return out


@jax.jit
@jaxtyped
@beartype
def sample_p_zs_given_zt(
    p: GetProbabilityType,
    t: Int[Array, "batch"],
    g_t: gd.OneHotGraph,
    transition_model: TransitionModel,
    rng,
):
    """Samples from zs ~ p(zs | zt). Only used during sampling.
    if last_step, return the graph prediction as well"""

    pred = norm_graph(p(g_t, t))

    (
        p_s_and_t_given_0_nodes,
        p_s_and_t_given_0_edges,
    ) = compute_batched_over0_posterior_distribution(
        g_t=g_t, t=t, transition_model=transition_model
    )

    # Dim of these two tensors: bs, N, d0, d_t-1
    weighted_nodes = (
        einop(pred.nodes, "bs n ne -> bs n ne 1") * p_s_and_t_given_0_nodes
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
        einop(pred.edges, "bs n1 n2 ee -> bs n1 n2 ee 1") * p_s_and_t_given_0_edges
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
        gd.create_dense(
            nodes=prob_nodes,
            edges=prob_edges,
            nodes_mask=g_t.nodes_mask,
            edges_mask=g_t.edges_mask,
        ),
        rng_key=jax.random.fold_in(rng, enc("sample_p_zs_given_zt")),
    )

    return sampled_s


def tests():
    jax.config.update("jax_debug_nans", True)
    import numpy as npo

    def to_jax(x):
        if isinstance(x, (list, tuple)):
            return [to_jax(y) for y in x]
        if isinstance(x, dict):
            if x.get("type") == "graph_dist":
                cls = (
                    gd.OneHotGraph
                    if (x["X"] != 0).sum(-1).max() == 1
                    else gd.DenseGraphDistribution
                )
                print(f"converting {cls.__name__}")
                res = cls.from_mask(
                    to_jax(x["X"]),
                    to_jax(x["E"]),
                    to_jax(x["node_mask"]),
                )
                print(f"converted {res.__class__.__name__}")
                return res
            return {k: to_jax(v) for k, v in x.items()}
        if isinstance(x, npo.ndarray):
            return np.array(x)
        return x

    import pickle

    re_logp_in_out = to_jax(pickle.load(open("in_out_tests/rec_lopg.pt", "rb")))

    g = re_logp_in_out["input"]["g"]  # type: GraphDistribution
    g_0 = re_logp_in_out["input"]["prob0"]  # type: GraphDistribution
    target_output = re_logp_in_out["output"]["loss_term_0"]  # type: Array
    # ipdb.set_trace()
    pred_output = _compute_reconstruction_logp(g, g_0).mean()
    assert np.allclose(pred_output, target_output), ipdb.set_trace()
    lt_in_out = to_jax(pickle.load(open("in_out_tests/lt.pt", "rb")))
    g = lt_in_out["input"]["g"]  # type: gd.OneHotGraph
    pred_prob = lt_in_out["input"]["prob"]  # type: gd.DenseGraphDistribution
    pred_prob = gd.create_dense(
        jax.nn.softmax(pred_prob.nodes),
        jax.nn.softmax(pred_prob.edges),
        nodes_mask=g.nodes_mask,
        edges_mask=g.edges_mask,
    )
    noisy_data = lt_in_out["input"]["noisy_data"]  # type: dict[str, Array]
    g_t = gd.OneHotGraph.from_mask(
        noisy_data["X_t"].astype(np.float32),
        noisy_data["E_t"].astype(np.float32),
        noisy_data["node_mask"],
    )
    transition_model = TransitionModel.from_dict(lt_in_out["input"]["transition_model"])
    t = (
        np.round(noisy_data["t"] * transition_model.diffusion_steps)
        .astype(int)
        .squeeze(-1)
    )  # type: Array
    target_output = lt_in_out["output"]["loss_all_t"]  # type: Array
    pred_output = _compute_lt(
        t=t, g=g, g_pred=pred_prob, g_t=g_t, transition_model=transition_model
    ).mean()
    assert np.allclose(pred_output, target_output), ipdb.set_trace()


if __name__ == "__main__":
    tests()
