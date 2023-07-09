"""
A lot of functions related to diffusion models. But which do not depend on the on the model itself.
"""
from operator import ipow
from sys import exception
from beartype import beartype
from flax.training.common_utils import onehot
import ipdb
from collections.abc import Callable
import jax
from jax import Array, random
import jax
from jax.debug import print as jprint  # type: ignore
from jax import numpy as np
from flax import linen as nn
import mate as m
from mate.jax import SInt, SFloat, Key, typed, SBool
from jaxtyping import Int, Float, Bool
from einop import einop
from orbax.checkpoint.pytree_checkpoint_handler import Transform
from ...shared.graph import graph_distribution as gd
from .types import TransitionModel
from einop import einop

GraphDistribution = gd.GraphDistribution
EdgeDistribution = gd.EdgeDistribution
NodeDistribution = gd.NodeDistribution
Q = gd.Q

GetProbabilityType = Callable[
    [gd.OneHotGraph, Int[Array, "batch_size"]], gd.DenseGraphDistribution
]
check = lambda _, __: None


@typed
def pseudo_assert(condition: SBool):
    """
    When debugging NaNs, the following function will raise an exception
    """
    return np.where(condition, 0, np.nan)


# prob_true = diffusion_utils.posterior_distributions(
#     X=X,
#     E=E,
#     y=y,
#     X_t=noisy_data["X_t"],
#     E_t=noisy_data["E_t"],
#     y_t=noisy_data["y_t"],
#     Qt=Qt,
#     Qsb=Qsb,
#     Qtb=Qtb,
# )


# def compute_posterior_distribution(M, M_t, Qt_M, Qsb_M, Qtb_M):
#     """M: X or E
#     Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T
#
#     Compute xt @ Qt.T * x0 @ Qsb / sum(x0 @ Qtb * xt, -1)
#     """
#     # Flatten feature tensors
#     M = M.flatten(start_dim=1, end_dim=-2).to(
#         torch.float32
#     )  # (bs, N, d) with N = n or n * n
#     M_t = M_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)  # same
#
#     Qt_M_T = torch.transpose(Qt_M, -2, -1)  # (bs, d, d)
#
#     left_term = M_t @ Qt_M_T  # (bs, N, d)
#     right_term = M @ Qsb_M  # (bs, N, d)
#     product = left_term * right_term  # (bs, N, d)
#
#     denom = M @ Qtb_M  # (bs, N, d) @ (bs, d, d) = (bs, N, d)
#     denom = (denom * M_t).sum(dim=-1)  # (bs, N, d) * (bs, N, d) + sum = (bs, N)
#     # denom = product.sum(dim=-1)
#     # denom[denom == 0.] = 1
#
#     prob = product / denom.unsqueeze(-1)  # (bs, N, d)
#     return prob
#


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
    q_t_e_transposed = einop(q_t, "bs e1 e2 -> bs n e2 e1", n=nodes.shape[1])
    left_term = einop(nodes_t, q_t_e_transposed, "bs n de, bs n de e1 -> bs n e1")
    right_term = nodes @ q_s_bar
    product = left_term * right_term

    denom = nodes @ q_t_bar
    denom = einop(denom * nodes_t, "bs n e -> bs n", reduction="sum")
    denom = np.where(denom == 0, 1, denom)
    prob = product / denom[..., None]  # type:ignore
    pseudo_assert((prob >= 0).all())
    return prob


# def posterior_distributions(X, E, y, X_t, E_t, y_t, Qt, Qsb, Qtb):
#     prob_X = compute_posterior_distribution(
#         M=X, M_t=X_t, Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X
#     )  # (bs, n, dx)
#     prob_E = compute_posterior_distribution(
#         M=E, M_t=E_t, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E
#     )  # (bs, n * n, de)
#
#     return PlaceHolder(X=prob_X, E=prob_E, y=y_t)
#


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
    # print(f"{q_t_bar.nodes[0,0]=}")
    # print(f"{q_s_bar.nodes[0,0]=}")
    # print(f"{q_t.nodes[0,0]=}")
    prob_x = __compute_posterior_distribution_nodes(
        nodes=g.nodes,
        nodes_t=g_t.nodes,
        q_t=q_t.nodes,
        q_s_bar=q_s_bar.nodes,
        q_t_bar=q_t_bar.nodes,
    )
    # print(f"{prob_x[0,0]=}")
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


@typed
def compute_lt(
    get_probability: GetProbabilityType,
    g: gd.OneHotGraph,
    transition_model: TransitionModel,
    rng: Key,
):
    t = random.randint(
        rng, (g.nodes.shape[0],), 1, transition_model.diffusion_steps + 1
    )
    g_t = gd.sample_one_hot(gd.matmul(g, transition_model.q_bars[t]), rng)
    g_pred = get_probability(g_t, t)
    # Compute distributions to compare with KL
    # bs, n, d = X.shape
    return _compute_lt(
        t=t,
        g=g,
        g_t=g_t,
        g_pred=g_pred,
        transition_model=transition_model,
    )


from rich import print


@typed
def _compute_lt(
    t: Int[Array, "b"],
    g: gd.OneHotGraph,
    g_t: gd.OneHotGraph,
    g_pred: gd.DenseGraphDistribution,
    transition_model: TransitionModel,
):
    g_pred = gd.softmax(g_pred)
    q_t = transition_model.qs[t]
    q_t_bar = transition_model.q_bars[t]
    q_s_bar = transition_model.q_bars[t - 1]
    # print(f"{q_t_bar.nodes[0,0]=}")
    # print(f"{q_s_bar.nodes[0,0]=}")
    # print(f"{q_t.nodes[0,0]=}")
    prob_true = posterior_distribution(
        g=g,
        g_t=g_t,
        transition_model=transition_model,
        t=t,
    )
    # print(f"{prob_true.nodes[0,0]=}")
    prob_pred = posterior_distribution(
        g=g_pred,
        g_t=g_t,
        transition_model=transition_model,
        t=t,
    )
    # print(f"{prob_pred.nodes[0,0]=}")
    prob_true_masked = gd.normalize_and_mask(prob_true)
    prob_pred_masked = gd.normalize_and_mask(prob_pred)
    kl_div = gd.kl_div(
        prob_true_masked, prob_pred_masked
    )  # prob_pred_masked, prob_true_masked)
    return transition_model.diffusion_steps * kl_div


@typed
def compute_reconstruction_logp(
    *,
    rng_key: Key,
    get_probability: GetProbabilityType,
    g: gd.OneHotGraph,
    transition_model: TransitionModel,
):
    t = np.zeros(g.nodes.shape[0], dtype=int)
    q_t = transition_model.qs[t]
    g_t = gd.sample_one_hot(gd.matmul(g, q_t), rng_key)
    g_pred = gd.softmax(get_probability(g_t, t))
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
    return gd.logprobs_at(g_pred, g)


@typed
def compute_kl_prior(
    *,
    target: gd.OneHotGraph,
    transition_model: TransitionModel,
) -> Float[Array, "batch_size"]:
    """Computes the KL between q(z1 | x) and the prior p(z1) (extracted from the data)

    This is essentially a lot of work for something that is in practice negligible in the loss. However, you
    compute it so that you see it when you've made a mistake in your noise schedule.
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
) -> Float[Array, "batch_size"]:
    # 3. Diffusion loss
    loss_all_t = compute_lt(
        rng=rng_key,
        g=target,
        transition_model=transition_model,
        get_probability=get_probability,
    )
    # 4. Reconstruction loss
    # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
    reconstruction_logp = compute_reconstruction_logp(
        rng_key=rng_key,
        g=target,
        transition_model=transition_model,
        get_probability=get_probability,
    ).mean()
    # edges_counts = jax.lax.select(
    #     bits_per_edge, target.edges_counts * 2, np.ones(len(target.edges_counts), int)
    # )
    tot_loss = loss_all_t - reconstruction_logp
    return tot_loss


from jaxtyping import jaxtyped


# @jax.jit
@jaxtyped
@beartype
def compute_val_loss(
    *,
    target: gd.OneHotGraph,
    transition_model: TransitionModel,
    get_probability: GetProbabilityType,
    nodes_dist: Float[Array, "node_types"],
    rng_key: Key,
) -> dict[str, Float[Array, "batch_size"]]:
    # base = jax.lax.select(bits_per_edge, 2.0, np.e)

    # 1.  log_prob of the target graph under the nodes distribution (based on # of nodes)
    log_pn = np.log(nodes_dist[target.nodes_mask.sum(-1)])  # / np.log(base)

    # # 2. The KL between q(z_T | x) and p(z_T) = (simply an Empirical prior). Should be close to zero.
    # kl_prior = compute_kl_prior(
    #     target=target,
    #     transition_model=transition_model,
    #     # bits_per_edge=bits_per_edge,
    # )
    kl_prior = 0.0
    # 3. Diffusion loss
    loss_all_t = compute_lt(
        rng=rng_key,
        g=target,
        transition_model=transition_model,
        get_probability=get_probability,
    )
    # 4. Reconstruction loss
    # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
    reconstruction_logp = compute_reconstruction_logp(
        rng_key=rng_key,
        g=target,
        transition_model=transition_model,
        get_probability=get_probability,
    ).mean()
    # edges_counts = jax.lax.select(
    #     bits_per_edge, target.edges_counts * 2, np.ones(len(target.edges_counts), int)
    # )
    tot_loss = -log_pn + kl_prior + loss_all_t - reconstruction_logp
    return {
        "log_pn": log_pn,
        "kl_prior": kl_prior,
        "diffusion_loss": loss_all_t,
        "rec_logp": reconstruction_logp,
        "nll": tot_loss,
    }


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


def __compute_batched_over0_posterior_distribution_nodes(X_t, Qt, Qsb, Qtb):
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
    # X_t = X_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)  # bs x N x dt

    Qt_T = einop(Qt, "bs n1 n2 -> bs n2 n1")  # bs, dt, d_t-1
    left_term = X_t @ Qt_T  # bs, N, d_t-1
    left_term = einop(left_term, "bs n2 n1 -> bs n2 1 n1")  # bs, N, 1, d_t-1
    right_term = einop(Qsb, "bs n1 n2 -> bs 1 n1 n2")
    # .unsqueeze(1)  # bs, 1, d0, d_t-1
    numerator = left_term * right_term  # bs, N, d0, d_t-1
    X_t_transposed = einop(X_t, "bs n1 n2 -> bs n2 n1")
    # .transpose(-1, -2)  # bs, dt, N
    prod = Qtb @ X_t_transposed  # bs, d0, N
    prod = einop(prod, "bs n1 n2 -> bs n2 n1")
    # .transpose(-1, -2)  # bs, N, d0
    denominator = einop(prod, "bs n1 n2 -> bs n1 n2 1")
    # .unsqueeze(-1)  # bs, N, d0, 1
    denominator = np.where(denominator != 0, denominator, 1e-6)
    out = numerator / denominator
    return out


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
    # .transpose(-1, -2)  # bs, dt, d_t-1
    left_term = edges_t_reformat @ Qt_T  # bs, N, d_t-1
    left_term = einop(left_term, "bs n2 n1 -> bs n2 1 n1")
    # .unsqueeze(dim=2)  # bs, N, 1, d_t-1

    right_term = einop(Qsb, "bs n1 n2 -> bs 1 n1 n2")
    # .unsqueeze(1)  # bs, 1, d0, d_t-1
    numerator = left_term * right_term  # bs, N, d0, d_t-1

    X_t_transposed = einop(edges_t_reformat, "bs m ee -> bs ee m")
    # .transpose(-1, -2)  # bs, dt, N

    prod = Qtb @ X_t_transposed  # bs, d0, N
    prod = einop(prod, "bs n1 n2 -> bs n2 n1")
    # .transpose(-1, -2)  # bs, N, d0
    denominator = einop(prod, "bs n ee -> bs n ee 1")
    # .unsqueeze(-1)  # bs, N, d0, 1
    denominator = np.where(denominator != 0, denominator, 1e-6)
    out = numerator / denominator
    # out = einop(out, "bs (n1 n2) ee1 ee2 -> bs n1 n2 ee", n1=n, n2=n)
    return out


@jax.jit
@jaxtyped
@beartype
def sample_p_zs_given_zt(
    get_probability: GetProbabilityType,
    t: Int[Array, "batch"],
    g_t: gd.OneHotGraph,
    transition_model: TransitionModel,
    rng,
):
    """Samples from zs ~ p(zs | zt). Only used during sampling.
    if last_step, return the graph prediction as well"""
    # #bs, n, dxs = X_t.shape
    # beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
    # alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
    # alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)
    #
    # # Retrieve transitions matrix
    # Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
    # Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
    # Qt = self.transition_model.get_Qt(beta_t, self.device)
    #
    # Neural net predictions
    # noisy_data = {
    #     "X_t": X_t,
    #     "E_t": E_t,
    #     "y_t": y_t,
    #     "t": t,
    #     "node_mask": node_mask,
    # }
    # extra_data = self.compute_extra_data(noisy_data)
    # pred = self.forward(noisy_data, extra_data, node_mask)
    pred = gd.softmax(get_probability(g_t, t))

    # tmp = diffusion_utils.compute_batched_over0_posterior_distribution(
    #     X_t=X_t, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X
    # )
    # posterior = diffusion_utils.posterior_distributions(
    #     X_t=X_t, Qtb=Qtb, Qsb=Qsb, Qt=Qt
    # )
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
        unnormalized_prob_nodes.sum(axis=-1, keepdims=True) == 0,
        1e-5,
        unnormalized_prob_nodes,
    )
    prob_nodes = unnormalized_prob_nodes / np.sum(
        unnormalized_prob_nodes, axis=-1, keepdims=True
    )  # bs, n, d_t-1

    # pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
    weighted_edges = (
        einop(pred.edges, "bs n1 n2 ee -> bs (n1 n2) ee 1") * p_s_and_t_given_0_edges
    )  # bs, N, d0, d_t-1
    unnormalized_prob_edges = weighted_edges.sum(axis=-2)
    unnormalized_prob_edges = np.where(
        unnormalized_prob_edges.sum(axis=-1, keepdims=True) == 0,
        1e-5,
        unnormalized_prob_edges,
    )
    prob_edges = unnormalized_prob_edges / np.sum(
        unnormalized_prob_edges, axis=-1, keepdims=True
    )
    # prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

    # assert (np.abs(prob_nodes.sum(axis=-1) - 1) < 1e-4).all()
    # assert (np.abs(prob_edges.sum(axis=-1) - 1) < 1e-4).all()

    # sampled_s = diffusion_utils.sample_discrete_features(
    #     prob_X, prob_E, node_mask=node_mask
    # )
    prob_edges = einop(
        prob_edges, "bs (n1 n2) ee -> bs n1 n2 ee", n1=pred.nodes.shape[1]
    )
    sampled_s = gd.sample_one_hot(
        gd.create_dense(
            nodes=prob_nodes,
            edges=prob_edges,
            nodes_mask=g_t.nodes_mask,
            edges_mask=g_t.edges_mask,
        ),
        rng_key=rng,
    )

    # X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
    # E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()
    #
    # assert (E_s == torch.transpose(E_s, 1, 2)).all()
    # assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)
    #
    # out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
    # out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
    #
    # return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(
    #     node_mask, collapse=True
    # ).type_as(y_t)
    return sampled_s


if __name__ == "__main__":
    tests()
