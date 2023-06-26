"""
A lot of functions related to diffusion models. But which do not depend on the on the model itself.
"""
from operator import ipow
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
from ...shared.graph import graph_distribution as gd
from .types import TransitionModel
from einop import einop

GraphDistribution = gd.GraphDistribution
EdgeDistribution = gd.EdgeDistribution
NodeDistribution = gd.NodeDistribution
Q = gd.Q

GetProbabilityType = Callable[
    [GraphDistribution, Int[Array, "batch_size"]], GraphDistribution
]
check = lambda _, __: None


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
    g: GraphDistribution,
    g_t: GraphDistribution,
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
    prob_e = __compute_posterior_distribution_edges(
        edges=g.edges,
        edges_t=g_t.edges,
        q_t=q_t.edges,
        q_s_bar=q_s_bar.edges,
        q_t_bar=q_t_bar.edges,
    )
    # prob_x = prob_x / prob_x.sum(-1, keepdims=True)
    # prob_e = prob_e / prob_e.sum(-1, keepdims=True)
    # prob_x = np.clip(prob_x, 0, 1)
    # prob_e = np.clip(prob_e, 0, 1)
    return GraphDistribution.create(
        nodes=prob_x,
        edges=prob_e,
        edges_counts=g.edges_counts,
        nodes_counts=g.nodes_counts,
    )


# get_probability: GetProbabilityType,
#     g: GraphDistribution,
#     transition_model: TransitionModel,
#     rng: Key,


@typed
def compute_lt(
    get_probability: GetProbabilityType,
    g: GraphDistribution,
    transition_model: TransitionModel,
    rng: Key,
):
    t = random.randint(rng, (g.batch_size,), 1, transition_model.diffusion_steps + 1)
    g_t: GraphDistribution = (g @ transition_model.q_bars[t]).sample_one_hot(rng)
    g_pred: GraphDistribution = get_probability(g_t, t).softmax()
    # Compute distributions to compare with KL
    # bs, n, d = X.shape
    return _compute_lt(
        t=t,
        g=g,
        g_t=g_t,
        g_pred=g_pred,
        transition_model=transition_model,
    )


@typed
def _compute_lt(
    t: Int[Array, "b"],
    g: GraphDistribution,
    g_t: GraphDistribution,
    g_pred: GraphDistribution,
    transition_model: TransitionModel,
):
    prob_true = posterior_distribution(
        g=g,
        g_t=g_t,
        transition_model=transition_model,
        t=t,
    )
    prob_pred = posterior_distribution(
        g_pred,
        g_t,
        transition_model=transition_model,
        t=t,
    )
    prob_true_masked = gd.normalize_and_mask(prob_true)
    prob_pred_masked = gd.normalize_and_mask(prob_pred)
    kl_div = gd.kl_div(prob_true_masked, prob_pred_masked)
    return transition_model.diffusion_steps * kl_div


@typed
def compute_reconstruction_logp(
    *,
    rng_key: Key,
    get_probability: GetProbabilityType,
    g: GraphDistribution,
    transition_model: TransitionModel,
):
    t = np.zeros(g.batch_size, dtype=int)
    q_t = transition_model.qs[t]
    g_t = (g @ q_t).sample_one_hot(rng_key)
    g_pred = get_probability(g_t, t).softmax()
    return _compute_reconstruction_logp(
        g=g,
        g_pred=g_pred,
    )


def _compute_reconstruction_logp(
    g: GraphDistribution,
    g_pred: GraphDistribution,
) -> Float[Array, "batch_size"]:
    result = g_pred.logprobs_at(g)
    return result


@typed
def compute_kl_prior(
    *,
    target: GraphDistribution,
    transition_model: TransitionModel,
) -> Float[Array, "batch_size"]:
    """Computes the KL between q(z1 | x) and the prior p(z1) (extracted from the data)

    This is essentially a lot of work for something that is in practice negligible in the loss. However, you
    compute it so that you see it when you've made a mistake in your noise schedule.
    """
    # Compute the last alpha value, alpha_T.
    qt_bar_T = transition_model.q_bars[np.array([-1])]

    # Compute transition probabilities
    transition_probs = target @ qt_bar_T

    limit_dist = transition_model.limit_dist.repeat(len(target))
    return gd.kl_div(transition_probs, limit_dist)


@typed
def compute_train_loss(
    *,
    target: GraphDistribution,
    transition_model: TransitionModel,
    get_probability: Callable[
        [GraphDistribution, Int[Array, "batch_size"]], GraphDistribution
    ],
    nodes_dist: Array,
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


@typed
def compute_val_loss(
    *,
    target: GraphDistribution,
    transition_model: TransitionModel,
    get_probability: Callable[
        [GraphDistribution, Int[Array, "batch_size"]], GraphDistribution
    ],
    nodes_dist: Array,
    rng_key: Key,
) -> dict[str, Float[Array, "batch_size"]]:
    # base = jax.lax.select(bits_per_edge, 2.0, np.e)

    # 1.  log_prob of the target graph under the nodes distribution (based on # of nodes)
    log_pn = np.log(nodes_dist[target.nodes_counts])  # / np.log(base)

    # 2. The KL between q(z_T | x) and p(z_T) = (simply an Empirical prior). Should be close to zero.
    kl_prior = compute_kl_prior(
        target=target,
        transition_model=transition_model,
        # bits_per_edge=bits_per_edge,
    )
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


# @typed
# def compute_lt(
#     get_probability: GetProbabilityType,
#     g: GraphDistribution,
#     diffusion_steps: SInt,
#     transition_model: TransitionModel,
#     rng: Key,
# ) -> Float[Array, "batch_size"]:
#     t = random.randint(rng, (g.batch_size,), 1, diffusion_steps + 1)
#     q = transition_model.qs
#     q_bar = transition_model.q_bars
#
#     # q_t = transition_model.qs[t]
#     # q_s_bar = transition_model.q_bars[t - 1]
#     g_t = (g @ q_bar[t]).sample_one_hot(rng)
#     p = get_probability(g_t, t)
#     # f(G_{t-1}) = q(G_{t-1} | G, G_t) = q(G_t | G, G_{t-1})q(G_{t-1} | G) / q(G_t | G)
#     # q_t_bar = transition_model.q_bars[t]
#
#     # use bayes rule to compute q(z_s | z_t, g)
#     # __unsafe=True because at that multiplication, the graph is not a distribution.
#     # but it will be after the division.
#     q_num = (g @ q[t]).__mul__(g @ q_bar[t - 1], _safe=False)
#     denom = g @ q_bar[t]
#     # q_denom_nodes, q_denom_edges = denom.nodes.sum(-1), denom.edges.sum(-1)
#     # q = GraphDistribution.create(
#     #     q_num.nodes / q_num.nodes.sum(-1)[..., None],  # q_denom_nodes[..., None],
#     #     q_num.edges / q_num.edges.sum(-1)[..., None],  # q_denom_edges[..., None],
#     #     edges_counts=q_num.edges_counts,
#     #     nodes_counts=q_num.nodes_counts,
#     # )
#     q = q_num / denom
#     result = gd.kl_div(q, p)
#     return result * diffusion_steps
#


def tests():
    jax.config.update("jax_debug_nans", True)
    import numpy as npo

    def to_jax(x):
        if isinstance(x, (list, tuple)):
            return [to_jax(y) for y in x]
        if isinstance(x, dict):
            if "type" in x and x["type"] == "graph_dist":
                return GraphDistribution.from_mask(
                    to_jax(x["X"]),
                    to_jax(x["E"]),
                    to_jax(x["node_mask"]),
                )
            return {k: to_jax(v) for k, v in x.items()}
        if isinstance(x, npo.ndarray):
            return np.array(x)
        return x

    import pickle

    re_logp_in_out = to_jax(pickle.load(open("in_out_tests/rec_lopg.pt", "rb")))

    g = re_logp_in_out["input"]["g"]  # type: GraphDistribution
    g_0 = re_logp_in_out["input"]["prob0"]  # type: GraphDistribution
    target_output = re_logp_in_out["output"]["loss_term_0"]  # type: Array
    pred_output = _compute_reconstruction_logp(g, g_0).mean()
    assert np.allclose(pred_output, target_output), ipdb.set_trace()
    lt_in_out = to_jax(pickle.load(open("in_out_tests/lt.pt", "rb")))
    g = lt_in_out["input"]["g"]  # type: GraphDistribution
    pred_prob = lt_in_out["input"]["prob"]  # type: GraphDistribution
    pred_prob = GraphDistribution.create(
        jax.nn.softmax(pred_prob.nodes),
        jax.nn.softmax(pred_prob.edges),
        nodes_counts=g.nodes_counts,
        edges_counts=g.edges_counts,
    )
    noisy_data = lt_in_out["input"]["noisy_data"]  # type: dict[str, Array]
    g_t = GraphDistribution.from_mask(
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
