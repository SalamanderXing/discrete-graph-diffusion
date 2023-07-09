import jax.numpy as np
from jax import Array
from jaxtyping import Float, Int
from mate.jax import SFloat, SInt, typed
from flax.struct import dataclass
import jax
import ipdb

# from .noise_schedule import NoiseSchedule
import einops as e
from ....shared.graph import graph_distribution as gd
from .distribution import Distribution
from .noise_schedules import NoiseSchedule_Scalar
from einop import einop

Q, GraphDistribution = gd.Q, gd.GraphDistribution


# @typed
# def cosine_beta_schedule_discrete(diffusion_steps: SInt, s=0.008) -> Float[Array, "n"]:
#     """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ."""
#     steps = diffusion_steps + 2
#     x = np.linspace(0, steps, steps)
#
#     alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
#     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#     alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
#     betas = 1 - alphas
#     return betas.squeeze().astype(np.float32)


def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    return betas.squeeze().astype(np.float32)[:timesteps]


@typed
def compute_noise_schedule(
    diffusion_steps: SInt,
    schedule_type: str = "cosine",
) -> tuple[Float[Array, "n"], Float[Array, "n"], Float[Array, "n"]]:
    beta_functions = {
        "cosine": lambda: cosine_beta_schedule_discrete(diffusion_steps),
        "linear": lambda: np.linspace(0.0, 1.0, diffusion_steps),
    }
    assert (
        schedule_type in beta_functions
    ), f"schedule_type must be one of {list(beta_functions.keys())}"

    betas = beta_functions[schedule_type]()
    # betas will be a linear schedule
    # betas = np.linspace(0.0, 1.0, diffusion_steps)
    alphas = 1 - np.clip(betas, a_min=0, a_max=0.9999)

    log_alpha = np.log(alphas)
    log_alpha_bar = np.cumsum(log_alpha, axis=0)
    alphas_bar = np.exp(log_alpha_bar)
    return betas, alphas, alphas_bar


def get_timestep_embedding(
    diffusion_steps: Int[Array, "diffusion_steps"], embedding_dim: int, dtype=np.float32
):
    """Build sinusoidal embeddings (from Fairseq)."""

    assert len(diffusion_steps.shape) == 1
    diffusion_steps *= 1000

    half_dim = embedding_dim // 2
    emb = np.log(10_000) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim, dtype=dtype) * -emb)
    emb = diffusion_steps.astype(dtype)[:, None] * emb[None, :]
    emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
    assert emb.shape == (diffusion_steps.shape[0], embedding_dim)
    return emb


# @typed
# def cumulative_matmul(qs: Q):
#     def f(a, b):
#         result = a @ b
#         return result, result
#
#     return jax.lax.scan(f, qs, qs[0])[0]
#


@dataclass
class TransitionModel:
    # prior: Distribution
    diffusion_steps: SInt
    qs: Q
    q_bars: Q
    betas: Float[Array, "T"]
    alpha_bars: Float[Array, "T"]
    temporal_embeddings: Float[Array, "temporal_embedding_dim"]
    limit_dist: gd.DenseGraphDistribution

    @classmethod
    @typed
    def from_dict(cls, d: dict):
        return cls(
            diffusion_steps=d["diffusion_steps"],
            qs=Q(nodes=d["qs"]["nodes"], edges=d["qs"]["edges"]),
            q_bars=Q(nodes=d["q_bars"]["nodes"], edges=d["q_bars"]["edges"]),
            temporal_embeddings=np.zeros(128),
            limit_dist=gd.create_dense_from_counts(
                nodes=d["limit_dist"]["nodes"],
                edges=d["limit_dist"]["edges"],
                nodes_counts=d["limit_dist"]["nodes_counts"],
            ),
        )

    @classmethod
    @typed
    def from_torch(cls, torch_transition_model, torch_noise_schedule):
        import torch as t

        device = t.device("cpu")
        betas = [
            torch_noise_schedule(t.tensor([i], device=device))
            for i in range(len(torch_noise_schedule.betas + 1))
        ]
        qs_raw = [
            torch_transition_model.get_Qt(beta_t[None], device) for beta_t in betas
        ]

        ipdb.set_trace()
        q_bars_raw = [
            torch_transition_model.get_Qt_bar(beta_t[None], device)
            for beta_t in torch_noise_schedule.betas
        ]
        q_nodes = einop([np.array(q.X.numpy()) for q in qs_raw], "* a b")
        q_bars_nodes = einop([np.array(q.X.numpy()) for q in q_bars_raw], "* a b")
        q_edges = einop([np.array(q.E.numpy()) for q in qs_raw], "* a b")
        q_bars_edges = einop([np.array(q.E.numpy()) for q in q_bars_raw], "* a b")
        # q_nodes = q_nodes + 0.00001
        # q_nodes = q_nodes / q_nodes.sum(axis=-1, keepdims=True)
        # q_edges = q_edges + 0.00001
        # q_edges = q_edges / q_edges.sum(axis=-1, keepdims=True)
        # q_bars_nodes = q_bars_nodes + 0.00001
        # q_bars_nodes = q_bars_nodes / q_bars_nodes.sum(axis=-1, keepdims=True)
        # q_bars_edges = q_bars_edges + 0.00001
        # q_bars_edges = q_bars_edges / q_bars_edges.sum(axis=-1, keepdims=True)
        qs = Q(nodes=q_nodes, edges=q_edges)
        q_bars = Q(nodes=q_bars_nodes, edges=q_bars_edges)
        limit_edges = einop(
            np.array(torch_transition_model.u_e[0, 0]), "ee -> 1 n1 n2 ee", n1=9, n2=9
        )  # FIXME: get te actual nodes counts
        limit_nodes = einop(
            np.array(torch_transition_model.u_x[0, 0].numpy()), "en -> 1 n en", n=9
        )
        print(limit_edges.shape)
        limit_dist = gd.create_dense_from_counts(
            nodes=limit_nodes,
            edges=limit_edges,
            nodes_counts=np.ones(limit_nodes.shape[0], int),
        )
        return cls(
            diffusion_steps=torch_noise_schedule.timesteps,
            qs=qs,
            q_bars=q_bars,
            temporal_embeddings=np.zeros(128),
            limit_dist=limit_dist,
        )

    @classmethod
    @typed
    def create(
        cls,
        x_priors: Float[Array, "n"],
        e_priors: Float[Array, "m"],
        diffusion_steps: int,
        temporal_embedding_dim: int,
        n: SInt,
        schedule_type: str = "linear",
        adjust_prior=False,
    ) -> "TransitionModel":
        import matplotlib.pyplot as plt

        # plt.plot(x_priors)
        # x_priors = jax.nn.softmax((x_priors + 1e-6))
        # plt.plot(x_priors)
        # plt.show()
        # e_priors = jax.nn.softmax(e_priors + 1e-6)
        prior_type = "reload"
        if prior_type == "uniform":
            x_priors = np.ones(x_priors.shape[0]) / x_priors.shape[0]
            e_priors = np.ones(e_priors.shape[0]) / e_priors.shape[0]
        elif prior_type == "reload":
            x_priors = np.array([0.7230, 0.1151, 0.1593, 0.0026])
            e_priors = np.array([0.7261, 0.2384, 0.0274, 0.0081, 0.0000])

        if adjust_prior:
            x_prior_adj = np.where(x_priors > 0, x_priors, 1e-6)
            x_priors = x_prior_adj / x_prior_adj.sum()
            e_prior_adj = np.where(e_priors > 0, e_priors, 1e-6)
            e_priors = e_prior_adj / e_prior_adj.sum()
        prior = Distribution(x=x_priors, e=e_priors)

        node_types = len(x_priors)
        edge_types = len(e_priors)
        # u_x = np.broadcast_to(x_priors[None, None], (1, x_classes, x_priors.shape[0]))
        # u_e = np.broadcast_to(e_priors[None, None], (1, e_classes, e_priors.shape[0]))
        u_x = e.repeat(x_priors, "p -> 1 x_classes p", x_classes=node_types)
        u_e = e.repeat(e_priors, "p -> 1 e_classes p", e_classes=edge_types)
        # u_y = np.ones((1, y_classes, y_classes)) / (y_classes if y_classes > 0 else 1)
        # noise_schedule = NoiseSchedule.create(0, diffusion_steps)  # 0 is cosine
        betas, _, alphas_bar = compute_noise_schedule(diffusion_steps, schedule_type)
        betas = np.concatenate([betas, np.ones(1)])
        alphas_bar = np.concatenate([alphas_bar, np.zeros(1)])
        betas = betas[:, None, None]
        Ie = np.eye(
            edge_types,
        )[None]
        In = np.eye(
            node_types,
        )[None]
        q_xs = betas * u_x + (1 - betas) * In
        q_es = betas * u_e + (1 - betas) * Ie
        qs = Q(nodes=q_xs, edges=q_es)
        betas_bar = 1 - alphas_bar[:, None, None]
        q_bar_xs = betas_bar * u_x + (1 - betas_bar) * In
        q_bar_es = betas_bar * u_e + (1 - betas_bar) * Ie
        q_bars = Q(nodes=q_bar_xs, edges=q_bar_es)
        # q_bars = qs.cumulative_matmul()

        # q_bars_test = qs.cumulative_matmul()
        temporal_embeddings = get_timestep_embedding(
            np.arange(diffusion_steps), temporal_embedding_dim
        )
        # corresponds to a null node
        temporal_embeddings = np.concatenate(
            (temporal_embeddings, np.zeros((temporal_embeddings.shape[0], 1))), axis=1
        )

        bs = 1
        limit_X = np.broadcast_to(
            np.expand_dims(prior.x, (0, 1)), (bs, n, prior.x.shape[-1])
        )
        limit_X /= limit_X.sum(-1, keepdims=True)
        limit_E = np.broadcast_to(
            np.expand_dims(prior.e, (0, 1, 2)), (bs, n, n, prior.e.shape[-1])
        )
        limit_dist = gd.create_dense_from_counts(
            nodes=limit_X,
            edges=limit_E,
            nodes_counts=np.array([n]),
        )
        return cls(
            diffusion_steps=diffusion_steps,
            qs=qs,
            q_bars=q_bars,
            betas=betas.squeeze(-1),  # these two are for debugging
            alpha_bars=alphas_bar,
            # prior=prior,
            # alpha_bars=alphas_bar,
            temporal_embeddings=temporal_embeddings,
            limit_dist=limit_dist,
        )
