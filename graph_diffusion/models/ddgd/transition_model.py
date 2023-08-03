import jax.numpy as np
from jax import Array
from jaxtyping import Float, Int
from mate.jax import SFloat, SInt
from flax.struct import dataclass
import jax
import ipdb

import einops as e
from ...shared.graph import graph_distribution as gd

Q, GraphDistribution = gd.Q, gd.GraphDistribution


def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    return betas.squeeze().astype(np.float32)[:timesteps]


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


@dataclass
class TransitionModel:
    diffusion_steps: SInt
    qs: Q
    q_bars: Q
    betas: Float[Array, "T 1"]
    alpha_bars: Float[Array, "T"]
    temporal_embeddings: Float[Array, "t1 temporal_embedding_dim"]
    limit_dist: gd.DenseGraphDistribution

    @classmethod
    def create(
        cls,
        nodes_prior: Float[Array, "n"],
        edge_prior: Float[Array, "m"],
        diffusion_steps: int,
        temporal_embedding_dim: int,
        n: SInt,
        schedule_type: str = "linear",
        adjust_prior=False,
        concat_flag_to_temporal_embeddings: bool | Int[Array, "flag"] = False,
    ) -> "TransitionModel":
        prior_type = ""
        if prior_type == "uniform":
            nodes_prior = np.ones(nodes_prior.shape[0]) / nodes_prior.shape[0]
            edge_prior = np.ones(edge_prior.shape[0]) / edge_prior.shape[0]
        elif prior_type == "reload":
            nodes_prior = np.array([0.7230, 0.1151, 0.1593, 0.0026])
            edge_prior = np.array([0.7261, 0.2384, 0.0274, 0.0081, 0.0000])

        if adjust_prior:
            x_prior_adj = np.where(nodes_prior > 0, nodes_prior, 1e-6)
            nodes_prior = x_prior_adj / x_prior_adj.sum()
            e_prior_adj = np.where(edge_prior > 0, edge_prior, 1e-6)
            edge_prior = e_prior_adj / e_prior_adj.sum()
        # prior = Distribution(x=x_priors, e=e_priors)

        node_types = len(nodes_prior)
        edge_types = len(edge_prior)
        u_x = e.repeat(nodes_prior, "p -> 1 x_classes p", x_classes=node_types)
        u_e = e.repeat(edge_prior, "p -> 1 e_classes p", e_classes=edge_types)
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
        temporal_embeddings = get_timestep_embedding(
            np.arange(diffusion_steps), temporal_embedding_dim
        )
        temporal_embeddings = np.concatenate(
            (temporal_embeddings, np.zeros((temporal_embeddings.shape[0], 1))), axis=1
        )
        if isinstance(concat_flag_to_temporal_embeddings, Array):
            # concat the same flag accross all timesteps
            temporal_embeddings = np.concatenate(
                (
                    temporal_embeddings,
                    np.ones((temporal_embeddings.shape[0], 1))
                    * concat_flag_to_temporal_embeddings,
                ),
                axis=1,
            )
        bs = 1
        limit_X = np.broadcast_to(
            np.expand_dims(nodes_prior, (0, 1)), (bs, n, nodes_prior.shape[-1])
        )
        limit_X /= limit_X.sum(-1, keepdims=True)
        limit_E = np.broadcast_to(
            np.expand_dims(edge_prior, (0, 1, 2)), (bs, n, n, edge_prior.shape[-1])
        )
        limit_dist = gd.DenseGraphDistribution.create_from_counts(
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
            temporal_embeddings=temporal_embeddings,
            limit_dist=limit_dist,
        )
