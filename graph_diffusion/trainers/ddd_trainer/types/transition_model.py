import jax.numpy as np
from jax import Array
from jaxtyping import Float, Int
from mate.jax import SFloat, SInt, typed
import jax_dataclasses as jdc
import jax
import ipdb

# from .noise_schedule import NoiseSchedule
from ....shared.graph_distribution import Q
from .distribution import Distribution
from .noise_schedules import NoiseSchedule_Scalar


@typed
def cosine_beta_schedule_discrete(diffusion_steps: SInt, s=0.008) -> Float[Array, "n"]:
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = diffusion_steps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    return betas.squeeze().astype(np.float32)


@typed
def compute_noise_schedule(
    diffusion_steps: SInt,
) -> tuple[Float[Array, "n"], Float[Array, "n"], Float[Array, "n"]]:
    betas = cosine_beta_schedule_discrete(diffusion_steps, 10000.0)
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


@jdc.pytree_dataclass
class TransitionModel(jdc.EnforcedAnnotationsMixin):
    prior: Distribution
    diffusion_steps: SInt
    qs: Q
    q_bars: Q
    alpha_bars: Float[Array, "n"]
    temporal_embeddings: Float[Array, "temporal_embedding_dim"]

    @classmethod
    @typed
    def create(
        cls,
        x_priors: Float[Array, "n"],
        e_priors: Float[Array, "m"],
        diffusion_steps: int,
        temporal_embedding_dim: int,
    ) -> "TransitionModel":
        import matplotlib.pyplot as plt

        # plt.plot(x_priors)
        # x_priors = jax.nn.softmax((x_priors + 1e-6))
        # plt.plot(x_priors)
        # plt.show()
        # e_priors = jax.nn.softmax(e_priors + 1e-6)
        prior_type = "custom"
        if prior_type == "uniform":
            x_priors = np.ones(x_priors.shape[0]) / x_priors.shape[0]
            e_priors = np.ones(e_priors.shape[0]) / e_priors.shape[0]
        prior = Distribution(x=x_priors, e=e_priors)
        x_classes = len(x_priors)
        e_classes = len(e_priors)
        u_x = np.broadcast_to(x_priors[None, None], (1, x_classes, x_priors.shape[0]))
        u_e = np.broadcast_to(e_priors[None, None], (1, e_classes, e_priors.shape[0]))
        # u_y = np.ones((1, y_classes, y_classes)) / (y_classes if y_classes > 0 else 1)
        # noise_schedule = NoiseSchedule.create(0, diffusion_steps)  # 0 is cosine
        betas, _, alphas_bar = compute_noise_schedule(diffusion_steps)
        betas = betas[:, None, None]
        q_xs = betas * u_x + (1 - betas) * np.eye(x_classes)[None]
        q_es = (
            betas * u_e
            + (1 - betas)
            * np.eye(
                e_classes,
            )[None]
        )
        qs = Q(x=q_xs, e=q_es)
        alpha_bars = alphas_bar[:, None, None]
        q_bar_xs = alpha_bars * np.eye(x_classes)[None] + (1 - alpha_bars) * u_x
        q_bar_es = alpha_bars * np.eye(e_classes)[None] + (1 - alpha_bars) * u_e
        q_bars = Q(x=q_bar_xs, e=q_bar_es)
        # q_bars_test = qs.cumulative_matmul()
        temporal_embeddings = get_timestep_embedding(
            np.arange(diffusion_steps), temporal_embedding_dim
        )
        # corresponds to a null node
        temporal_embeddings = np.concatenate(
            (temporal_embeddings, np.zeros((temporal_embeddings.shape[0], 1))), axis=1
        )
        return cls(
            diffusion_steps=diffusion_steps,
            qs=qs,
            q_bars=q_bars,
            prior=prior,
            alpha_bars=alphas_bar,
            temporal_embeddings=temporal_embeddings,
        )
