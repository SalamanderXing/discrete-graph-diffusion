import jax.numpy as np
from .q import Q
from abc import ABC, abstractmethod
from jax import Array
from jaxtyping import Float
from mate.jax import SFloat, SInt, typed
import jax_dataclasses as jdc
import ipdb

# from .noise_schedule import NoiseSchedule
from .q import Q
from .distribution import Distribution


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
    betas = cosine_beta_schedule_discrete(diffusion_steps)
    alphas = 1 - np.clip(betas, a_min=0, a_max=0.9999)

    log_alpha = np.log(alphas)
    log_alpha_bar = np.cumsum(log_alpha, axis=0)
    alphas_bar = np.exp(log_alpha_bar)
    return betas, alphas, alphas_bar


@jdc.pytree_dataclass
class TransitionModel(jdc.EnforcedAnnotationsMixin):
    prior: Distribution
    y_classes: SInt
    diffusion_steps: SInt
    qs: Q
    q_bars: Q
    betas: Float[Array, "n"]
    alphas: Float[Array, "n"]
    alpha_bars: Float[Array, "n"]

    @classmethod
    @typed
    def create(
        cls,
        x_priors: Float[Array, "n"],
        e_priors: Float[Array, "m"],
        y_classes: SInt,
        diffusion_steps: SInt,
    ) -> "TransitionModel":
        # TODO: review this
        prior = Distribution(x=x_priors, e=e_priors, y=np.ones(y_classes) / y_classes)
        x_classes = len(x_priors)
        e_classes = len(e_priors)
        u_x = np.broadcast_to(x_priors[None, None], (1, x_classes, x_priors.shape[0]))
        u_e = np.broadcast_to(e_priors[None, None], (1, e_classes, e_priors.shape[0]))
        u_y = np.ones((1, y_classes, y_classes)) / (y_classes if y_classes > 0 else 1)
        # noise_schedule = NoiseSchedule.create(0, diffusion_steps)  # 0 is cosine
        betas, alphas, alphas_bar = compute_noise_schedule(diffusion_steps)
        betas = betas[:, None, None]
        q_xs = betas * u_x + (1 - betas) * np.eye(x_classes)[None]
        q_es = (
            betas * u_e
            + (1 - betas)
            * np.eye(
                e_classes,
            )[None]
        )
        q_ys = betas * u_y + (1 - betas) * np.eye(y_classes)[None]
        qs = Q(x=q_xs, e=q_es, y=q_ys)

        alpha_bars = alphas_bar[:, None, None]
        q_bar_xs = alpha_bars * np.eye(x_classes)[None] + (1 - alpha_bars) * u_x
        q_bar_es = alpha_bars * np.eye(e_classes)[None] + (1 - alpha_bars) * u_e
        q_bar_ys = alpha_bars * np.eye(y_classes)[None] + (1 - alpha_bars) * u_y
        q_bars = Q(x=q_bar_xs, e=q_bar_es, y=q_bar_ys)
        return cls(
            y_classes=y_classes,
            diffusion_steps=diffusion_steps,
            qs=qs,
            q_bars=q_bars,
            prior=prior,
            alphas=alphas,
            betas=betas,
            alpha_bars=alpha_bars,
        )
