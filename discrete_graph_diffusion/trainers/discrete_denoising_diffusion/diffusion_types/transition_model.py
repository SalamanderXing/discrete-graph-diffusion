import jax.numpy as np
from .q import Q
from abc import ABC, abstractmethod
from jax import Array
from jaxtyping import Float
from mate.jax import SFloat, SInt, typed
import jax_dataclasses as jdc
import ipdb
from .noise_schedule import NoiseSchedule
from .q import Q


@jdc.pytree_dataclass
class TransitionModel(jdc.EnforcedAnnotationsMixin):
    x_marginals: Float[Array, "n"]
    e_marginals: Float[Array, "m"]
    y_classes: int
    diffusion_steps: int
    qs: Q
    q_bars: Q

    @classmethod
    @typed
    def create(
        cls,
        x_marginals: Float[Array, "n"],
        e_marginals: Float[Array, "m"],
        y_classes: int,
        diffusion_steps: int,
    ) -> "TransitionModel":
        x_classes = len(x_marginals)
        e_classes = len(e_marginals)
        u_x = np.broadcast_to(
            x_marginals[None, None], (1, x_classes, x_marginals.shape[0])
        )
        u_e = np.broadcast_to(
            e_marginals[None, None], (1, e_classes, e_marginals.shape[0])
        )
        u_y = np.ones((1, y_classes, y_classes)) / (y_classes if y_classes > 0 else 1)
        noise_schedule = NoiseSchedule.create(0, diffusion_steps)  # 0 is cosine
        betas = noise_schedule.betas[:, None, None]
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

        alpha_bars = noise_schedule.alphas_bar[:, None, None]
        q_bar_xs = alpha_bars * np.eye(x_classes)[None] + (1 - alpha_bars) * u_x
        q_bar_es = alpha_bars * np.eye(e_classes)[None] + (1 - alpha_bars) * u_e
        q_bar_ys = alpha_bars * np.eye(y_classes)[None] + (1 - alpha_bars) * u_y
        q_bars = Q(x=q_bar_xs, e=q_bar_es, y=q_bar_ys)
        return cls(
            x_marginals=x_marginals,
            e_marginals=e_marginals,
            y_classes=y_classes,
            diffusion_steps=diffusion_steps,
            qs=qs,
            q_bars=q_bars,
        )

    
