"""
Class defining the noise schedule for the discrete diffusion model. 
"""
# TODO: implement a **learnable noise schedule**. As in https://arxiv.org/pdf/2107.00630.pdf appendix H
import jax.numpy as np
import jax
from jax import Array
from jax import numpy as np
import ipdb
import jax_dataclasses as jdc
from mate.jax import typed, SInt
from jaxtyping import Float, Bool
from .geometric import to_dense
from .data_batch import DataBatch


@typed
def cosine_beta_schedule_discrete(timesteps: int, s=0.008):
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    return betas.squeeze()


def custom_beta_schedule_discrete(timesteps: int, average_num_nodes=50, s=0.008):
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas

    assert timesteps >= 100

    p = 4 / 5  # 1 - 1 / num_edge_classes
    num_edges = average_num_nodes * (average_num_nodes - 1) / 2

    # First 100 steps: only a few updates per graph
    updates_per_graph = 1.2
    beta_first = updates_per_graph / (p * num_edges)

    betas[betas < beta_first] = beta_first
    return np.array(betas)


@jdc.pytree_dataclass
class NoiseSchedule(jdc.EnforcedAnnotationsMixin):
    name: str
    diffusion_steps: int
    betas: Float[Array, "n"]
    alphas: Float[Array, "n"]
    alphas_bar: Float[Array, "n"]

    @classmethod
    @typed
    def create(cls, schedule_index: int, diffusion_steps: int) -> "NoiseSchedule":
        schedule_names = ["cosine", "custom"]
        name = schedule_names[schedule_index]
        if name == "cosine":
            betas = cosine_beta_schedule_discrete(diffusion_steps)
        elif name == "custom":
            betas = custom_beta_schedule_discrete(diffusion_steps)
        else:
            raise NotImplementedError(name)

        betas = np.array(betas, dtype=np.float32)
        alphas = 1 - np.clip(betas, a_min=0, a_max=0.9999)

        log_alpha = np.log(alphas)
        log_alpha_bar = np.cumsum(log_alpha, axis=0)
        alphas_bar = np.exp(log_alpha_bar)

        return cls(name, diffusion_steps, betas, alphas, alphas_bar)


if __name__ == "__main__":
    jax.config.update("jax_platform_name", "cpu")  # run on CPU for now.
    import matplotlib.pyplot as plt

    noise_schedule = NoiseSchedule("cosine", 1000)
    plt.plot(noise_schedule.betas)
    plt.show()
