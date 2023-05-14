from jax import Array
import flax
from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np
from flax.struct import dataclass
from .timesteps_embedding import get_timestep_embedding


class Base2FourierFeatures(nn.Module):
    start: int = 0
    stop: int = 8
    step: int = 1

    @nn.compact
    def __call__(self, inputs):
        freqs = jnp.arange(self.start, self.stop, self.step, dtype=inputs.dtype)

        # Create Base 2 Fourier features
        w = 2.0**freqs * 2 * jnp.pi
        w = jnp.tile(w[None, :], (1, inputs.shape[-1]))

        # Compute features
        h = jnp.repeat(inputs, len(freqs), axis=-1)
        h = w * h
        h = jnp.concatenate([jnp.sin(h), jnp.cos(h)], axis=-1)
        return h
