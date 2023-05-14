from typing import Callable, Optional, Iterable
from jax import Array
from mate.jax import typed
import chex
import flax
from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np
from flax.struct import dataclass
import ipdb


class EncoderDecoder(nn.Module):
    """Encoder and decoder."""

    vocab_size: int

    def __call__(self, x, g_0):
        # For initialization purposes
        h = self.encode(x)
        return self.decode(h, g_0)

    def encode(self, x):
        # This transforms x from discrete values (0, 1, ...)
        # to the domain (-1,1).
        # Rounding here just a safeguard to ensure the input is discrete
        # (although typically, x is a discrete variable such as uint8)
        x = x.round()
        return 2 * ((x + 0.5) / self.vocab_size) - 1

    def decode(self, z, g_0):
        # Logits are exact if there are no dependencies between dimensions of x
        x_vals = jnp.arange(0, self.vocab_size)[:, None]
        x_vals = jnp.repeat(x_vals, 3, 1)
        x_vals = self.encode(x_vals).transpose([1, 0])[None, None, None, :, :]
        inv_stdev = jnp.exp(-0.5 * g_0[..., None])
        logits = -0.5 * jnp.square((z[..., None] - x_vals) * inv_stdev)

        logprobs = jax.nn.log_softmax(logits)
        return logprobs

    def logprob(self, x, z, g_0):
        x = x.round().astype("int32")
        x_onehot = jax.nn.one_hot(x, self.vocab_size)
        logprobs = self.decode(z, g_0)
        logprob = jnp.sum(x_onehot * logprobs, axis=(1, 2, 3, 4))
        return logprob
