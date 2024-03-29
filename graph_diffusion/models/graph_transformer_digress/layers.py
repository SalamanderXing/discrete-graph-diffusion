import flax.linen as nn
import jax.numpy as np
import ipdb
import jax
from .config import initializers


class XToY(nn.Module):
    """Maps X to Y."""

    dy: int
    initializer: str

    @nn.compact
    def __call__(self, x):
        m = x.mean(axis=1)
        m1 = x.min(axis=1)
        ma = x.max(axis=1)
        std = x.std(axis=1)
        z = np.concatenate([m, m1, ma, std], axis=1)
        out = nn.Dense(
            features=self.dy, name="out", kernel_init=initializers[self.initializer]
        )(z)
        return out


class EToY(nn.Module):
    """Maps Y to X."""

    dy: int
    initializer: str

    @nn.compact
    def __call__(self, e):
        m = e.mean(axis=(1, 2))
        m1 = e.min(axis=2).min(axis=1)
        ma = e.max(axis=2).max(axis=1)
        std = e.std(axis=(1, 2))
        z = np.concatenate([m, m1, ma, std], axis=1)
        # prints all the shapes
        out = nn.Dense(
            features=self.dy, name="out", kernel_init=initializers[self.initializer]
        )(z)
        return out


def masked_softmax(x, mask, **kwargs):
    """Softmax with masking."""
    x_masked = np.where((mask == 0)[..., None], -1e9, x)  # x.at[mask == 0].set(-1e9)
    return nn.softmax(x_masked, **kwargs)
