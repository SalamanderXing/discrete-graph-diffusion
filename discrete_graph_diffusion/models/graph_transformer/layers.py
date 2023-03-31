import flax.linen as nn
import jax.numpy as np
import ipdb


class XToY(nn.Module):
    """Maps X to Y."""

    dy: int

    @nn.compact
    def __call__(self, x):
        m = x.mean(axis=1)
        m1 = x.min(axis=1)
        ma = x.max(axis=1)
        std = x.std(axis=1)
        z = np.concatenate([m, m1, ma, std], axis=1)
        out = nn.Dense(features=self.dy, name="out")(z)
        return out


class EToY(nn.Module):
    """Maps Y to X."""

    dy: int

    @nn.compact
    def __call__(self, e):
        m = e.mean(axis=(1, 2))
        m1 = e.min(axis=2).min(axis=1)
        ma = e.max(axis=2).max(axis=1)
        std = e.std(axis=(1, 2))
        z = np.concatenate([m, m1, ma, std], axis=1)
        # prints all the shapes
        out = nn.Dense(features=self.dy, name="out")(z)
        return out


def masked_softmax(x, mask, **kwargs):
    """Softmax with masking."""
    if mask.sum() == 0:
        return x
    x_masked = x.at[mask == 0].set(-1e9)
    return nn.softmax(x_masked, **kwargs)
