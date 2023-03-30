import flax.linen as nn
import jax.numpy as jnp


class XToY(nn.Module):
    """Maps X to Y."""

    @nn.compact
    def __call__(self, x):
        m = x.mean(axis=1)
        m1 = x.min(axis=1)[0]
        ma = x.max(axis=1)[0]
        std = x.std(axis=1)
        z = jnp.stack([m, m1, ma, std], axis=1)
        out = nn.Dense(features=1, name="out")(z)
        return out


class EToY(nn.Module):
    """Maps Y to X."""

    @nn.compact
    def __call__(self, e):
        m = e.mean(axis=(1, 2))
        m1 = e.min(axis=2)[0].min(axis=1)[0]
        ma = e.max(axis=2)[0].max(axis=1)[0]
        std = e.std(axis=(1, 2))
        z = jnp.stack([m, m1, ma, std], axis=1)
        out = nn.Dense(features=1, name="out")(z)
        return out


def masked_softmax(x, mask, **kwargs):
    """Softmax with masking."""
    if mask.sum() == 0:
        return x
    x_masked = x.at[mask == 0].set(-1e9)
    return nn.softmax(x_masked, **kwargs)
