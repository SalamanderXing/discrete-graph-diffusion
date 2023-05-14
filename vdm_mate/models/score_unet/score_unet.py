from typing import Callable, Optional, Iterable
from jax import Array
from mate.jax import typed
from jaxtyping import Float
import chex
import flax
from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np
from flax.struct import dataclass
import ipdb


@dataclass
class ScoreUNetConfig:
    sm_n_embd: int
    sm_n_layer: int
    sm_pdrop: float
    with_attention: bool = True


######### Score model #########
class ScoreUNet(nn.Module):
    config: ScoreUNetConfig

    @typed
    @nn.compact
    def __call__(
        self,
        *,
        cond: Float[Array, "n k"],
        h: Float[Array, "b w h c_fourier"],
        z: Float[Array, "b w h 3"],
        deterministic: bool = True,
    ) -> Float[Array, "b w h 3"]:  # , z, g_t, conditioning, deterministic=True):
        config = self.config
        n_embd = config.sm_n_embd
        # Linear projection of input
        h = nn.Conv(
            features=n_embd, kernel_size=(3, 3), strides=(1, 1), name="conv_in"
        )(h)
        hs = [h]

        # Downsampling
        for i_block in range(self.config.sm_n_layer):
            block = ResnetBlock(
                sm_pdrop=config.sm_pdrop, out_ch=n_embd, name=f"down.block_{i_block}"
            )
            h = block(hs[-1], cond, deterministic)[0]
            if config.with_attention:
                h = AttnBlock(num_heads=1, name=f"down.attn_{i_block}")(h)
            hs.append(h)

        # Middle
        h = hs[-1]
        h = ResnetBlock(sm_pdrop=config.sm_pdrop, out_ch=n_embd, name="mid.block_1")(
            h, cond, deterministic
        )[0]
        h = AttnBlock(num_heads=1, name="mid.attn_1")(h)
        h = ResnetBlock(sm_pdrop=config.sm_pdrop, out_ch=n_embd, name="mid.block_2")(
            h, cond, deterministic
        )[0]

        # Upsampling
        for i_block in range(self.config.sm_n_layer + 1):
            b = ResnetBlock(
                sm_pdrop=config.sm_pdrop,
                out_ch=n_embd,
                name=f"up.block_{i_block}",
            )
            h = b(jnp.concatenate([h, hs.pop()], axis=-1), cond, deterministic)[0]
            if config.with_attention:
                h = AttnBlock(num_heads=1, name=f"up.attn_{i_block}")(h)

        assert not hs

        # Predict noise
        normalize = nn.normalization.GroupNorm()
        h = nn.swish(normalize(h))
        eps_pred = nn.Conv(
            features=z.shape[-1],
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_init=nn.initializers.zeros,
            name="conv_out",
        )(h)

        # Base measure
        eps_pred += z

        return eps_pred


######### ResNet block #########


class ResnetBlock(nn.Module):
    """Convolutional residual block with two convs."""

    # config: VDMConfig
    sm_pdrop: float
    out_ch: int | None = None

    @typed
    @nn.compact
    def __call__(
        self, x: Array, cond: Array, deterministic: bool, enc: Array | None = None
    ):
        # config = self.config

        nonlinearity = nn.swish
        normalize1 = nn.normalization.GroupNorm()
        normalize2 = nn.normalization.GroupNorm()

        if enc is not None:
            x = jnp.concatenate([x, enc], axis=-1)

        B, _, _, C = x.shape  # pylint: disable=invalid-name
        out_ch = C if self.out_ch is None else self.out_ch

        h = x
        h = nonlinearity(normalize1(h))
        h = nn.Conv(features=out_ch, kernel_size=(3, 3), strides=(1, 1), name="conv1")(
            h
        )

        # add in conditioning
        if cond is not None:
            assert cond.shape[0] == B and len(cond.shape) == 2
            h += nn.Dense(
                features=out_ch,
                use_bias=False,
                kernel_init=nn.initializers.zeros,
                name="cond_proj",
            )(cond)[:, None, None, :]

        h = nonlinearity(normalize2(h))
        h = nn.Dropout(rate=self.sm_pdrop)(h, deterministic=deterministic)
        h = nn.Conv(
            features=out_ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_init=nn.initializers.zeros,
            name="conv2",
        )(h)

        if C != out_ch:
            x = nn.Dense(features=out_ch, name="nin_shortcut")(x)

        assert x.shape == h.shape
        x = x + h
        return x, x


class AttnBlock(nn.Module):
    """Self-attention residual block."""

    num_heads: int

    @typed
    @nn.compact
    def __call__(self, x: Array):
        B, H, W, C = x.shape  # pylint: disable=invalid-name,unused-variable
        assert C % self.num_heads == 0

        normalize = nn.normalization.GroupNorm()

        h = normalize(x)
        if self.num_heads == 1:
            q = nn.Dense(features=C, name="q")(h)
            k = nn.Dense(features=C, name="k")(h)
            v = nn.Dense(features=C, name="v")(h)
            h = dot_product_attention(
                q[:, :, :, None, :],
                k[:, :, :, None, :],
                v[:, :, :, None, :],
                axis=(1, 2),
            )[:, :, :, 0, :]
            h = nn.Dense(
                features=C, kernel_init=nn.initializers.zeros, name="proj_out"
            )(h)
        else:
            head_dim = C // self.num_heads
            q = nn.DenseGeneral(features=(self.num_heads, head_dim), name="q")(h)
            k = nn.DenseGeneral(features=(self.num_heads, head_dim), name="k")(h)
            v = nn.DenseGeneral(features=(self.num_heads, head_dim), name="v")(h)
            assert q.shape == k.shape == v.shape == (B, H, W, self.num_heads, head_dim)
            h = dot_product_attention(q, k, v, axis=(1, 2))
            h = nn.DenseGeneral(
                features=C,
                axis=(-2, -1),
                kernel_init=nn.initializers.zeros,
                name="proj_out",
            )(h)

        assert h.shape == x.shape
        return x + h


def dot_product_attention(
    query,
    key,
    value,
    dtype=jnp.float32,
    bias=None,
    axis=None,
    # broadcast_dropout=True,
    # dropout_rng=None,
    # dropout_rate=0.,
    # deterministic=False,
    precision=None,
):
    """Computes dot-product attention given query, key, and value.

    This is the core function for applying attention based on
    https://arxiv.org/abs/1706.03762. It calculates the attention weights given
    query and key and combines the values using the attention weights. This
    function supports multi-dimensional inputs.


    Args:
      query: queries for calculating attention with shape of `[batch_size, dim1,
        dim2, ..., dimN, num_heads, mem_channels]`.
      key: keys for calculating attention with shape of `[batch_size, dim1, dim2,
        ..., dimN, num_heads, mem_channels]`.
      value: values to be used in attention with shape of `[batch_size, dim1,
        dim2,..., dimN, num_heads, value_channels]`.
      dtype: the dtype of the computation (default: float32)
      bias: bias for the attention weights. This can be used for incorporating
        autoregressive mask, padding mask, proximity bias.
      axis: axises over which the attention is applied.
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rng: JAX PRNGKey: to be used for dropout
      dropout_rate: dropout rate
      deterministic: bool, deterministic or not (to apply dropout)
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.

    Returns:
      Output of shape `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]`.
    """
    assert key.shape[:-1] == value.shape[:-1]
    assert query.shape[0:1] == key.shape[0:1] and query.shape[-1] == key.shape[-1]
    assert query.dtype == key.dtype == value.dtype
    input_dtype = query.dtype

    if axis is None:
        axis = tuple(range(1, key.ndim - 2))
    if not isinstance(axis, Iterable):
        axis = (axis,)
    assert key.ndim == query.ndim
    assert key.ndim == value.ndim
    for ax in axis:
        if not (query.ndim >= 3 and 1 <= ax < query.ndim - 2):
            raise ValueError(
                "Attention axis must be between the batch "
                "axis and the last-two axes."
            )
    depth = query.shape[-1]
    n = key.ndim
    # batch_dims is  <bs, <non-attention dims>, num_heads>
    batch_dims = tuple(np.delete(range(n), axis + (n - 1,)))
    # q & k -> (bs, <non-attention dims>, num_heads, <attention dims>, channels)
    qk_perm = batch_dims + axis + (n - 1,)
    key = key.transpose(qk_perm)
    query = query.transpose(qk_perm)
    # v -> (bs, <non-attention dims>, num_heads, channels, <attention dims>)
    v_perm = batch_dims + (n - 1,) + axis
    value = value.transpose(v_perm)

    key = key.astype(dtype)
    query = query.astype(dtype) / np.sqrt(depth)
    batch_dims_t = tuple(range(len(batch_dims)))
    attn_weights = jax.lax.dot_general(
        query,
        key,
        (((n - 1,), (n - 1,)), (batch_dims_t, batch_dims_t)),
        precision=precision,
    )

    # apply attention bias: masking, droput, proximity bias, ect.
    if bias is not None:
        attn_weights = attn_weights + bias

    # normalize the attention weights
    norm_dims = tuple(range(attn_weights.ndim - len(axis), attn_weights.ndim))
    attn_weights = jax.nn.softmax(attn_weights, axis=norm_dims)
    assert attn_weights.dtype == dtype
    attn_weights = attn_weights.astype(input_dtype)

    # compute the new values given the attention weights
    assert attn_weights.dtype == value.dtype
    wv_contracting_dims = (norm_dims, range(value.ndim - len(axis), value.ndim))
    y = jax.lax.dot_general(
        attn_weights,
        value,
        (wv_contracting_dims, (batch_dims_t, batch_dims_t)),
        precision=precision,
    )

    # back to (bs, dim1, dim2, ..., dimN, num_heads, channels)
    perm_inv = _invert_perm(qk_perm)
    y = y.transpose(perm_inv)
    assert y.dtype == input_dtype
    return y


@typed
def _invert_perm(perm: tuple[int | np.int64, ...]) -> tuple[int, ...]:
    perm_inv = [0] * len(perm)
    for i, j in enumerate(perm):
        perm_inv[j] = i
    return tuple(perm_inv)
