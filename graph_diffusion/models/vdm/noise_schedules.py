from collections.abc import Callable
import chex
import flax
from flax import linen as nn
import jax
from mate.jax import typed
from jax import Array
from jax import numpy as jnp
import numpy as np
from flax.struct import dataclass
from jaxtyping import Float


class CosineSchedule(nn.Module):
    n_cosine_steps: int = 100000

    # @typed
    # def cosine_beta_schedule_discrete(
    #     self, diffusion_steps: int, s=0.008
    # ) -> Float[Array, "n"]:
    #     """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ."""
    #     steps = diffusion_steps + 2
    #     x = jnp.linspace(0, steps, steps)
    #
    #     alphas_cumprod = jnp.cos(0.5 * jnp.pi * ((x / steps) + s) / (1 + s)) ** 2
    #     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    #     alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    #     betas = 1 - alphas
    #     return betas.squeeze().astype(jnp.float32)

    @typed
    def __call__(self, t: Float[Array, "b"]) -> Float[Array, "b"]:
        # t_round = jnp.round(t * self.n_cosine_steps).astype(jnp.int32)
        # return self.betas[t_round]
        start_lr = 0
        end_lr = 1
        c_i = 0.5 * (1 + jnp.cos(t * jnp.pi))
        return end_lr + (start_lr - end_lr) * c_i


class NoiseSchedule_NNet(nn.Module):
    # config: VDMConfig
    gamma_min: float
    gamma_max: float
    n_features: int = 1024
    nonlinear: bool = True

    def setup(self):
        config = self.config

        n_out = 1
        kernel_init = nn.initializers.normal()

        init_bias = self.gamma_min
        init_scale = self.gamma_max - init_bias

        self.l1 = DenseMonotone(
            n_out,
            kernel_init=constant_init(init_scale),
            bias_init=constant_init(init_bias),
        )
        if self.nonlinear:
            self.l2 = DenseMonotone(self.n_features, kernel_init=kernel_init)
            self.l3 = DenseMonotone(n_out, kernel_init=kernel_init, use_bias=False)

    @nn.compact
    def __call__(self, t: Float[Array, "b"], det_min_max=False) -> Float[Array, "b"]:
        assert jnp.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1

        if jnp.isscalar(t) or len(t.shape) == 0:
            t = t * jnp.ones((1, 1))
        else:
            t = jnp.reshape(t, (-1, 1))

        h = self.l1(t)
        if self.nonlinear:
            _h = 2.0 * (t - 0.5)  # scale input to [-1, +1]
            _h = self.l2(_h)
            _h = 2 * (nn.sigmoid(_h) - 0.5)  # more stable than jnp.tanh(h)
            _h = self.l3(_h) / self.n_features
            h += _h

        return nn.sigmoid(jnp.squeeze(h, axis=-1))


class NoiseSchedule_FixedLinear(nn.Module):
    gamma_min: float
    gamma_max: float

    @nn.compact
    @typed
    def __call__(self, t: Float[Array, "b"]) -> Float[Array, "b"]:
        return nn.sigmoid(
            jnp.array(self.gamma_min + (self.gamma_max - self.gamma_min) * t)
        )


class NoiseSchedule_Scalar(nn.Module):
    gamma_min: float
    gamma_max: float

    def setup(self):
        init_bias = self.gamma_min
        init_scale = self.gamma_max - init_bias
        self.w = self.param("w", constant_init(init_scale), (1,))
        self.b = self.param("b", constant_init(init_bias), (1,))

    @nn.compact
    @typed
    def __call__(self, t: Float[Array, "b"]) -> Float[Array, "b"]:
        return nn.sigmoid(self.b + abs(self.w) * t)


def constant_init(value, dtype="float32"):
    def _init(key, shape, dtype=dtype):
        return value * jnp.ones(shape, dtype)

    return _init


class DenseMonotone(nn.Dense):
    """Strictly increasing Dense layer."""

    @nn.compact
    def __call__(self, inputs):
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = self.param(
            "kernel", self.kernel_init, (inputs.shape[-1], self.features)
        )
        kernel = abs(jnp.asarray(kernel, self.dtype))
        y = jax.lax.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,))
            bias = jnp.asarray(bias, self.dtype)
            y = y + bias
        return nn.sigmoid(y)
