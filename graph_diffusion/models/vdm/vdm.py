from typing import Callable, Optional, Iterable, Any

from jax import Array
import flax
import ipdb
from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np
from flax.struct import dataclass
from .timesteps_embedding import get_timestep_embedding
from .fourier_features import Base2FourierFeatures
from mate.jax import Key, typed, SFloat
from flax.core.frozen_dict import FrozenDict
from jaxtyping import Float, Int, Bool
from mate.types import Interface
from .noise_schedules import (
    NoiseSchedule_NNet,
    NoiseSchedule_FixedLinear,
    NoiseSchedule_Scalar,
)
from ...shared.graph import Nodes, Edges, Graph


class EncoderDecoder(nn.Module, metaclass=Interface):
    """Encoder and decoder."""

    # config: VDMConfig

    def __call__(self, x, g_0):
        # For initialization purposes
        ...

    def encode(self, x: Any) -> Any:
        # # This transforms x from discrete values (0, 1, ...)
        # # to the domain (-1,1).
        # # Rounding here just a safeguard to ensure the input is discrete
        # # (although typically, x is a discrete variable such as uint8)
        # x = x.round()
        # return 2 * ((x + 0.5) / self.config.vocab_size) - 1
        ...

    def decode(self, z: Any, g_0: Array) -> Any:
        # config = self.config
        #
        # # Logits are exact if there are no dependencies between dimensions of x
        # x_vals = jnp.arange(0, config.vocab_size)[:, None]
        # x_vals = jnp.repeat(x_vals, 3, 1)
        # x_vals = self.encode(x_vals).transpose([1, 0])[None, None, None, :, :]
        # inv_stdev = jnp.exp(-0.5 * g_0[..., None])
        # logits = -0.5 * jnp.square((z[..., None] - x_vals) * inv_stdev)
        #
        # logprobs = jax.nn.log_softmax(logits)
        # return logprobs
        ...

    def logprob(self, x: Any, z: Any, g_0: Array) -> Any:
        # x = x.round().astype("int32")
        # x_onehot = jax.nn.one_hot(x, self.config.vocab_size)
        # logprobs = self.decode(z, g_0)
        # logprob = jnp.sum(x_onehot * logprobs, axis=(1, 2, 3, 4))
        # return logprob
        ...


@dataclass
class VDMConfig:
    """VDM configurations."""

    vocab_size: int
    sample_softmax: bool
    antithetic_time_sampling: bool
    with_fourier_features: bool
    with_attention: bool

    # configurations of the noise schedule
    gamma_type: str
    gamma_min: float
    gamma_max: float

    # configurations of the score model
    sm_n_timesteps: int
    num_time_embedding: int
    sm_n_layer: int
    sm_pdrop: float
    sm_kernel_init: Callable = jax.nn.initializers.normal(0.02)


class Conditioner(nn.Module):
    gamma_min: float
    # with_fourier_features: bool
    gamma_max: float
    num_time_embedding: int
    model: nn.Module

    @typed
    @nn.compact
    def __call__(self, z: Graph, g_t: Float[Array, "batch_size"], deterministic: bool):
        # Compute conditioning vector based on 'g_t' and 'conditioning'
        n_embd = self.num_time_embedding

        lb = self.gamma_min
        ub = self.gamma_max
        t = (g_t - lb) / (ub - lb)  # ---> [0,1]
        assert jnp.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
        if jnp.isscalar(t):
            t = jnp.ones((z.shape[0],), z.dtype) * t
        elif len(t.shape) == 0:
            t = jnp.tile(t[None], z.shape[0])

        timestep_embedding = get_timestep_embedding(t, n_embd)
        cond = timestep_embedding  # jnp.concatenate([timestep_embedding, conditioning[:, None]], axis=1)
        cond = nn.swish(nn.Dense(features=n_embd, name="dense0")(cond))
        cond = nn.tanh(nn.Dense(features=n_embd, name="dense1")(cond))[:, None].repeat(
            z.nodes.shape[1], axis=1
        )
        # Concatenate Fourier features to input
        # if self.with_fourier_features:
        #     z_f = Base2FourierFeatures(start=6, stop=8, step=1)(z)
        #     h = jnp.concatenate([z, z_f], axis=-1)
        # else:
        #     h = z
        z = Graph.create(
            nodes=jnp.concatenate([z.nodes, cond], axis=-1),
            edges=z.edges,
            nodes_counts=z.nodes_counts,
            edges_counts=z.edges_counts,
        )
        z_pred = self.model(z, deterministic=deterministic)
        return z_pred


######### Latent VDM model #########


@dataclass
class VDMOutput:
    loss_recon: Float[Array, "b"]  # [B]
    loss_klz: Float[Array, "b"]  # [B]
    loss_diff: Float[Array, "b"]  # [B]
    var_0: float
    var_1: float


class VDM(nn.Module):
    config: VDMConfig
    probability_model: nn.Module
    encoder_decoder: EncoderDecoder

    @classmethod
    @typed
    def create(
        cls,
        config: VDMConfig,
        example_input: Graph,
        probability_model: nn.Module,
        encoder_decoder: EncoderDecoder,
        rng: Key,
    ) -> tuple[nn.Module, FrozenDict]:
        # config = self.config
        # config = vdm.model_vdm.VDMConfig(**config.model)
        model = cls(
            config=config,
            probability_model=probability_model,
            encoder_decoder=encoder_decoder,
        )
        # inputs = {
        #     "images": jnp.zeros((2, 32, 32, 3), "uint8"),
        #     "conditioning": jnp.zeros((2,)),
        # }
        rng1, rng2 = jax.random.split(rng)
        rng3, rng4 = jax.random.split(rng2)
        params = model.init(
            {"params": rng1, "sample": rng2, "dropout": rng4}, example_input
        )
        return model, params

    def setup(self):
        self.score_model = Conditioner(
            gamma_min=self.config.gamma_min,
            gamma_max=self.config.gamma_max,
            # with_fourier_features=self.config.with_fourier_features,
            model=self.probability_model,
            num_time_embedding=self.config.num_time_embedding,
        )
        # self.encoder_decoder = EncDec(self.config)
        if self.config.gamma_type == "learnable_nnet":
            self.gamma = NoiseSchedule_NNet(
                gamma_max=self.config.gamma_max, gamma_min=self.config.gamma_min
            )
        elif self.config.gamma_type == "fixed":
            self.gamma = NoiseSchedule_FixedLinear(
                gamma_max=self.config.gamma_max, gamma_min=self.config.gamma_min
            )
        elif self.config.gamma_type == "learnable_scalar":
            self.gamma = NoiseSchedule_Scalar(
                gamma_max=self.config.gamma_max, gamma_min=self.config.gamma_min
            )
        else:
            raise Exception("Unknown self.var_model")

    @typed
    def __call__(self, x: Graph, deterministic: bool = True):
        g_0, g_1 = self.gamma(0.0), self.gamma(1.0)
        var_0, var_1 = nn.sigmoid(g_0), nn.sigmoid(g_1)
        n_batch = x.nodes.shape[0]

        # encode
        f = self.encoder_decoder.encode(x)

        sample_rng = self.make_rng("sample")
        # 1. RECONSTRUCTION LOSS
        # add noise and reconstruct
        eps_0 = f.noise_like(sample_rng)
        # z_0 = (
        #     jnp.sqrt(1.0 - var_0) * f + jnp.sqrt(var_0) * eps_0
        # )  # FIXME why is this not accessed??
        z_0_rescaled = f + (
            jnp.exp(0.5 * g_0) * eps_0
        )  # = z_0/sqrt(1-var) # FIXME remove that 2!! just for testing
        loss_recon = -self.encoder_decoder.logprob(x, z_0_rescaled, g_0)

        # 2. LATENT LOSS
        # KL z1 with N(0,1) prior
        mean1_sqr = (1.0 - var_1) * (f**2)
        loss_klz = 0.5 * (mean1_sqr + var_1 - jnp.log(var_1) - 1.0).sum()

        # 3. DIFFUSION LOSS
        # sample time steps
        rng1 = self.make_rng("sample")
        if self.config.antithetic_time_sampling:
            t0 = jax.random.uniform(rng1)
            t = jnp.mod(t0 + jnp.arange(0.0, 1.0, step=1.0 / n_batch), 1.0)
        else:
            t = jax.random.uniform(rng1, shape=(n_batch,))

        # discretize time steps if we're working with discrete time
        T = self.config.sm_n_timesteps
        if T > 0:
            t = jnp.ceil(t * T) / T

        # sample z_t
        g_t = self.gamma(t)
        var_t = nn.sigmoid(g_t)[:, None, None]
        eps = f.noise_like(self.make_rng("sample"))
        z_t = jnp.sqrt(1.0 - var_t) * f + jnp.sqrt(var_t) * eps
        # compute predicted noise
        eps_hat = self.score_model(z_t, g_t, deterministic)
        # compute MSE of predicted noise
        loss_diff_mse = ((eps - eps_hat) ** 2).sum()
        if T == 0:
            # loss for infinite depth T, i.e. continuous time
            _, g_t_grad = jax.jvp(self.gamma, (t,), (jnp.ones_like(t),))
            loss_diff = 0.5 * g_t_grad * loss_diff_mse
        else:
            # loss for finite depth T, i.e. discrete time
            s = t - (1.0 / T)
            g_s = self.gamma(s)
            loss_diff = 0.5 * T * jnp.expm1(g_t - g_s) * loss_diff_mse

        # End of diffusion loss computation

        return VDMOutput(
            loss_recon=loss_recon,
            loss_klz=loss_klz,
            loss_diff=loss_diff,
            var_0=var_0,
            var_1=var_1,
        )

    def sample(self, i, T, z_t, conditioning, rng):
        rng_body = jax.random.fold_in(rng, i)
        eps = jax.random.normal(rng_body, z_t.shape)

        t = (T - i) / T
        s = (T - i - 1) / T

        g_s, g_t = self.gamma(s), self.gamma(t)
        eps_hat = self.score_model(
            z_t,
            g_t * jnp.ones((z_t.shape[0],), g_t.dtype),
            conditioning,
            deterministic=True,
        )
        a = nn.sigmoid(-g_s)
        b = nn.sigmoid(-g_t)
        c = -jnp.expm1(g_s - g_t)
        sigma_t = jnp.sqrt(nn.sigmoid(g_t))

        z_s = (
            jnp.sqrt(nn.sigmoid(-g_s) / nn.sigmoid(-g_t))
            * (z_t - sigma_t * c * eps_hat)
            + jnp.sqrt((1.0 - a) * c) * eps
        )

        return z_s

    def generate_x(self, z_0):
        g_0 = self.gamma(0.0)

        var_0 = nn.sigmoid(g_0)
        z_0_rescaled = z_0 / jnp.sqrt(1.0 - var_0)

        logits = self.encoder_decoder.decode(z_0_rescaled, g_0)

        # get output samples
        if self.config.sample_softmax:
            out_rng = self.make_rng("sample")
            samples = jax.random.categorical(out_rng, logits)
        else:
            samples = jnp.argmax(logits, axis=-1)

        return samples
