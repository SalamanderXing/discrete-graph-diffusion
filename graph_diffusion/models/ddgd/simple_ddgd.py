from jax import numpy as np, Array
from tqdm import tqdm
import jax
from flax.struct import dataclass
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from jaxtyping import Float, Int
from mate.jax import Key, SBool
import ipdb
from typing import Protocol, runtime_checkable, no_type_check
from beartype import beartype
from jaxtyping import jaxtyped
from jax import jit

from .transition_model import TransitionModel
from . import diffusion_functions as df
from .extra_features.extra_features import compute as compute_extra_features
from ...shared.graph import graph_distribution as gd
from mate.jax import SInt
from .ddgd import GetProbability


@no_type_check
class SimpleDDGD(nn.Module):
    nodes_dist: Float[Array, "k"]
    nodes_prior: Float[Array, "edge_features"]
    edges_prior: Float[Array, "edge_features"]
    temporal_embedding_dim: int
    diffusion_steps: int
    n: int
    model_class: type[nn.Module]
    n_layers: int = 5
    use_extra_features: SBool = False
    noise_schedule_type: str = "cosine"
    use_extra_features: SBool = False

    def setup(self):
        self.transition_model = TransitionModel.create(
            schedule_type=self.noise_schedule_type,
            nodes_prior=self.nodes_prior,
            edge_prior=self.edges_prior,
            diffusion_steps=self.diffusion_steps,
            temporal_embedding_dim=self.temporal_embedding_dim,
            n=self.n,
        )
        self.p = GetProbability(
            model=self.model_class(n_layers=self.n_layers),
            transition_model=self.transition_model,
            use_extra_features=self.use_extra_features,
        )
        self.p_deterministic = jax.tree_util.Partial(self.p, deterministic=True)
        self.p_nondeterministic = jax.tree_util.Partial(self.p, deterministic=False)

    @jaxtyped
    @beartype
    def __call__(
        self,
        target: gd.OneHotGraph,
        rng_key: Key,
    ):
        return df.compute_train_loss(
            target=target,
            transition_model=self.transition_model,
            rng_key=rng_key,
            get_probability=self.p_nondeterministic,
        )

    @jaxtyped
    @beartype
    def compute_val_loss(
        self,
        target: gd.OneHotGraph,
        rng_key: Key,
    ):
        return df.compute_val_loss(
            target=target,
            transition_model=self.transition_model,
            rng_key=rng_key,
            p=self.p_deterministic,
            nodes_dist=self.nodes_dist,
        )

    def sample_step(self, rng, t, g):
        return df.sample_step(
            p=self.p_deterministic,
            t=t,
            g_t=g,
            transition_model=self.transition_model,
            rng=rng,
        )

    def _sample(self, sample_step: callable, rng: Key, n_samples: SInt):
        rng_prior, rng_initial = jax.random.split(rng)

        g = gd.sample_one_hot(
            self.transition_model.limit_dist.repeat(n_samples), rng_initial
        )
        for t in tqdm(list(reversed(range(1, self.diffusion_steps)))):
            rng = jax.random.fold_in(rng, t)
            t = np.array(t).repeat(n_samples)
            g = sample_step(rng, t, g)
        g_prior = gd.sample_one_hot(
            self.transition_model.limit_dist.repeat(n_samples), rng_prior
        )
        return g, g_prior

    def sample(self, params: FrozenDict, rng: Key, n_samples: SInt):
        @jit
        def sample_step(rng, t, g):
            return self.apply(params, rng, t, g, method=self.sample_step)

        return self.apply(params, sample_step, rng, n_samples, method=self._sample)

    def get_model_input(self, g: gd.GraphDistribution, t: Int[Array, "bs"]):
        return get_model_input(g, t, self.use_extra_features, self.transition_model)
