from jax import numpy as np, Array
from tqdm import tqdm
import jax
from flax.struct import dataclass
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState
from jaxtyping import Float, Int
from mate.jax import Key, SBool
from abc import ABC, abstractmethod
from .transition_model import TransitionModel
from . import diffusion_functions as df
from .extra_features.extra_features import compute as compute_extra_features
from ...shared.graph import graph_distribution as gd


@dataclass
class GetProbabilityFromState:
    state: TrainState
    dropout_rng: Key
    transition_model: TransitionModel
    use_extra_features: SBool = False
    deterministic: SBool = False

    def __call__(
        self, g: gd.OneHotGraph, t: Int[Array, "batch_size"]
    ) -> gd.DenseGraphDistribution:
        g_input, temporal_embeddings = get_model_input(
            g, t, self.use_extra_features, self.transition_model
        )
        pred_graph = self.state.apply_fn(
            self.state.params,
            g_input,
            temporal_embeddings,
            deterministic=True,
            rngs={"dropout": self.dropout_rng},
        )
        return pred_graph


from functools import partial
from jax import jit


# sets the parameter use_extra_features to be partially applied to the function
def get_model_input(
    g: gd.GraphDistribution,
    t: Int[Array, "bs"],
    use_extra_features: SBool,
    transition_model: TransitionModel,
):
    if False:
        extra_features = compute_extra_features(g, g.nodes.shape[1])
        g = gd.GraphDistribution.create(
            nodes=np.concatenate([g.nodes, extra_features.nodes], axis=-1),
            edges=np.concatenate([g.edges, extra_features.edges], axis=-1),
            nodes_mask=g.nodes_mask,
            edges_mask=g.edges_mask,
        )
        temporal_embeddings = extra_features.y
    else:
        temporal_embeddings = transition_model.temporal_embeddings[t]
    return g, temporal_embeddings


class DDGD(ABC):
    @abstractmethod
    def compute_val_loss(
        self,
        target: gd.OneHotGraph,
        rng_key: Key,
        state: TrainState,
    ) -> Float[Array, "batch_size"]:
        pass

    @abstractmethod
    def compute_train_loss(
        self,
        target: gd.OneHotGraph,
        rng_key: Key,
        params: FrozenDict,
        state: TrainState,
        dropout_rng: Key,
    ) -> Float[Array, "batch_size"]:
        pass

    @abstractmethod
    def sample(self, rng, p, n_samples=1):
        pass

    @abstractmethod
    def get_model_input(self, g: gd.GraphDistribution, t: Int[Array, "bs"]):
        pass


@dataclass
class SimpleDDGD(DDGD):
    transition_model: TransitionModel
    nodes_dist: Float[Array, "k"]
    use_extra_features: SBool = False

    @classmethod
    def create(
        cls,
        *,
        nodes_dist: Float[Array, "k"],
        nodes_prior: Float[Array, "n"],
        edges_prior: Float[Array, "m"],
        diffusion_steps: int,
        temporal_embedding_dim: int,
        n: int,
        noise_schedule_type: str = "cosine",
        use_extra_features: SBool = False,
    ):
        transition_model = TransitionModel.create(
            schedule_type=noise_schedule_type,
            nodes_prior=nodes_prior,
            edge_prior=edges_prior,
            diffusion_steps=diffusion_steps,
            temporal_embedding_dim=temporal_embedding_dim,
            n=n,
        )
        return cls(
            transition_model=transition_model,
            nodes_dist=nodes_dist,
            use_extra_features=use_extra_features,
        )

    def compute_val_loss(
        self,
        target: gd.OneHotGraph,
        rng_key: Key,
        state: TrainState,
    ):
        p = GetProbabilityFromState(
            state=state,
            dropout_rng=jax.random.PRNGKey(0),  # dummy, cause it wont be used
            transition_model=self.transition_model,
            use_extra_features=self.use_extra_features,
            deterministic=True,
        )
        return df.compute_val_loss(
            target=target,
            transition_model=self.transition_model,
            rng_key=rng_key,
            p=p,
            nodes_dist=self.nodes_dist,
        )

    def compute_train_loss(
        self,
        target: gd.OneHotGraph,
        rng_key: Key,
        params: FrozenDict,
        state: TrainState,
        dropout_rng: Key,
    ):
        p = GetProbabilityFromState(
            state=state.replace(params=params),
            dropout_rng=dropout_rng,
            transition_model=self.transition_model,
            use_extra_features=self.use_extra_features,
            deterministic=False,
        )
        return df.compute_train_loss(
            target=target,
            transition_model=self.transition_model,
            rng_key=rng_key,
            get_probability=p,
        )

    def sample(self, rng, p, n_samples=1):
        _, rng_this_epoch = jax.random.split(rng)
        g = gd.sample_one_hot(
            self.transition_model.limit_dist.repeat(n_samples), rng_this_epoch
        )
        for t in tqdm(list(reversed(range(1, self.transition_model.diffusion_steps)))):
            rng = jax.random.fold_in(rng, df.enc(f"sample_{t}"))
            t = np.array(t).repeat(n_samples)
            g = df.sample_step(
                p=p,
                t=t,
                g_t=g,
                transition_model=self.transition_model,
                rng=rng,
            )
        return g

    def get_model_input(self, g: gd.GraphDistribution, t: Int[Array, "bs"]):
        return get_model_input(g, t, self.use_extra_features, self.transition_model)


@dataclass
class StructureFirstDDGD(DDGD):
    structure_transition_model: TransitionModel
    features_transition_model: TransitionModel
    nodes_dist: Float[Array, "k"]
    use_extra_features: SBool = False

    @classmethod
    def create(
        cls,
        *,
        nodes_dist: Float[Array, "node_features"],
        feature_nodes_prior: Float[Array, "edge_features"],
        feature_edges_prior: Float[Array, "edge_features"],
        structure_nodes_prior: Float[Array, "2"],
        structure_edges_prior: Float[Array, "2"],
        temporal_embedding_dim: int,
        n: int,
        noise_schedule_type: str = "cosine",
        use_extra_features: SBool = False,
        structure_diffusion_steps: int,
        feature_diffusion_steps: int,
    ):
        feature_transition_model = TransitionModel.create(
            schedule_type=noise_schedule_type,
            nodes_prior=feature_nodes_prior,
            edge_prior=feature_edges_prior,
            diffusion_steps=feature_diffusion_steps,
            temporal_embedding_dim=temporal_embedding_dim,
            n=n,
        )
        structure_transition_model = TransitionModel.create(
            schedule_type=noise_schedule_type,
            nodes_prior=structure_nodes_prior,
            edge_prior=structure_edges_prior,
            diffusion_steps=structure_diffusion_steps,
            temporal_embedding_dim=temporal_embedding_dim,
            n=n,
        )
        return cls(
            features_transition_model=feature_transition_model,
            structure_transition_model=structure_transition_model,
            nodes_dist=nodes_dist,
            use_extra_features=use_extra_features,
        )

    def compute_val_loss(
        self,
        target: gd.OneHotGraph,
        rng_key: Key,
        state: TrainState,
    ):
        p = GetProbabilityFromState(
            state=state,
            dropout_rng=jax.random.PRNGKey(0),  # dummy, cause it wont be used
            transition_model=self.transition_model,
            use_extra_features=self.use_extra_features,
            deterministic=True,
        )
        return df.compute_val_loss(
            target=target,
            transition_model=self.transition_model,
            rng_key=rng_key,
            p=p,
            nodes_dist=self.nodes_dist,
        )

    def compute_train_loss(
        self,
        target: gd.OneHotGraph,
        rng_key: Key,
        params: FrozenDict,
        state: TrainState,
        dropout_rng: Key,
    ):
        p = GetProbabilityFromState(
            state=state.replace(params=params),
            dropout_rng=dropout_rng,
            transition_model=self.transition_model,
            use_extra_features=self.use_extra_features,
            deterministic=False,
        )
        return df.compute_train_loss(
            target=target,
            transition_model=self.transition_model,
            rng_key=rng_key,
            get_probability=p,
        )

    def sample(self, rng, p, n_samples=1):
        _, rng_this_epoch = jax.random.split(rng)
        g = gd.sample_one_hot(
            self.transition_model.limit_dist.repeat(n_samples), rng_this_epoch
        )
        for t in tqdm(list(reversed(range(1, self.transition_model.diffusion_steps)))):
            rng = jax.random.fold_in(rng, df.enc(f"sample_{t}"))
            t = np.array(t).repeat(n_samples)
            g = df.sample_step(
                p=p,
                t=t,
                g_t=g,
                transition_model=self.transition_model,
                rng=rng,
            )
        return g

    def get_model_input(self, g: gd.GraphDistribution, t: Int[Array, "bs"]):
        return get_model_input(g, t, self.use_extra_features, self.transition_model)

