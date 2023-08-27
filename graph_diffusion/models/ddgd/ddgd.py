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



# # sets the parameter use_extra_features to be partially applied to the function
def get_model_input(
    g: gd.GraphDistribution,
    t: Int[Array, "bs"],
    use_extra_features: SBool,
    transition_model: TransitionModel,
):
    if use_extra_features:
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



@runtime_checkable
class DDGD(Protocol):
    def compute_val_loss(
        self,
        target: gd.OneHotGraph,
        rng_key: Key,
        state: TrainState,
    ) -> Float[Array, "batch_size"]:
        pass

    def compute_train_loss(
        self,
        target: gd.OneHotGraph,
        rng_key: Key,
        params: FrozenDict,
        state: TrainState,
        dropout_rng: Key,
    ) -> Float[Array, "batch_size"]:
        pass

    def sample(self, rng, p, n_samples=1):
        pass

    def get_model_input(self, g: gd.GraphDistribution, t: Int[Array, "bs"]):
        pass


class GetProbability(nn.Module):
    transition_model: TransitionModel
    model: nn.Module
    use_extra_features: SBool = False

    def __call__(
        self,
        g: gd.OneHotGraph,
        t: Int[Array, "batch_size"],
        deterministic: SBool = False,
    ) -> gd.DenseGraphDistribution:
        g_input, temporal_embeddings = get_model_input(
            g, t, self.use_extra_features, self.transition_model
        )
        pred_graph = self.model(
            g_input, temporal_embeddings, deterministic=deterministic
        )
        pred_graph = gd.DenseGraphDistribution.create(
            nodes=pred_graph.nodes[:, :, : self.transition_model.qs.nodes.shape[-1]],
            edges=pred_graph.edges,
            nodes_mask=pred_graph.nodes_mask,
            edges_mask=pred_graph.edges_mask,
            unsafe=True,
        )
        return pred_graph






