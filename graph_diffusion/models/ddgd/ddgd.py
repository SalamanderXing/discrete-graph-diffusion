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


# @dataclass
# class GetProbability:
#     state: TrainState
#     dropout_rng: Key
#     transition_model: TransitionModel
#     use_extra_features: SBool = False
#     deterministic: SBool = False
#
#     def __call__(
#         self, g: gd.OneHotGraph, t: Int[Array, "batch_size"]
#     ) -> gd.DenseGraphDistribution:
#         g_input, temporal_embeddings = get_model_input(
#             g, t, self.use_extra_features, self.transition_model
#         )
#         pred_graph = self.state.apply_fn(
#             self.state.params,
#             g_input,
#             temporal_embeddings,
#             deterministic=True,
#             rngs={"dropout": self.dropout_rng},
#         )
#         return pred_graph
#
#
# @dataclass
# class FeatureGetProbability:
#     state: TrainState
#     dropout_rng: Key
#     structure: gd.StructureOneHotGraph
#     transition_model: TransitionModel
#     use_extra_features: SBool = False
#     deterministic: SBool = False
#
#     def __call__(
#         self, g: gd.OneHotGraph, t: Int[Array, "batch_size"]
#     ) -> gd.DenseGraphDistribution:
#         g_input = gd.FeatureOneHotGraph.create(g).restore_structure(self.structure)
#         g_input, temporal_embeddings = get_model_input(
#             g_input, t, self.use_extra_features, self.transition_model
#         )
#         pred_graph = self.state.apply_fn(
#             self.state.params,
#             g_input,
#             temporal_embeddings,
#             deterministic=True,
#             rngs={"dropout": self.dropout_rng},
#         )
#         _, pred_graph_same_structure = (
#             gd.FeatureOneHotGraph.create(pred_graph)
#             .apply_structure(self.structure)
#             .decompose_structure_and_feature()
#         )
#         return gd.DenseGraphDistribution.create(
#             nodes=pred_graph_same_structure.nodes,
#             edges=pred_graph_same_structure.edges,
#             nodes_mask=pred_graph_same_structure.nodes_mask,
#             edges_mask=pred_graph_same_structure.edges_mask,
#         )
#
#
# @dataclass
# class StructureGetProbability:
#     state: TrainState
#     dropout_rng: Key
#     transition_model: TransitionModel
#     node_feature_count: int
#     edge_feature_count: int
#     use_extra_features: SBool = False
#     deterministic: SBool = False
#
#     def __call__(
#         self, g: gd.OneHotGraph, t: Int[Array, "batch_size"]
#     ) -> gd.DenseGraphDistribution:
#         g_structure = gd.StructureOneHotGraph.create(g)
#         g_input = gd.one_hot_structure_to_dense(
#             g_structure,
#             node_feature_count=self.node_feature_count,
#             edge_feature_count=self.edge_feature_count,
#         )
#         g_input, temporal_embeddings = get_model_input(
#             g_input, t, self.use_extra_features, self.transition_model
#         )
#         pred_graph = self.state.apply_fn(
#             self.state.params,
#             g_input,
#             temporal_embeddings,
#             deterministic=True,
#             rngs={"dropout": self.dropout_rng},
#         )
#         pred_graph_structure_only = gd.dense_to_structure_dense(pred_graph)
#         return pred_graph_structure_only
#
#
# # sets the parameter use_extra_features to be partially applied to the function
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


#
#
#
# def _compute_train_loss(
#     transition_model: TransitionModel,
#     target: gd.OneHotGraph,
#     rng_key: Key,
#     params: FrozenDict,
#     state: TrainState,
#     dropout_rng: Key,
#     use_extra_features: SBool,
# ):
#


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
        return pred_graph


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

    def sample(self, sample_step, rng, n_samples):
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

    def get_model_input(self, g: gd.GraphDistribution, t: Int[Array, "bs"]):
        return get_model_input(g, t, self.use_extra_features, self.transition_model)


class GetProbabilityFeature(nn.Module):
    transition_model: TransitionModel
    model: nn.Module
    use_extra_features: SBool = False

    def __call__(
        self,
        g: gd.OneHotGraph,
        t: Int[Array, "batch_size"],
        structure: gd.OneHotGraph,
        deterministic: SBool = False,
    ) -> gd.DenseGraphDistribution:
        nodes_with_structure = np.where(
            structure.nodes.argmax(-1), g.nodes.argmax(-1) + 1, 0
        )
        edges_with_structure = np.where(
            structure.edges.argmax(-1), g.edges.argmax(-1) + 1, 0
        )
        nodes_with_structure_one_hot = jax.nn.one_hot(
            nodes_with_structure, g.nodes.shape[-1]
        )
        edges_with_structure_one_hot = jax.nn.one_hot(
            edges_with_structure, g.edges.shape[-1]
        )
        g_with_structure = gd.OneHotGraph.create(
            nodes=nodes_with_structure_one_hot,
            edges=edges_with_structure_one_hot,
            nodes_mask=structure.nodes_mask,
            edges_mask=structure.edges_mask,
        )
        g_input, temporal_embeddings = get_model_input(
            g_with_structure, t, self.use_extra_features, self.transition_model
        )
        pred_graph = self.model(
            g_input, temporal_embeddings, deterministic=deterministic
        )
        _, pred_features = pred_graph.decompose_structure_and_feature()
        pred_features_dense = gd.DenseGraphDistribution.create(
            nodes=pred_features.nodes,
            edges=pred_features.edges,
            nodes_mask=g.nodes_mask,
            edges_mask=g.edges_mask,
        )
        return pred_features_dense


@no_type_check
class StructureFirstDDGD(nn.Module):
    nodes_dist: Float[Array, "nodes_features"]
    feature_nodes_prior: Float[Array, "edge_features"]
    feature_edges_prior: Float[Array, "edge_features"]
    structure_nodes_prior: Float[Array, "2"]
    structure_edges_prior: Float[Array, "2"]
    temporal_embedding_dim: int
    structure_diffusion_steps: int
    feature_diffusion_steps: int
    n: int
    n_layers: int
    model_class: type[nn.Module]
    use_extra_features: SBool = False
    noise_schedule_type: str = "cosine"
    use_extra_features: SBool = False

    def setup(self):
        feature_edges_prior = (
            self.feature_edges_prior[1:] / self.feature_edges_prior[1:].sum()
        )
        self.feature_transition_model = TransitionModel.create(
            schedule_type=self.noise_schedule_type,
            nodes_prior=self.feature_nodes_prior,
            edge_prior=feature_edges_prior,
            diffusion_steps=self.feature_diffusion_steps,
            temporal_embedding_dim=self.temporal_embedding_dim,
            n=self.n,
        )
        self.structure_transition_model = TransitionModel.create(
            schedule_type=self.noise_schedule_type,
            nodes_prior=self.structure_nodes_prior,
            edge_prior=self.structure_edges_prior,
            diffusion_steps=self.structure_diffusion_steps,
            temporal_embedding_dim=self.temporal_embedding_dim,
            n=self.n,
        )
        self.p_structure = GetProbability(
            transition_model=self.structure_transition_model,
            model=self.model_class(n_layers=self.n_layers),
            use_extra_features=self.use_extra_features,
        )
        self.p_feature = GetProbabilityFeature(
            transition_model=self.feature_transition_model,
            model=self.model_class(n_layers=self.n_layers),
            use_extra_features=self.use_extra_features,
        )
        self.p_structure_deterministic = jax.tree_util.Partial(
            self.p_structure, deterministic=True
        )
        self.p_structure_nondeterministic = jax.tree_util.Partial(
            self.p_structure, deterministic=False
        )

    @jaxtyped
    @beartype
    def compute_val_loss(
        self,
        target: gd.OneHotGraph,
        rng_key: Key,
    ):
        """
        Val loss in this case is the mean of the
        structure only diffusion loss and the feature only diffusion loss
        """

        target_structure, target_feature = target.decompose_structure_and_feature()
        p_feature_deterministic = jax.tree_util.Partial(
            self.p_feature, structure=target_structure, deterministic=True
        )
        # structure_loss = df.compute_val_loss(
        #     target=target_structure,
        #     transition_model=self.structure_transition_model,
        #     rng_key=rng_key,
        #     p=self.p_structure_deterministic,
        #     nodes_dist=self.nodes_dist,
        # )
        feature_loss = df.compute_val_loss(
            target=target_feature,
            transition_model=self.feature_transition_model,
            rng_key=rng_key,
            p=p_feature_deterministic,
            nodes_dist=self.nodes_dist,
        )
        # merged = {
        #     key: np.array([structure_loss[key], feature_loss[key]]).mean(-1)
        #     for key in structure_loss.keys()
        # }
        # return merged
        return feature_loss

    @jaxtyped
    @beartype
    def __call__(
        self,
        target: gd.OneHotGraph,
        rng_key: Key,
    ):
        target_structure, target_feature = target.decompose_structure_and_feature()
        p_feature_nondeterministic = jax.tree_util.Partial(
            self.p_feature, structure=target_structure, deterministic=False
        )
        # loss1 = df.compute_train_loss(
        #     target=target_structure,
        #     transition_model=self.structure_transition_model,
        #     rng_key=rng_key,
        #     get_probability=self.p_structure_nondeterministic,
        # )
        loss2 = df.compute_train_loss(
            target=target_feature,
            transition_model=self.feature_transition_model,
            rng_key=rng_key,
            get_probability=p_feature_nondeterministic,
        )
        # return loss1 + loss2
        return loss2

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
        return get_model_input(
            g, t, self.use_extra_features, self.structure_transition_model
        )
