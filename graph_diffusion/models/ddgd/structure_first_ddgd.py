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
from .display import ValidationResultWrapper
from . import diffusion_functions as df
from .extra_features.extra_features import compute as compute_extra_features
from ...shared.graph import graph_distribution as gd
from .ddgd import GetProbability, get_model_input
from mate.jax import SInt

# from typing import Callable


class GetProbabilityFeature(nn.Module):
    transition_model: TransitionModel
    model: nn.Module
    use_extra_features: SBool = False

    def __call__(
        self,
        g: gd.OneHotGraph,
        t: Int[Array, "batch_size"],
        structure: gd.OneHotGraph,
        feature: gd.OneHotGraph,
        deterministic: SBool = False,
    ) -> gd.DenseGraphDistribution:
        g_with_structure = structure.feature_like(g)
        g_input, temporal_embeddings = get_model_input(
            g_with_structure, t, self.use_extra_features, self.transition_model
        )
        pred_graph = self.model(
            g_input, temporal_embeddings, deterministic=deterministic
        )
        pred_features = pred_graph.feature(unsafe=True)
        graph_with_pred_feature = structure.feature_like(pred_features, unsafe=True)
        pred_features_dense = gd.DenseGraphDistribution.to_dense(
            graph_with_pred_feature.feature(unsafe=True), unsafe=True
        )
        # pred_features_dense = gd.DenseGraphDistribution.to_dense(
        #     feature, unsafe=True
        # ).scalar_multiply(100, unsafe=True)
        return pred_features_dense


from typing import Callable


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
    use_feature: SBool = True  # useful for debugging
    use_structure: SBool = False  # useful for debugging

    def setup(self):
        feature_edges_prior = (
            self.feature_edges_prior[1:] / self.feature_edges_prior[1:].sum()
        )
        self.structure_transition_model = TransitionModel.create(
            schedule_type=self.noise_schedule_type,
            nodes_prior=np.array([1.0]),
            edge_prior=self.structure_edges_prior,
            diffusion_steps=self.structure_diffusion_steps,
            temporal_embedding_dim=self.temporal_embedding_dim,
            n=self.n,
        )
        # assert np.sum(np.array(self.structure_transition_model.qs.nodes.shape[1:])) == 2
        self.feature_transition_model = TransitionModel.create(
            schedule_type=self.noise_schedule_type,
            nodes_prior=self.feature_nodes_prior,
            edge_prior=feature_edges_prior,
            diffusion_steps=self.feature_diffusion_steps,
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
    ) -> ValidationResultWrapper:
        """
        Val loss in this case is the mean of the
        structure only diffusion loss and the feature only diffusion loss
        """

        target_structure, target_feature = target.decompose_structure_and_feature()

        losses = {}
        if self.use_structure:
            structure_loss = df.compute_val_loss(
                target=target_structure,
                transition_model=self.structure_transition_model,
                rng_key=rng_key,
                p=self.p_structure_deterministic,
                nodes_dist=self.nodes_dist,
            )
            losses["structure"] = structure_loss
        if self.use_feature:
            p_feature_deterministic = jax.tree_util.Partial(
                self.p_feature,
                structure=target_structure,
                feature=target_feature,  # FIXME: just for debugging
                deterministic=True,
            )
            feature_loss = df.compute_val_loss(
                target=target_feature,
                transition_model=self.feature_transition_model,
                rng_key=rng_key,
                p=p_feature_deterministic,
                nodes_dist=self.nodes_dist,
            )
            losses["feature"] = feature_loss
        return ValidationResultWrapper(data=losses)

    def predict_feature(
        self, g: gd.OneHotGraph, t: Int[Array, "b"], rng: Key
    ) -> tuple[gd.OneHotGraph, gd.OneHotGraph]:
        """Applies noise and then predicts the 'corrected' features of the graph.
        Returns the graph with noised features and the graph with predicted features.
        """
        target_structure, target_feature = g.decompose_structure_and_feature()
        p_feature_deterministic = jax.tree_util.Partial(
            self.p_feature, structure=target_structure, deterministic=True
        )
        t, g_t, g_pred_raw = df.predict_from_timesteps(
            p_feature_deterministic,
            target_feature,
            self.feature_transition_model,
            t,
            rng,
        )
        g_t_corrected_structure = target_structure.feature_like(g_t)
        g_pred = gd.softmax(g_pred_raw).argmax()
        return g_t_corrected_structure, g_pred

    def predict_structure(
        self, g: gd.OneHotGraph, t: Int[Array, "b"], rng: Key
    ) -> tuple[gd.OneHotGraph, gd.OneHotGraph]:
        """Applies noise and then predicts the corrected structure of the graph.
        Returns the graph with noised structure and the graph with predicted structure.
        """
        target_structure = g.structure_one_hot()
        t, g_t, g_pred_raw = df.predict_from_timesteps(
            self.p_structure_deterministic,
            target_structure,
            self.structure_transition_model,
            t,
            rng,
        )
        g_pred = gd.softmax(g_pred_raw).argmax()
        return g_t, g_pred

    @jaxtyped
    @beartype
    def __call__(
        self,
        target: gd.OneHotGraph,
        rng_key: Key,
    ):
        target_structure, target_feature = target.decompose_structure_and_feature()
        p_feature_nondeterministic = jax.tree_util.Partial(
            self.p_feature,
            structure=target_structure,
            feature=target_feature,
            deterministic=False,
        )
        tot_loss = 0
        if self.use_structure:
            structure_loss = df.compute_train_loss(
                target=target_structure,
                transition_model=self.structure_transition_model,
                rng_key=rng_key,
                get_probability=self.p_structure_nondeterministic,
            )
            tot_loss += structure_loss
        if self.use_feature:
            feature_loss = df.compute_train_loss(
                target=target_feature,
                transition_model=self.feature_transition_model,
                rng_key=rng_key,
                get_probability=p_feature_nondeterministic,
            )
            tot_loss += feature_loss

        return tot_loss

    def _sample_step_structure(self, rng: Key, t: Int[Array, "b"], g):
        return df.sample_step(
            p=self.p_structure_deterministic,
            t=t,
            g_t=g,
            transition_model=self.structure_transition_model,
            rng=rng,
        )

    def _sample_step_feature(
        self,
        target_structure: gd.OneHotGraph,
        rng: Key,
        t: Int[Array, "b"],
        g: gd.OneHotGraph,
    ):
        p_feature_deterministic = jax.tree_util.Partial(
            self.p_feature, structure=target_structure, deterministic=True
        )
        sampled = df.sample_step(
            p=p_feature_deterministic,
            t=t,
            g_t=g.feature(),
            transition_model=self.feature_transition_model,
            rng=rng,
        )
        return target_structure.feature_like(sampled)

    def _sample_structure(self, sample_step: Callable, rng: Key, n_samples: SInt):
        rng_n_nodes = jax.random.fold_in(rng, 0)
        rng_prior, rng_initial = jax.random.split(rng)
        nodes_counts = jax.random.categorical(
            rng_n_nodes, np.log(self.nodes_dist), shape=(n_samples,)
        )
        nodes_mask, edges_mask = gd.get_masks(
            nodes_counts, self.nodes_dist.shape[0] - 1
        )
        g = gd.sample_one_hot(
            self.structure_transition_model.limit_dist.repeat(n_samples), rng_initial
        )
        g = gd.OneHotGraph.create(
            nodes=g.nodes * nodes_mask[..., None],
            edges=g.edges * edges_mask[..., None],
            nodes_mask=nodes_mask,
            edges_mask=edges_mask,
        )
        for t in tqdm(
            list(reversed(range(1, self.structure_transition_model.diffusion_steps)))
        ):
            rng = jax.random.fold_in(rng, t)
            t = np.array(t).repeat(n_samples)
            g = sample_step(rng, t, g)
        g_prior = gd.sample_one_hot(
            self.structure_transition_model.limit_dist.repeat(n_samples), rng_prior
        )
        return g, g_prior

    @jaxtyped
    @beartype
    def _sample_feature(
        self,
        sample_step: Callable,
        structure: gd.OneHotGraph,
        rng: Key,
        n_samples: SInt,
    ):
        rng_prior, rng_initial = jax.random.split(rng)
        g = gd.sample_one_hot(
            self.feature_transition_model.limit_dist.repeat(n_samples), rng_initial
        )
        g = structure.feature_like(g)
        for t in tqdm(
            list(reversed(range(1, self.feature_transition_model.diffusion_steps)))
        ):
            rng = jax.random.fold_in(rng, t)
            t = np.array(t).repeat(n_samples)
            g = sample_step(structure, rng, t, g)
        g_prior = gd.sample_one_hot(
            self.feature_transition_model.limit_dist.repeat(n_samples), rng_prior
        )
        return g, g_prior

    def sample(self, params: FrozenDict, rng: Key, n_samples: SInt):
        sample_step_structure = jit(
            jax.tree_util.Partial(
                self.apply, params, method=self._sample_step_structure
            )
        )
        sample_step_feature = jit(
            jax.tree_util.Partial(self.apply, params, method=self._sample_step_feature)
        )

        sampled_structure, sample_structure_rior = self.apply(
            params,
            sample_step_structure,
            rng,
            n_samples,
            method=self._sample_structure,
        )
        sampled_feature, sample_feature_prior = self.apply(
            params,
            sample_step_feature,
            sampled_structure,
            rng,
            n_samples,
            method=self._sample_feature,
        )
        return sampled_feature

    def get_model_input(self, g: gd.GraphDistribution, t: Int[Array, "bs"]):
        return get_model_input(
            g, t, self.use_extra_features, self.structure_transition_model
        )
