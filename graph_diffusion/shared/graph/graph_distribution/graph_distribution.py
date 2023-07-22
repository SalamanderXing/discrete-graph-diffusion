from jax import numpy as np, Array
import jax
import networkx as nx
import optax
from jax.experimental.checkify import check
from rich import print
import ipdb
from flax.struct import dataclass
from mate.jax import SFloat, SInt, SBool, Key
from jaxtyping import Float, Bool, Int
from typing import Sequence
from jaxtyping import jaxtyped
from beartype import beartype
from jax.scipy.special import logit

# import einops as e
from einop import einop
import wandb

# from .geometric import to_dense
from jax import random

# from .data_batch import DataBatch
from .q import Q

# from ..graph import SimpleGraphDist

check = lambda x, y: None  # to be replaced with JAX's checkify.check function

NodeDistribution = Float[Array, "b n en"]
EdgeDistribution = Float[Array, "b n n ee"]
NodeMaskType = Bool[Array, "b n"]
EdgeMaskType = Bool[Array, "b n n"]
EdgeCountType = Int[Array, "b"]


def safe_div(a: Array, b: Array):
    mask = b == 0
    return np.where(mask, 0, a / np.where(mask, 1, b))


@jaxtyped
@beartype
def pseudo_assert(condition: SBool):
    """
    When debugging NaNs, the following function will raise an exception
    """
    return np.where(condition, 0, np.nan)


@jaxtyped
@beartype
def get_masks(
    nodes_counts: Int[Array, "bs"], n: SInt
) -> tuple[NodeMaskType, EdgeMaskType]:
    n_range = np.arange(n)
    bs = nodes_counts.shape[0]
    mask_x = einop(n_range, "n -> bs n", bs=bs) < einop(nodes_counts, "bs -> bs n", n=n)
    e_ranges = einop(mask_x, "bs n -> bs n n1", n1=n)
    e_diag = einop(np.eye(n, dtype=bool), "n1 n2 -> 1 n1 n2")
    mask_e = e_ranges & einop(e_ranges, "bs n n1 -> bs n1 n") & ~e_diag
    return mask_x, mask_e


@jaxtyped
@beartype
def to_symmetric(edges: EdgeDistribution) -> EdgeDistribution:
    upper = einop(
        np.triu(np.ones((edges.shape[1], edges.shape[2]))), "n1 n2 -> 1 n1 n2 1"
    )
    return np.where(upper, edges, einop(edges, "b n1 n2 ee -> b n2 n1 ee"))


@dataclass
class GraphDistribution:
    nodes: NodeDistribution
    edges: EdgeDistribution
    nodes_mask: NodeMaskType
    edges_mask: EdgeMaskType
    # mask: MaskType
    _created_internally: SBool  # trick to prevent users from creating this class directly

    @property
    def nodes_counts(self):
        return self.nodes_mask.sum(-1)

    # def pseudo_assert(self):
    #     is_nodes_dist = np.array(is_dist(self.nodes))
    #     pseudo_assert(is_nodes_dist)
    #     is_edges_dist = np.array(is_dist(self.edges))
    #     pseudo_assert(is_edges_dist)

    @classmethod
    @jaxtyped
    @beartype
    def from_mask(
        cls, nodes: NodeDistribution, edges: EdgeDistribution, mask: NodeMaskType
    ):
        nodes_mask, edges_mask = get_masks(mask.sum(-1), nodes.shape[1])
        return cls(
            nodes=nodes,
            edges=edges,
            nodes_mask=nodes_mask,
            edges_mask=edges_mask,
            _created_internally=True,
        )

    @classmethod
    @jaxtyped
    @beartype
    def create(
        cls,
        nodes: NodeDistribution,
        edges: EdgeDistribution,
        nodes_mask: NodeMaskType,
        edges_mask: EdgeMaskType,
    ):
        return cls(
            nodes=nodes,
            edges=edges,
            nodes_mask=nodes_mask,
            edges_mask=edges_mask,
            _created_internally=True,
        )

    @classmethod
    @jaxtyped
    @beartype
    def noise(
        cls,
        key: Key,
        num_node_features: int,
        num_edge_features: int,
        num_nodes: int,
    ):
        rng_nodes, rng_edges = random.split(key)
        batch_size = 1

        edges = random.normal(
            rng_edges, (batch_size, num_nodes, num_nodes, num_edge_features)
        )
        edges = to_symmetric(edges)
        edges = np.where(np.eye(num_nodes)[None, :, :, None], 0, edges)
        nodes = random.normal(rng_nodes, (batch_size, num_nodes, num_node_features))
        nodes_mask = np.ones(nodes.shape[:-1], bool)
        edges_mask = np.ones(edges.shape[:-1], bool)
        return cls.create(
            nodes=nodes,
            edges=edges,
            nodes_mask=nodes_mask,
            edges_mask=edges_mask,
        )

        # # overrides the square bracket indexing

    def __getitem__(self, key: Int[Array, "n"] | slice) -> "GraphDistribution":
        return self.__class__(
            nodes=self.nodes[key],
            edges=self.edges[key],
            nodes_mask=self.nodes_mask[key],
            edges_mask=self.edges_mask[key],
            _created_internally=True,
        )  # , mask=self.mask[key])

    #
    def __len__(self):
        return self.nodes.shape[0]


def is_dist(x):
    return (
        (np.min(x) >= 0).all() & (np.max(x) <= 1).all() & np.allclose(np.sum(x, -1), 1)
    )