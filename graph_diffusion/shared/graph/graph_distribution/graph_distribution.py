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
from jax import random

# from .data_batch import DataBatch
from .q import Q

from einops import rearrange, reduce, repeat

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


def pseudo_assert(condition: SBool):
    """
    When debugging NaNs, the following function will raise an exception
    """
    return np.where(condition, 0, np.nan)


def is_dist(x):
    return (
        (np.min(x) >= 0).all() & (np.max(x) <= 1).all() & np.allclose(np.sum(x, -1), 1)
    )


def get_masks(
    nodes_counts: Int[Array, "bs"], n: SInt
) -> tuple[NodeMaskType, EdgeMaskType]:
    n_range = np.arange(n)
    bs = nodes_counts.shape[0]
    mask_x = repeat(n_range, "n -> bs n", bs=bs) < repeat(
        nodes_counts, "bs -> bs n", n=n
    )
    e_ranges = repeat(mask_x, "bs n -> bs n n1", n1=n)
    e_diag = rearrange(np.eye(n, dtype=bool), "n1 n2 -> 1 n1 n2")
    mask_e = e_ranges & rearrange(e_ranges, "bs n n1 -> bs n1 n") & ~e_diag
    return mask_x, mask_e


def to_symmetric(edges: EdgeDistribution) -> EdgeDistribution:
    upper = rearrange(
        np.triu(np.ones((edges.shape[1], edges.shape[2]))), "n1 n2 -> 1 n1 n2 1"
    )
    return np.where(upper, edges, rearrange(edges, "b n1 n2 ee -> b n2 n1 ee"))


@jaxtyped
@beartype
@dataclass
class GraphDistribution:
    nodes: NodeDistribution
    edges: EdgeDistribution
    nodes_mask: NodeMaskType
    edges_mask: EdgeMaskType
    # mask: MaskType
    _created_internally: SBool  # trick to prevent users from creating this class directly

    @property
    def batch_size(self):
        return self.nodes.shape[0]

    @property
    def n(self):
        return self.nodes.shape[1]

    @property
    def nodes_counts(self):
        return self.nodes_mask.sum(-1)

    # def pseudo_assert(self):
    #     is_nodes_dist = np.array(is_dist(self.nodes))
    #     pseudo_assert(is_nodes_dist)
    #     is_edges_dist = np.array(is_dist(self.edges))
    #     pseudo_assert(is_edges_dist)

    @classmethod
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

    def __getitem__(self, key) -> "GraphDistribution":
        new_nodes = self.nodes[key]
        cls = (
            self.__class__
            if self.__class__ != StructureOneHotGraph
            else GraphDistribution
        )
        return cls.create(
            nodes=new_nodes,
            edges=self.edges[key],
            nodes_mask=self.nodes_mask[key],
            edges_mask=self.edges_mask[key],
        )  # , mask=self.mask[key])

    #
    def __len__(self):
        return self.nodes.shape[0]

    @classmethod
    @beartype
    @jaxtyped
    def create_and_mask(
        cls,
        nodes: NodeDistribution,
        edges: EdgeDistribution,
        nodes_mask: NodeMaskType,
        edges_mask: EdgeMaskType,
    ):
        nodes = np.where(nodes_mask[..., None], nodes, 0)
        edges = to_symmetric(np.where(edges_mask[..., None], edges, 0))
        return cls(
            nodes=nodes,
            edges=edges,
            nodes_mask=nodes_mask,
            edges_mask=edges_mask,
            _created_internally=True,
        )

    @classmethod
    def create_from_counts(
        cls,
        nodes: NodeDistribution,
        edges: EdgeDistribution,
        nodes_counts: EdgeCountType,
    ):
        nodes_mask, edges_mask = get_masks(nodes_counts, nodes.shape[1])
        return cls(
            nodes=nodes,
            edges=edges,
            nodes_mask=nodes_mask,
            edges_mask=edges_mask,
            _created_internally=True,
        )

    @classmethod
    def create_minimal(
        cls,
        nodes: NodeDistribution,
        edges: EdgeDistribution,
        nodes_mask: NodeMaskType,
    ):
        _, edges_mask = get_masks(nodes_mask.sum(-1), edges.shape[1])
        return cls(
            nodes=nodes,
            edges=edges,
            nodes_mask=nodes_mask,
            edges_mask=edges_mask,
            _created_internally=True,
        )

    def repeat(self, n: SInt):
        return GraphDistribution.create(
            nodes=np.repeat(self.nodes, n, axis=0),
            edges=np.repeat(self.edges, n, axis=0),
            nodes_mask=np.repeat(self.nodes_mask, n, axis=0),
            edges_mask=np.repeat(self.edges_mask, n, axis=0),
        )

    def decompose_structure_and_feature(self):
        return (self.__structure(), self.feature())

    def __structure(self):
        yes = np.array([1, 0])
        no = np.array([0, 1])
        new_nodes = np.where(
            self.nodes_mask[..., None], no[None, None], yes[None, None]
        ).astype(float)
        new_edges = np.where(
            (self.edges_mask & (self.edges[..., 0] < 0.5))[..., None],
            no[None, None, None],
            yes[None, None, None],
        ).astype(float)
        # ipdb.set_trace()
        return StructureOneHotGraph.create(
            OneHotGraph.create(
                nodes=new_nodes,
                edges=new_edges,
                nodes_mask=self.nodes_mask,
                edges_mask=self.edges_mask,
            )
        )

    def feature(self):
        g = OneHotGraph.create(
            nodes=self.nodes,
            edges=self.edges[..., 1:],
            nodes_mask=self.nodes_mask,
            edges_mask=self.edges_mask,
        )
        return g


@jaxtyped
@beartype
@dataclass
class OneHotGraph(GraphDistribution):
    @classmethod
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

    def repeat(self, n: int):
        return OneHotGraph.create(
            nodes=np.repeat(self.nodes, n, axis=0),
            edges=np.repeat(self.edges, n, axis=0),
            nodes_mask=np.repeat(self.nodes_mask, n, axis=0),
            edges_mask=np.repeat(self.edges_mask, n, axis=0),
        )

    def mask(self) -> "OneHotGraph":
        g = self
        nodes = np.where(g.nodes_mask[..., None], g.nodes, 0)
        edges = np.where(g.edges_mask[..., None], g.edges, 0)
        return OneHotGraph.create(
            nodes=nodes,
            edges=edges,
            nodes_mask=g.nodes_mask,
            edges_mask=g.edges_mask,
        )

    @classmethod
    def empty(cls, b: int, n: int, d_nodes: int, d_edges: int):
        id = ~np.eye(n, bool)
        nodes_mask = np.ones((b, n), bool)
        edges_mask = repeat(id, "n n -> b n n", b=b)
        return cls.create(
            nodes=np.zeros((b, n, d_nodes)),
            edges=np.zeros((b, n, n, d_edges)),
            nodes_mask=nodes_mask,
            edges_mask=edges_mask,
        )


@jaxtyped
@beartype
@dataclass
class DenseGraphDistribution(GraphDistribution):
    @classmethod
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
    def create_minimal(
        cls,
        nodes: NodeDistribution,
        edges: EdgeDistribution,
        nodes_mask: NodeMaskType,
    ):
        _, edges_mask = get_masks(nodes_mask.sum(-1), edges.shape[1])
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
    def create_and_mask(
        cls,
        nodes: NodeDistribution,
        edges: EdgeDistribution,
        nodes_mask: NodeMaskType,
        edges_mask: EdgeMaskType,
    ):
        nodes = np.where(nodes_mask[..., None], nodes, 0)
        edges = np.where(edges_mask[..., None], edges, 0)
        return cls(
            nodes=nodes,
            edges=edges,
            nodes_mask=nodes_mask,
            edges_mask=edges_mask,
            _created_internally=True,
        )

    def copy_structure(self, other: OneHotGraph) -> "DenseGraphDistribution":
        """
        Wherever other has a 1 in the fist position of the last dimension, copy that
        value from other to self. In all other cases, the value remains unchanged.
        """
        nodes = np.where(other.nodes[..., 0][..., None], other.nodes, self.nodes)
        edges = np.where(other.edges[..., 0][..., None], other.edges, self.edges)
        return DenseGraphDistribution.create(
            nodes=nodes,
            edges=edges,
            nodes_mask=self.nodes_mask,
            edges_mask=self.edges_mask,
        )

    def mask_dense(self) -> "DenseGraphDistribution":
        g = self
        nodes = np.where(g.nodes_mask[..., None], g.nodes, 0)
        edges = np.where(g.edges_mask[..., None], g.edges, 0)
        return DenseGraphDistribution.create(
            nodes=nodes,
            edges=edges,
            nodes_mask=g.nodes_mask,
            edges_mask=g.edges_mask,
        )

    def repeat(self, n: SInt) -> "DenseGraphDistribution":
        return DenseGraphDistribution.create(
            nodes=np.repeat(self.nodes, n, axis=0),
            edges=np.repeat(self.edges, n, axis=0),
            nodes_mask=np.repeat(self.nodes_mask, n, axis=0),
            edges_mask=np.repeat(self.edges_mask, n, axis=0),
        )

    def argmax(self) -> OneHotGraph:
        id_nodes = np.eye(self.nodes.shape[-1])
        id_edges = np.eye(self.edges.shape[-1])
        nodes = id_nodes[self.nodes.argmax(-1)]
        edges = id_edges[self.edges.argmax(-1)]
        return OneHotGraph.create(
            nodes=nodes,
            edges=edges,
            nodes_mask=self.nodes_mask,
            edges_mask=self.edges_mask,
        )


class StructureOneHotGraph(OneHotGraph):
    nodes: Float[Array, "b n 2"]
    edges: Float[Array, "b n n 2"]
    nodes_mask: Bool[Array, "b n 2"]
    edges_mask: Bool[Array, "b n n 2"]

    @classmethod
    def create(
        cls,
        g: OneHotGraph,
    ):
        return cls(
            nodes=g.nodes,
            edges=g.edges,
            nodes_mask=g.nodes_mask,
            edges_mask=g.edges_mask,
            _created_internally=True,
        )


class FeatureOneHotGraph(OneHotGraph):
    @classmethod
    @jaxtyped
    @beartype
    def create(cls, g: GraphDistribution):
        return cls(
            nodes=g.nodes,
            edges=g.edges,
            nodes_mask=g.nodes_mask,
            edges_mask=g.edges_mask,
            _created_internally=True,
        )

    def apply_structure(self, structure: StructureOneHotGraph):
        structure_nodes = structure.nodes.argmax(-1)
        structure_edges = structure.edges.argmax(-1)
        null_node = np.eye(self.nodes.shape[-1])[0]
        new_nodes = np.where(structure_nodes[..., None], self.nodes, null_node)
        null_edge = np.eye(self.edges.shape[-1])[0]
        new_edges = np.where(structure_edges[..., None], self.edges, null_edge)
        return DenseGraphDistribution.create(
            nodes=new_nodes,
            edges=new_edges,
            nodes_mask=self.nodes_mask,
            edges_mask=self.edges_mask,
        )

    def restore_structure(
        self, structure: StructureOneHotGraph
    ) -> DenseGraphDistribution:
        structure_nodes = structure.nodes.argmax(-1)
        structure_edges = structure.edges.argmax(-1)
        # puts a zero in the first position of the last dimension
        extended_nodes = np.concatenate(
            [np.zeros_like(self.nodes[..., :1]), self.nodes], axis=-1
        )
        # a null node has the same shape as extended_nodes but 0 in the first position
        null_node = np.eye(extended_nodes.shape[-1])[0]
        new_nodes = np.where(structure_nodes[..., None], extended_nodes, null_node)
        extended_edges = np.concatenate(
            [np.zeros_like(self.edges[..., :1]), self.edges], axis=-1
        )
        null_edge = np.eye(extended_edges.shape[-1])[0]
        new_edges = np.where(structure_edges[..., None], extended_edges, null_edge)
        return DenseGraphDistribution.create(
            nodes=new_nodes,
            edges=new_edges,
            nodes_mask=self.nodes_mask,
            edges_mask=self.edges_mask,
        )
