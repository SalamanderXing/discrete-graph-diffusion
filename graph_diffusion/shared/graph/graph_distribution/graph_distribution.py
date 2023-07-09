from jax import numpy as np, Array
import jax
import networkx as nx
import optax
from jax.experimental.checkify import check
from rich import print
import ipdb
from flax.struct import dataclass
from mate.jax import SFloat, SInt, typed, SBool, Key
from jaxtyping import Float, Bool, Int
from typing import Sequence
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


@typed
def pseudo_assert(condition: SBool):
    """
    When debugging NaNs, the following function will raise an exception
    """
    return np.where(condition, 0, np.nan)


@typed
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


@typed
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
    @typed
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

    # @classmethod
    # @typed
    # def create(
    #     cls,
    #     nodes: NodeDistribution,
    #     edges: EdgeDistribution,
    #     nodes_mask: NodeMaskType,
    #     edges_mask: EdgeMaskType,
    # ):
    #     return cls(
    #         nodes=nodes,
    #         edges=edges,
    #         nodes_mask=nodes_mask,
    #         edges_mask=edges_mask,
    #         _created_internally=True,
    #     )
    #
    # @classmethod
    # @typed
    # def create_and_mask(
    #     cls,
    #     nodes: NodeDistribution,
    #     edges: EdgeDistribution,
    #     nodes_mask: NodeMaskType,
    #     edges_mask: EdgeMaskType,
    # ):
    #     nodes = np.where(nodes_mask[..., None], nodes, 0)
    #     edges = np.where(edges_mask[..., None], edges, 0)
    #     return cls(
    #         nodes=nodes,
    #         edges=edges,
    #         nodes_mask=nodes_mask,
    #         edges_mask=edges_mask,
    #         _created_internally=True,
    #     )
    #
    # @classmethod
    # @typed
    # def create_from_counts(
    #     cls,
    #     nodes: NodeDistribution,
    #     edges: EdgeDistribution,
    #     nodes_counts: EdgeCountType,
    #     _safe: SBool = True,
    # ):
    #     is_nodes_dist = np.logical_or(is_dist(nodes), (~_safe))
    #     # if not is_nodes_dist:
    #     #     ipdb.set_trace()
    #     pseudo_assert(is_nodes_dist)
    #     is_edges_dist = np.logical_or(is_dist(edges), (~_safe))
    #     # if not is_edges_dist:
    #     #     ipdb.set_trace()
    #     pseudo_assert(is_edges_dist)
    #     nodes_mask, edges_mask = get_masks(nodes_counts, nodes.shape[1])
    #     return cls(
    #         nodes=nodes,
    #         edges=edges,
    #         nodes_mask=nodes_mask,
    #         edges_mask=edges_mask,
    #         _created_internally=True,
    #     )

    # def __str__(self):
    #     def arr_str(arr: Array):
    #         return f"Array({arr.shape}, max={arr.max():.2f}, min={arr.min():.2f}, mean={arr.mean():.2f}, dtype={arr.dtype}, is_dist={is_dist(arr)})"
    #
    #     def self_arr(props: dict):
    #         return ",\n ".join(
    #             [
    #                 f"{key}: {arr_str(val)}"
    #                 for key, val in props.items()
    #                 if not key.startswith("_")
    #             ]
    #         )
    #
    #     return f"{self.__class__.__name__}(\n {self_arr(self.__dict__)}\n)"

    # @property
    # def shape(self) -> dict[str, tuple[int, ...]]:
    #     return dict(
    #         nodes=self.nodes.shape, edges=self.edges.shape
    #     )  # , mask=self.mask.shape)
    #
    # @property
    # def batch_size(self) -> int:
    #     return self.nodes.shape[0]
    #
    # @property
    # def n(self) -> int:
    #     return self.nodes.shape[1]
    #
    # def __repr__(self):
    #     return self.__str__()
    #
    # # def __rmul__(
    # #     self, other: "GraphDistribution | SFloat | SInt"
    # # ) -> "GraphDistribution":
    # #     return self.__mul__(other)
    #
    # def set(
    #     self,
    #     key: str,
    #     value: NodeDistribution | EdgeDistribution,
    # ) -> "GraphDistribution":
    #     """Sets the values of X, E, y."""
    #     new_vals = {
    #         k: v
    #         for k, v in (self.__dict__.copy() | {key: value}).items()
    #         if not k.startswith("__")
    #     }
    #     return GraphDistribution(**new_vals)
    #
    # # overrides te addition operator
    #
    # # def __sub__(self, other) -> "GraphDistribution":
    # #     if isinstance(other, GraphDistribution):
    # #         return self.__class__.create(
    # #             nodes=self.nodes - other.nodes,
    # #             edges=self.edges - other.edges,
    # #             nodes_mask=self.nodes_mask,
    # #             edges_mask=self.edges_mask,
    # #         )
    # #     elif isinstance(other, (int, float, Array)):
    # #         return self.__class__.create(
    # #             nodes=self.nodes - other,
    # #             edges=self.edges - other,
    # #             nodes_mask=self.nodes_mask,
    # #             edges_mask=self.edges_mask,
    # #         )
    # #
    # # __rsub__ = __sub__
    #
    # # def __matmul__(self, q: Q) -> "GraphDistribution":
    #
    # # def __pow__(self, n: int) -> "GraphDistribution":
    # #     return self.__class__.create(
    # #         nodes=self.nodes**n,
    # #         edges=self.edges**n,
    # #         nodes_mask=self.nodes_mask,
    # #         edges_mask=self.edges_mask,
    # #     )
    # #
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

    #
    # @classmethod
    # @typed
    # def noise(
    #     cls,
    #     key: Key,
    #     num_node_features: int,
    #     num_edge_features: int,
    #     num_nodes: int,
    #     batch_size: int,
    # ) -> "GraphDistribution":
    #     nodes_counts = np.array(
    #         [num_nodes] * batch_size
    #     )  # jax.random.randint(key, shape=(batch_size,),   int)
    #     # nodes_counts = jax.random.randint(
    #     #     key, shape=(batch_size,), minval=8, maxval=num_nodes + 1
    #     # )
    #     edges = jax.nn.softmax(
    #         random.normal(key, (batch_size, num_nodes, num_nodes, num_edge_features))
    #     )
    #     edges = to_symmetric(edges)
    #     # sets the diagonal to 0
    #
    #     tmp = einop(
    #         np.eye(edges.shape[-1])[0],
    #         "f -> b n1 n2 f",
    #         n1=num_nodes,
    #         n2=num_nodes,
    #         b=batch_size,
    #     )
    #     edges = np.where(np.eye(num_nodes)[None, :, :, None], tmp, edges)
    #     return cls.create_from_counts(
    #         nodes=jax.nn.softmax(
    #             random.normal(key, (batch_size, num_nodes, num_node_features))
    #         ),
    #         edges=edges,
    #         nodes_counts=nodes_counts,
    #     )
    #


def is_dist(x):
    return (
        (np.min(x) >= 0).all() & (np.max(x) <= 1).all() & np.allclose(np.sum(x, -1), 1)
    )
