from .graph_distribution import (
    GraphDistribution,
    NodeDistribution,
    EdgeDistribution,
    NodeMaskType,
    EdgeMaskType,
    get_masks,
    to_symmetric,
)
from jaxtyping import jaxtyped
from flax.struct import dataclass
from beartype import beartype
from jax import random
from jax import numpy as np
from mate.jax import Key


@dataclass
class OneHotGraph(GraphDistribution):
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

    @jaxtyped
    @beartype
    def repeat(self, n: int):
        return OneHotGraph.create(
            nodes=np.repeat(self.nodes, n, axis=0),
            edges=np.repeat(self.edges, n, axis=0),
            nodes_mask=np.repeat(self.nodes_mask, n, axis=0),
            edges_mask=np.repeat(self.edges_mask, n, axis=0),
        )
