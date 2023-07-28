from .graph_distribution import (
    GraphDistribution,
    NodeDistribution,
    EdgeDistribution,
    NodeMaskType,
    EdgeMaskType,
    get_masks,
)
from jax import numpy as np
from jaxtyping import jaxtyped
from flax.struct import dataclass
from beartype import beartype
from .one_hot_graph_distribution import OneHotGraph

@dataclass
class DenseGraphDistribution(GraphDistribution):
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

    @jaxtyped
    @beartype
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

    @jaxtyped
    @beartype
    def repeat(self, n: int) -> "DenseGraphDistribution":
        return DenseGraphDistribution.create(
            nodes=np.repeat(g.nodes, n, axis=0),
            edges=np.repeat(g.edges, n, axis=0),
            nodes_mask=np.repeat(g.nodes_mask, n, axis=0),
            edges_mask=np.repeat(g.edges_mask, n, axis=0),
        )

    @jaxtyped
    @beartype
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
