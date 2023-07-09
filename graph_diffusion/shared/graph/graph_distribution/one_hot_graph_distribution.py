from .graph_distribution import (
    GraphDistribution,
    NodeDistribution,
    EdgeDistribution,
    NodeMaskType,
    EdgeMaskType,
    get_masks,
)
from jaxtyping import jaxtyped
from flax.struct import dataclass
from beartype import beartype
from jax import random
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
    def noise(
        cls,
        key: Key,
        num_node_features: int,
        num_edge_features: int,
        num_nodes: int,
    ) -> "OneHotGraph":
        key, subkey = random.split(key)
        batch_size = 2

        edges = random.normal(
            key, (batch_size, num_nodes, num_nodes, num_edge_features)
        )
        edges = to_symmetric(edges)
        edges = np.where(np.eye(num_nodes)[None, :, :, None], 0, edges)
        nodes = random.normal(subkey, (batch_size, num_nodes, num_node_features))
        nodes_mask = np.ones(nodes.shape[:-1], bool)
        edges_mask = np.ones(edges.shape[:-1], bool)
        return cls.create(
            nodes=nodes,
            edges=edges,
            nodes_mask=nodes_mask,
            edges_mask=edges_mask,
        )
