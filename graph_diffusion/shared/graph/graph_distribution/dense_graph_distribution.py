from .graph_distribution import (
    GraphDistribution,
    NodeDistribution,
    EdgeDistribution,
    NodeMaskType,
    get_masks,
)
from jaxtyping import jaxtyped
from flax.struct import dataclass
from beartype import beartype


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
