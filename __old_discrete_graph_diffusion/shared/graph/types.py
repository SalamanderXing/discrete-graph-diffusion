from mate.jax import typed, SInt
from mate.types import Interface
from jaxtyping import Array, Float, Bool, Int
from numpy import ndarray
from flax.struct import dataclass
from mate.jax import typed, Key
from mate.types import Interface
import jax
from jax import numpy as jnp
import ipdb

Nodes = Float[Array, "b n k"]
Edges = Float[Array, "b n n"]
Masks = Int[Array, "b"]


@dataclass
class Graph(metaclass=Interface):
    nodes: Nodes
    edges: Edges
    edges_counts: Masks
    nodes_counts: Masks
    _internal: bool = False

    def __post_init__(self):
        assert self._internal, "Graph must be created with Graph.create"

    @classmethod
    @typed
    def create(
        cls, nodes: Nodes, edges: Edges, edges_counts: Masks, nodes_counts: Masks
    ) -> "Graph":
        # masks = jnp.ones((nodes.shape[0], nodes.shape[1]), dtype=bool)
        return cls(nodes, edges, edges_counts, nodes_counts, _internal=True)

    @typed
    def __add__(self, other) -> "Graph":
        if isinstance(other, Graph):
            return Graph.create(
                nodes=self.nodes + other.nodes,
                edges=self.edges + other.edges,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
            )
        else:
            return Graph.create(
                nodes=self.nodes + other,
                edges=self.edges + other,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
            )

    @typed
    def __mul__(self, other) -> "Graph":
        if isinstance(other, Graph):
            return Graph.create(
                nodes=self.nodes * other.nodes,
                edges=self.edges * other.edges,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
            )
        else:
            return Graph.create(
                nodes=self.nodes * other,
                edges=self.edges * other,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
            )

    @typed
    def __sub__(self, other) -> "Graph":
        if isinstance(other, Graph):
            return Graph.create(
                nodes=self.nodes - other.nodes,
                edges=self.edges - other.edges,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
            )
        else:
            return Graph.create(
                nodes=self.nodes - other,
                edges=self.edges - other,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
            )

    @typed
    def __truediv__(self, other) -> "Graph":
        if isinstance(other, Graph):
            return Graph.create(
                nodes=self.nodes / other.nodes,
                edges=self.edges / other.edges,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
            )
        else:
            return Graph.create(
                nodes=self.nodes / other,
                edges=self.edges / other,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
            )

    @typed
    def __pow__(self, other) -> "Graph":
        if isinstance(other, Graph):
            return Graph.create(
                nodes=self.nodes**other.nodes,
                edges=self.edges**other.edges,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
            )
        else:
            return Graph.create(
                nodes=self.nodes**other,
                edges=self.edges**other,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
            )

    @typed
    def noise_like(self, key: Key) -> "Graph":
        edge_noise = jax.random.normal(key, self.edges.shape)
        edge_noise = (edge_noise + edge_noise.transpose((0, 2, 1))) / 2
        return Graph.create(
            nodes=jax.random.normal(key, self.nodes.shape),
            edges=edge_noise,
            nodes_counts=self.nodes_counts,
            edges_counts=self.edges_counts,
        )

    @property
    def node_mask(self):
        mask = (
            jnp.broadcast_to(
                jnp.arange(self.nodes.shape[1])[None, :, None],
                self.nodes.shape,
            )
            < self.nodes_counts[:, None, None]
        )
        return mask

    @property
    def edge_mask(self):
        mask = (
            jnp.broadcast_to(
                jnp.arange(self.edges.shape[1])[None, :, None],
                self.edges.shape,
            )
            < self.edges_counts[:, None, None]
        )
        return mask

    def sum(self) -> Float[Array, "b"]:
        nodes = self.nodes * self.node_mask
        edges = self.edges * self.edge_mask
        return nodes.sum((1, 2)) + edges.sum((1, 2))

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rtruediv__ = __truediv__
