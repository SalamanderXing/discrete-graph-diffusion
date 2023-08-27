from dataclasses import dataclass
from jax import Array
import jax
from jax import numpy as np
import ipdb
from jaxtyping import Float, Bool, Int, jaxtyped
from flax.struct import dataclass
from beartype import beartype
from typing import Union

from ...shared.graph.graph_distribution import DenseGraphDistribution, GraphDistribution


@jaxtyped
@beartype
@dataclass
class Q:
    nodes: Float[Array, "t en en"]
    edges: Float[Array, "t ee ee"]

    # overrides the square bracket indexing
    @jaxtyped
    def __getitem__(self, key) -> "Q":
        return Q(nodes=self.nodes[key], edges=self.edges[key])

    @property
    def shape(self) -> dict[str, tuple[int, ...]]:
        return dict(x=self.nodes.shape, edges=self.edges.shape)

    def __truediv__(self, other: Union[Float[Array, " "], "Q"]) -> "Q":
        if isinstance(other, Q):
            return Q(nodes=self.nodes / other.nodes, edges=self.edges / other.edges)
        else:
            return Q(nodes=self.nodes / other, edges=self.edges / other)

    def __matmul__(self, other: "Q") -> "Q":
        return Q(nodes=self.nodes @ other.nodes, edges=self.edges @ other.edges)

    def __len__(self) -> int:
        return self.nodes.shape[0]

    def cumulative_matmul(self) -> "Q":
        def f(a, b):
            result = a @ b
            return result, result

        res_x = jax.lax.scan(f, self.nodes, self.nodes[0][None])[1]
        res_e = jax.lax.scan(f, self.edges, self.edges[0][None])[1]
        return Q(nodes=res_x[0], edges=res_e[0])

    def apply_to(self, g: GraphDistribution) -> DenseGraphDistribution:
        return DenseGraphDistribution.create(
            nodes=g.nodes @ self.nodes,
            edges=g.edges @ self.edges[:, None],
            edges_mask=g.edges_mask,
            nodes_mask=g.nodes_mask,
        )


@jaxtyped
@beartype
@dataclass
class StructureOnlyQ(Q):
    nodes: Float[Array, "t en en"]
    edges: Float[Array, "t ee ee"]

    # overrides the square bracket indexing
    @jaxtyped
    def __getitem__(self, key) -> "Q":
        return Q(nodes=self.nodes[key], edges=self.edges[key])

    @property
    def shape(self) -> dict[str, tuple[int, ...]]:
        return dict(x=self.nodes.shape, edges=self.edges.shape)

    def __truediv__(self, other: Union[Float[Array, " "], "Q"]) -> "Q":
        if isinstance(other, Q):
            return Q(nodes=self.nodes / other.nodes, edges=self.edges / other.edges)
        else:
            return Q(nodes=self.nodes / other, edges=self.edges / other)

    def __matmul__(self, other: "Q") -> "Q":
        return Q(nodes=self.nodes @ other.nodes, edges=self.edges @ other.edges)

    def __len__(self) -> int:
        return self.nodes.shape[0]

    def cumulative_matmul(self) -> "Q":
        def f(a, b):
            result = a @ b
            return result, result

        res_x = jax.lax.scan(f, self.nodes, self.nodes[0][None])[1]
        res_e = jax.lax.scan(f, self.edges, self.edges[0][None])[1]
        return Q(nodes=res_x[0], edges=res_e[0])

    def apply_to(self, g: GraphDistribution) -> DenseGraphDistribution:
        return DenseGraphDistribution.create(
            nodes=g.nodes,
            edges=g.edges @ self.edges[:, None],
            edges_mask=g.edges_mask,
            nodes_mask=g.nodes_mask,
        )


@jaxtyped
@beartype
@dataclass
class FeatureOnlyQ(Q):
    nodes: Float[Array, "t en en"]
    edges: Float[Array, "t ee ee"]

    # overrides the square bracket indexing
    @jaxtyped
    def __getitem__(self, key) -> "Q":
        return Q(nodes=self.nodes[key], edges=self.edges[key])

    @property
    def shape(self) -> dict[str, tuple[int, ...]]:
        return dict(x=self.nodes.shape, edges=self.edges.shape)

    def __truediv__(self, other: Union[Float[Array, " "], "Q"]) -> "Q":
        if isinstance(other, Q):
            return Q(nodes=self.nodes / other.nodes, edges=self.edges / other.edges)
        else:
            return Q(nodes=self.nodes / other, edges=self.edges / other)

    def __matmul__(self, other: "Q") -> "Q":
        return Q(nodes=self.nodes @ other.nodes, edges=self.edges @ other.edges)

    def __len__(self) -> int:
        return self.nodes.shape[0]

    def cumulative_matmul(self) -> "Q":
        def f(a, b):
            result = a @ b
            return result, result

        res_x = jax.lax.scan(f, self.nodes, self.nodes[0][None])[1]
        res_e = jax.lax.scan(f, self.edges, self.edges[0][None])[1]
        return Q(nodes=res_x[0], edges=res_e[0])

    def apply_to(self, g: GraphDistribution) -> DenseGraphDistribution:
        return DenseGraphDistribution.create(
            nodes=g.nodes @ self.nodes,
            edges=g.edges @ self.edges[:, None],
            edges_mask=g.edges_mask,
            nodes_mask=g.nodes_mask,
        )
