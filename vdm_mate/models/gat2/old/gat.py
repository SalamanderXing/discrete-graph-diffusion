import jax
from flax import linen as nn
from mate.jax import typed, SInt, SFloat, Key
from jaxtyping import Array, Float, Int, Bool
from .gat_layer import GraphAttentionHead
from flax.core.frozen_dict import FrozenDict
import ipdb
from jax import numpy as np


class GraphAttentionLayer(nn.Module):
    num_heads: int
    in_node_features: int
    in_edge_features: int

    def setup(self):
        self.att_layers = [
            GraphAttentionHead(
                in_node_features=self.in_node_features // self.num_heads,
                in_edge_features=self.in_edge_features // self.num_heads,
                out_features=self.in_node_features // self.num_heads,
            )
            for _ in range(self.num_heads)
        ]

    @typed
    def __call__(
        self,
        nodes: Float[Array, "b n en"],
        edges: Float[Array, "b n n ee"],
        node_mask: Bool[Array, "b n"],
    ) -> Float[Array, "b n en"]:
        proto = [
            self.att_layers[i](nodes, edges, node_mask) for i in range(self.num_heads)
        ]
        return np.concatenate(proto, axis=-1)


class GAT(nn.Module):
    in_node_features: int
    in_edge_features: int
    hidden_features: int
    out_node_features: int
    out_edge_features: int
    num_heads: int = 4
    num_layers: int = 2

    def setup(self):
        self.att_layers = [
            GraphAttentionLayer(
                num_heads=self.num_heads,
                in_node_features=self.in_node_features,
                in_edge_features=self.in_edge_features,
            )
            for _ in range(self.num_heads)
        ]
        self.final_node_layer = nn.Dense(self.out_node_features)
        self.final_edge_layer = nn.Dense(self.out_edge_features)

    @typed
    def __call__(
        self,
        nodes: Float[Array, "b n en"],
        edges: Float[Array, "b n n ee"],
        node_mask: Bool[Array, "b n"],
    ) -> tuple[Float[Array, "b n en"], Float[Array, "b n n ee"]]:

        for i, att_layer in enumerate(self.att_layers):
            if i > 0:
                nodes_with_timestep = jax.nn.elu(nodes)
            nodes_with_timestep = att_layer(
                nodes_with_timestep, edges, node_mask=node_mask
            )

        out_nodes = self.final_node_layer(nodes_with_timestep)
        out_edges = self.final_edge_layer(edges)
        return out_nodes, out_edges

    @typed
    @classmethod
    def initialize(
        cls,
        key: Key,
        batch_size: int,
        n: int,
        in_node_features: int,
        in_edge_features: int,
        hidden_features: int = 128,
        num_heads: int = 8,
    ) -> tuple[nn.Module, FrozenDict]:
        model = cls(
            in_node_features=in_node_features,
            in_edge_features=in_edge_features,
            hidden_features=hidden_features,
            out_node_features=in_node_features,
            out_edge_features=in_edge_features,
            num_heads=num_heads,
        )

        key_nodes, key_edges = jax.random.split(key, num=2)
        nodes_shape = (batch_size, n, in_node_features)
        edges_shape = (batch_size, n, n, in_edge_features)
        nodes = jax.random.normal(key_nodes, nodes_shape)
        edges = jax.random.normal(key_edges, edges_shape)
        node_mask = np.ones((batch_size, n), dtype=bool)

        params = model.init(
            key,
            nodes,
            edges,
            node_mask,
        )
        return model, params
