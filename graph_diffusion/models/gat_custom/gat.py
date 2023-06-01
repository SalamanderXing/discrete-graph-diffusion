import jax
import jax.numpy as jnp
import flax.linen as nn
from jaxtyping import Array, Float, Bool
from mate.jax import typed, SInt, SFloat, Key, SBool
from flax.core.frozen_dict import FrozenDict
from rich import print
import ipdb
from jax import jit
from functools import partial
from ...shared.graph_distribution import GraphDistribution


class DenseWithDropout(nn.Module):
    out_features: int
    use_bias: bool = True
    dropout_rate: float = 0.2

    def setup(self):
        self.dense = nn.Dense(self.out_features, self.use_bias)
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x, deterministic: SBool):
        x = self.dense(x)
        x = self.dropout(x, deterministic)
        return x


class GraphAttentionLayer(nn.Module):
    in_node_features: int
    in_edge_features: int
    out_node_features: int
    out_edge_features: int
    num_heads: int = 8
    dropout_rate: float = 0.1

    def setup(self):
        self.node_linear = DenseWithDropout(self.out_node_features * self.num_heads)
        self.edge_linear = DenseWithDropout(self.out_edge_features * self.num_heads)
        self.attention_linear = DenseWithDropout(self.num_heads, use_bias=False)
        self.attention_linear_nodes = DenseWithDropout(self.num_heads, use_bias=False)
        self.attention_linear_edges = DenseWithDropout(self.num_heads, use_bias=False)

    @typed
    def __call__(
        self,
        nodes: Float[Array, "b n en"],
        edges: Float[Array, "b n n ee"],
        node_mask: Bool[Array, "b n"],
        deterministic: SBool,
    ) -> tuple[Float[Array, "b n enx"], Float[Array, "b n n eex"]]:
        node_features = self.node_linear(nodes, deterministic)
        edge_features = self.edge_linear(edges, deterministic)

        # Compute attention scores for nodes
        attention_node_query = self.attention_linear(node_features, deterministic)
        attention_scores_nodes = jnp.einsum(
            "bik,bjk->bij", attention_node_query, attention_node_query
        )
        attention_scores_nodes = attention_scores_nodes / jnp.sqrt(
            self.out_node_features
        )

        # Compute attention scores for edges
        attention_scores_edges = jnp.einsum(
            "bik,bijk->bij",
            attention_node_query,
            self.attention_linear(edge_features, deterministic),
        )
        attention_scores_edges = attention_scores_edges / jnp.sqrt(
            self.out_edge_features
        )

        attention_weights_nodes = jax.nn.softmax(
            attention_scores_nodes * node_mask[..., None], axis=-1
        )
        attention_weights_edges = jax.nn.softmax(
            attention_scores_edges * node_mask[..., None], axis=-1
        )

        # Attention-weighted node features
        attention_weighted_nodes = jnp.einsum(
            "bij,bik->bjk", attention_weights_nodes, node_features
        )

        # Attention-weighted edge features
        attention_weighted_edges = jnp.einsum(
            "bij,bijk->bijk", attention_weights_edges, edge_features
        )

        # Combine edge features with node features
        aggregated_edge_features = jnp.einsum(
            "bij,bijk->bik", attention_weights_edges, edge_features
        )
        updated_nodes = (
            node_features + attention_weighted_nodes + aggregated_edge_features
        )

        # Combine node features with edge features
        aggregated_node_features = jnp.einsum(
            "bij,bik->bijk", attention_weights_nodes, node_features
        )
        updated_edges = (
            edge_features + attention_weighted_edges + aggregated_node_features
        )

        return jax.nn.elu(updated_nodes), jax.nn.elu(updated_edges).reshape(
            edges.shape[0], edges.shape[1], edges.shape[2], -1
        )


class GAT(nn.Module):
    in_node_features: int
    in_edge_features: int
    out_node_features: int
    out_edge_features: int
    hidden_node_features: int = 64
    hidden_edge_features: int = 64
    num_layers: int = 2
    num_heads: int = 8

    def setup(self):
        layers = []
        layers.append(
            GraphAttentionLayer(
                in_node_features=self.in_node_features,
                in_edge_features=self.in_edge_features,
                out_node_features=self.hidden_node_features,
                out_edge_features=self.hidden_edge_features,
            )
        )

        for _ in range(self.num_layers - 1):
            layers.append(
                GraphAttentionLayer(
                    in_node_features=self.hidden_node_features,
                    in_edge_features=self.hidden_edge_features,
                    out_edge_features=self.hidden_edge_features,
                    out_node_features=self.hidden_node_features,
                )
            )

        self.att_layers = layers
        self.final_node_layer = nn.Dense(self.out_node_features)
        self.final_edge_layer = nn.Dense(self.out_edge_features)
        self.to_nodes_cond = DenseWithDropout(
            self.hidden_node_features * self.num_heads
        )
        self.to_edges_cond = DenseWithDropout(
            self.hidden_edge_features * self.num_heads
        )

    @typed
    def __call__(
        self,
        g: GraphDistribution,
        temporal_embedding: Float[Array, "b t"],
        deterministic: SBool,
    ) -> GraphDistribution:
        nodes, edges = g.nodes, g.edges
        # jax.debug.print("YOOO {det}", det=deterministic)
        temporal_embedding_node = jnp.broadcast_to(
            nn.relu(self.to_nodes_cond(temporal_embedding, deterministic))[:, None],
            (
                nodes.shape[0],
                nodes.shape[1],
                self.hidden_node_features * self.num_heads,
            ),
        )
        temporal_embedding_edge = jnp.broadcast_to(
            nn.relu(self.to_edges_cond(temporal_embedding, deterministic))[
                :, None, None
            ],
            (
                edges.shape[0],
                edges.shape[1],
                edges.shape[2],
                self.hidden_edge_features * self.num_heads,
            ),
        )
        node_mask = nodes[:, :, -1] == 0
        for i, att_layer in enumerate(self.att_layers):
            nodes, edges = att_layer(nodes, edges, node_mask, deterministic)
            if i < self.num_layers - 1:
                nodes = nodes + temporal_embedding_node
                edges = edges + temporal_embedding_edge

        out_nodes = self.final_node_layer(nodes)
        out_edges = self.final_edge_layer(edges)
        return GraphDistribution.create(
            nodes=out_nodes,
            e=out_edges,
            nodes_counts=g.nodes_counts,
            edges_counts=g.edges_counts,
        )

    @classmethod
    @typed
    def initialize(
        cls,
        key: Key,
        n: int,
        in_node_features: int,
        in_edge_features: int,
        out_node_features: int = -1,
        out_edge_features: int = -1,
        hidden_node_features: int = 64,
        hidden_edge_features: int = 64,
        num_layers: int = 3,
        num_heads: int = 8,
    ) -> tuple[nn.Module, FrozenDict]:
        out_node_features = (
            out_node_features if out_node_features > 0 else in_node_features
        )
        out_edge_features = (
            out_edge_features if out_edge_features > 0 else in_edge_features
        )
        model = cls(
            in_node_features=in_node_features,
            in_edge_features=in_edge_features,
            out_node_features=out_node_features,
            out_edge_features=out_edge_features,
            hidden_node_features=hidden_node_features,
            hidden_edge_features=hidden_edge_features,
            num_layers=num_layers,
            num_heads=num_heads,
        )

        key_nodes, key_edges = jax.random.split(key, num=2)
        nodes_shape = (2, n, in_node_features)
        edges_shape = (2, n, n, in_edge_features)
        nodes = jax.random.normal(key_nodes, nodes_shape)
        edges = jax.random.normal(key_edges, edges_shape)
        # node_mask = jnp.ones((batch_size, n), dtype=bool)
        dummy_graph = GraphDistribution.create(
            nodes,
            edges,
            edges_counts=jnp.ones((2), dtype=int),
            nodes_counts=jnp.ones((2), dtype=int),
        )
        print(f"[orange]Init nodes[/orange]: {nodes.shape}")
        print(f"Init edges: {edges.shape}")
        params = model.init(
            key,
            dummy_graph,
            jnp.zeros((2, 129)),
            deterministic=True,
        )
        return model, params
