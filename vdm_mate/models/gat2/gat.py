import jax
import jax.numpy as jnp
import flax.linen as nn
from jaxtyping import Array, Float, Bool
from mate.jax import typed, SInt, SFloat, Key
from flax.core.frozen_dict import FrozenDict
import ipdb
from ...shared.graph import Graph


class GraphAttentionLayer(nn.Module):
    out_node_features: int
    out_edge_features: int
    num_heads: int = 8
    dropout_rate: float = 0.1

    def setup(self):
        self.node_linear = nn.Dense(self.out_node_features * self.num_heads)
        self.edge_linear = nn.Dense(self.out_edge_features * self.num_heads)
        self.attention_linear = nn.Dense(self.num_heads, use_bias=False)
        self.attention_linear_nodes = nn.Dense(self.num_heads, use_bias=False)
        self.attention_linear_edges = nn.Dense(self.num_heads, use_bias=False)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.batch_norm_edges = nn.BatchNorm()
        self.batch_norm_nodes = nn.BatchNorm()

    @typed
    def __call__(
        self,
        nodes: Float[Array, "b n en"],
        edges: Float[Array, "b n n ee"],
        node_mask: Bool[Array, "b n"],
        deterministic: bool = False,
    ) -> tuple[Float[Array, "b n enx"], Float[Array, "b n n eex"]]:
        # nodes = self.batch_norm_nodes(nodes, use_running_average=not deterministic)
        # edges = self.batch_norm_edges(edges, use_running_average=not deterministic)

        node_features = self.dropout(
            self.node_linear(nodes), deterministic=deterministic
        )
        edge_features = self.dropout(
            self.edge_linear(edges), deterministic=deterministic
        )

        # Compute attention scores for nodes
        attention_node_query = self.dropout(
            self.attention_linear(node_features), deterministic=deterministic
        )
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
            self.dropout(
                self.attention_linear(edge_features), deterministic=deterministic
            ),
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
        result_edges = jax.nn.elu(updated_edges).reshape(
            edges.shape[0], edges.shape[1], edges.shape[2], -1
        )
        # symmetrise the edges
        result_edges = (result_edges + jnp.swapaxes(result_edges, 1, 2)) / 2

        return (jax.nn.elu(updated_nodes), result_edges)


class GAT(nn.Module):
    out_node_features: int
    out_edge_features: int
    hidden_node_features: int = 64
    hidden_edge_features: int = 64
    num_layers: int = 2
    num_heads: int = 2
    dropout_rate: float = 0.1
    in_dense: bool = False

    def setup(self):
        self.in_dense_edges = nn.Dense(100) if self.in_dense else lambda x: x
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        layers = []
        layers.append(
            GraphAttentionLayer(
                out_node_features=self.hidden_node_features,
                out_edge_features=self.hidden_edge_features,
            )
        )

        for _ in range(self.num_layers - 1):
            layers.append(
                GraphAttentionLayer(
                    out_edge_features=self.hidden_edge_features,
                    out_node_features=self.hidden_node_features,
                )
            )

        self.att_layers = layers
        self.final_node_layer = nn.Dense(self.out_node_features)
        self.final_edge_layer = nn.Dense(self.out_edge_features)

    @typed
    def __call__(
        self,
        # nodes: Float[Array, "b n en"],
        # edges: Float[Array, "b n n ee"],
        # node_mask: Bool[Array, "b n"],
        graph: Graph,
        deterministic: bool = False,
    ) -> Graph:  # tuple[Float[Array, "b n en"], Float[Array, "b n n ee"]]:
        nodes = graph.nodes
        edges = graph.edges[..., None]
        # new_e
        edges = jax.vmap(
            lambda x: self.in_dense_edges(x),
            1,
            1,
        )(edges)
        edges = (edges + jnp.swapaxes(edges, 1, 2)) / 2
        edges = self.dropout(nn.relu(edges), deterministic)

        # node_mask = nodes[:, :, -1] == 0
        node_mask = graph.node_mask.mean(-1).astype(bool)
        for att_layer in self.att_layers:
            nodes, edges = att_layer(
                nodes, edges, node_mask=node_mask, deterministic=deterministic
            )

        out_nodes = nn.tanh(self.final_node_layer(nodes))
        out_edges = nn.tanh(self.final_edge_layer(edges)).squeeze(-1)
        return Graph.create(
            nodes=out_nodes,
            edges=out_edges,
            edges_counts=graph.edges_counts,
            nodes_counts=graph.nodes_counts,
        )

    @classmethod
    @typed
    def initialize(
        cls,
        key: Key,
        batch_size: int,
        n: int,
        in_node_features: int,
        in_edge_features: int,
        out_node_features: int = -1,
        out_edge_features: int = -1,
        hidden_node_features: int = 64,
        hidden_edge_features: int = 64,
        num_layers: int = 2,
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
        nodes_shape = (batch_size, n, in_node_features)
        edges_shape = (batch_size, n, n, in_edge_features)
        nodes = jax.random.normal(key_nodes, nodes_shape)
        edges = jax.random.normal(key_edges, edges_shape)
        node_mask = jnp.ones((batch_size, n), dtype=bool)

        params = model.init(
            key,
            nodes,
            edges,
            # node_mask,
        )
        return model, params
