import jax.numpy as jnp
import flax.linen as nn
import jax
import ipdb
from mate.jax import typed, Key
from jaxtyping import Array, Float, Bool
from flax.core.frozen_dict import FrozenDict
from ...shared.graph import Nodes, Edges, Graph


class GATLayer(nn.Module):
    c_out: int  # Dimensionality of output features
    num_heads: int  # Number of heads, i.e. attention mechanisms to apply in parallel.
    num_nodes: int
    concat_heads: bool = True  # If True, the output of the different heads is concatenated instead of averaged.
    alpha: float = 0.2  # Negative slope of the LeakyReLU activation.
    dropout_prob: float = 0.2

    def setup(self):
        if self.concat_heads:
            assert (
                self.c_out % self.num_heads == 0
            ), "Number of output features must be a multiple of the count of heads."
            c_out_per_head = self.c_out // self.num_heads
        else:
            c_out_per_head = self.c_out

        # Sub-modules and parameters needed in the layer
        self.projection = nn.Dense(
            c_out_per_head * self.num_heads,
            kernel_init=nn.initializers.glorot_uniform(),
        )
        self.a = self.param(
            "a", nn.initializers.glorot_uniform(), (self.num_heads, 2 * c_out_per_head)
        )  # One per head
        self.to_adj_dense = nn.Dense(
            self.num_nodes, kernel_init=nn.initializers.glorot_uniform()
        )
        self.dropout = nn.Dropout(self.dropout_prob)

    def to_adj(self, edges: Edges, deterministic: bool) -> Edges:
        edges_reshaped = edges.reshape((edges.shape[0] * edges.shape[1], -1))
        edges_reshaped = self.dropout(
            nn.sigmoid(self.to_adj_dense(edges_reshaped)), deterministic
        )
        edges_reshaped = edges_reshaped.reshape(edges.shape)
        edges_reshaped = (edges_reshaped + edges_reshaped.transpose((0, 2, 1))) / 2
        edges_reshaped = (edges_reshaped > 0.5).astype(float)
        # symmetrize the matrix
        return edges_reshaped

    def __call__(self, node_feats, edges, deterministic: bool = False):
        """
        Inputs:
            node_feats - Input features of the node. Shape: [batch_size, c_in]
            adj_matrix - Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
        """
        adj_matrix = self.to_adj(edges, deterministic)
        # ipdb.set_trace()
        batch_size, num_nodes = node_feats.shape[0], node_feats.shape[1]

        # Apply linear layer and sort nodes by head
        node_feats = self.dropout(self.projection(node_feats), deterministic)
        node_feats = node_feats.reshape((batch_size, num_nodes, self.num_heads, -1))

        # We need to calculate the attention logits for every edge in the adjacency matrix
        # In order to take advantage of JAX's just-in-time compilation, we should not use
        # arrays with shapes that depend on e.g. the number of edges. Hence, we calculate
        # the logit for every possible combination of nodes. For efficiency, we can split
        # a[Wh_i||Wh_j] = a_:d/2 * Wh_i + a_d/2: * Wh_j.
        # print(f"{node_feats.shape=}")
        # print(f"{self.a[None, None, :, : self.a.shape[0] // 2].shape}")
        # print(f"{self.a[None, None, :, self.a.shape[0] // 2 :].shape}")
        # print(f"{self.a.shape=}")
        logit_parent = (node_feats * self.a[None, None, :, : self.a.shape[0] // 2]).sum(
            axis=-1
        )
        logit_child = (node_feats * self.a[None, None, :, self.a.shape[0] // 2 :]).sum(
            axis=-1
        )
        attn_logits = logit_parent[:, :, None, :] + logit_child[:, None, :, :]
        attn_logits = nn.leaky_relu(attn_logits, self.alpha)
        attn_logits = self.dropout(attn_logits, deterministic)

        # Mask out nodes that do not have an edge between them
        attn_logits = jnp.where(
            adj_matrix[..., None] == 1.0,
            attn_logits,
            jnp.ones_like(attn_logits) * (-9e15),
        )

        # Weighted average of attention
        attn_probs = nn.softmax(attn_logits, axis=2)
        # if print_attn_probs:
        #     print("Attention probs\n", attn_probs.transpose(0, 3, 1, 2))
        node_feats = jnp.einsum("bijh,bjhc->bihc", attn_probs, node_feats)

        # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(axis=2)

        return node_feats


from mate.jax import typed, Key


class GAT(nn.Module):
    n_heads_per_layer: tuple[int, ...]
    hidden_node_features: int = 64
    hidden_edge_features: int = 64

    @classmethod
    @typed
    def create(
        cls,
        key: Key,
        n: int,
        n_heads_per_layer: tuple[int, ...],
        hidden_node_features: int = 64,
        hidden_edge_features: int = 64,
    ) -> tuple[nn.Module, FrozenDict]:
        model = cls(
            n_heads_per_layer=n_heads_per_layer,
            hidden_node_features=hidden_node_features,
            hidden_edge_features=hidden_edge_features,
        )

        key_nodes, key_edges = jax.random.split(key, num=2)
        nodes_shape = (2, n)
        edges_shape = (2, n, n)
        nodes = jax.random.normal(key_nodes, nodes_shape)
        edges = jax.random.normal(key_edges, edges_shape)

        params = model.init(
            key,
            nodes,
            edges,
        )
        return model, params

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
        n_heads_per_layer = (num_heads,) * num_layers
        model = cls(
            n_heads_per_layer=n_heads_per_layer,
            hidden_node_features=hidden_node_features,
            hidden_edge_features=hidden_edge_features,
        )

        key_nodes, key_edges = jax.random.split(key, num=2)
        nodes_shape = (batch_size, n, in_node_features)
        edges_shape = (batch_size, n, n, in_edge_features)
        nodes = jax.random.normal(key_nodes, nodes_shape)
        edges = jax.random.normal(key_edges, edges_shape).argmax(-1).astype(float)
        node_mask = jnp.ones((batch_size, n), dtype=bool)
        print(f"[orange]Init nodes[/orange]: {nodes.shape}")
        print(f"Init edges: {edges.shape}")
        graph = Graph.create(
            nodes=nodes,
            edges=edges,
            edges_counts=jnp.ones(batch_size, dtype=int),
            nodes_counts=jnp.ones(batch_size, dtype=int),
        )
        params = model.init(
            key,
            nodes,
            edges,
        )
        return model, params

    @nn.compact
    def __call__(
        self,
        g: Graph,
        deterministic: bool = True,
    ) -> Graph:
        nodes = g.nodes
        edges = g.edges
        n_nodes = nodes.shape[1]

        for i in range(2):
            # print(f"{nodes.shape=} {edges.shape=}")
            nodes = GATLayer(c_out=8, num_heads=4, num_nodes=n_nodes)(
                nodes, edges, deterministic=deterministic
            )
            # nodes = n.relu(out_nodes)
            edge_mapping = GATLayer(c_out=n_nodes, num_heads=1, num_nodes=n_nodes)
            edges = jax.vmap(
                lambda x: edge_mapping(x, edges, deterministic=deterministic).squeeze(
                    -1
                ),
                1,
                1,
            )(edges)
            edges = (edges + jnp.swapaxes(edges, 1, 2)) / 2
            # edges = nn.relu(edges)
        nodes = nodes.mean(-1)[..., None]
        # edges_reshaped = edges.reshape(edges.shape[0] * n_nodes, -1)
        # adj = (edges == 0).astype(jnp.float32)

        # nodes = nn.Dense(64)(
        #     nodes.reshape((nodes.shape[0] * nodes.shape[1], -1))
        # ).reshape((edges.shape[0], n_nodes, 64))
        # for i, n_heads in enumerate(self.n_heads_per_layer):
        #     nodes = GATLayer(
        #         num_heads=n_heads,
        #         out_features=self.hidden_node_features * n_heads,
        #         n_nodes=n_nodes,
        #     )(nodes, edges)
        #     # edges = GATLayer(
        #     #     num_heads=n_heads,
        #     #     out_features=self.hidden_edge_features * n_heads,
        #     # )(edges, nodes)
        #     edges = GATLayer(num_heads=n_heads, out_features=n_nodes, n_nodes=n_nodes)(
        #         nodes, edges
        #     ).squeeze(-1)
        #     edges = jax.nn.sigmoid(
        #         edges
        #     )  # Apply sigmoid to get values in the range [0, 1]
        #     edges = (
        #         edges > 0.5
        #     ).float()  # Threshold at 0.5 to get binary adjacency matrix

        # nodes = nn.Dense(self.out_node_features)(nodes)
        # edges = nn.Dense(self.out_edge_features)(edges)
        nodes = jax.nn.tanh(edges)
        edges = jax.nn.tanh(edges)
        # symmetrize the out_edges
        return Graph.create(
            nodes, edges, edges_counts=g.edges_counts, nodes_counts=g.nodes_counts
        )
