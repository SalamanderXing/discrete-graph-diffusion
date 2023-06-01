import jax
from jax import numpy as np
from flax import linen as nn
from mate.jax import typed, SInt, SFloat
from jaxtyping import Float, Bool
from jax import Array
import ipdb


class GraphAttentionHead(nn.Module):
    in_node_features: int
    in_edge_features: int
    out_features: int
    edge_threshold: float = 0.5

    def setup(self):
        self.W_node = nn.Dense(
            self.out_features,
            use_bias=False,
            kernel_init=jax.nn.initializers.xavier_uniform(),
        )
        self.W_edge = nn.Dense(
            self.out_features,
            use_bias=False,
            kernel_init=jax.nn.initializers.xavier_uniform(),
        )
        self.a = nn.Dense(
            1, use_bias=False, kernel_init=jax.nn.initializers.xavier_uniform()
        )

    @typed
    def __call__(
        self,
        nodes: Float[Array, "b n en"],
        edges: Float[Array, "b n n ee"],
        node_mask: Bool[Array, "b n"],
    ):
        batch_size, n = nodes.shape[0], nodes.shape[1]
        nodes = self.W_node(nodes)
        edges = self.W_edge(edges)

        row = np.repeat(nodes, n, axis=1).reshape(batch_size, n, n, -1)
        col = np.repeat(nodes, n, axis=2).reshape(batch_size, n, n, -1)

        attention_input = np.concatenate([row, edges, col], axis=-1)
        attention_weights = jax.nn.leaky_relu(
            self.a(attention_input), negative_slope=0.2
        )

        attention_weights = attention_weights * node_mask[..., None, None]

        adj_matrix = jax.nn.sigmoid(attention_weights) > self.edge_threshold
        attention_weights = (
            jax.nn.softmax(attention_weights, axis=2) * adj_matrix
        ).squeeze(-1)
        h_prime = np.matmul(attention_weights, nodes)
        return h_prime
