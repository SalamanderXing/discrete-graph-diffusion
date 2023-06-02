from typing import Callable, Optional, Iterable
from jax import Array
from mate.jax import typed
import chex
import flax
from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import ipdb
from mate.jax import typed, SInt
from jaxtyping import Array, Float
from ...shared.graph import Edges, Nodes, Graph


class EncoderDecoder(nn.Module):
    """Encoder and decoder."""

    edge_vocab_size: int
    node_vocab_size: int

    @typed
    def __call__(self, x: Graph, g_0: Float[Array, "b"]):
        # For initialization purposes
        h = self.encode(x)
        return self.decode(h, g_0)

    @typed
    def __encode_single(self, x, vocab_size: SInt):
        x = x.round()
        return 2 * ((x + 0.5) / vocab_size) - 1

    @typed
    def encode(self, x: Graph):
        nodes = x.x
        edges = x.e
        # This transforms x from discrete values (0, 1, ...)
        # to the domain (-1,1).
        # Rounding here just a safeguard to ensure the input is discrete
        # (although typically, x is a discrete variable such as uint8)
        return Graph.create(
            nodes=self.__encode_single(nodes, self.node_vocab_size),
            edges=self.__encode_single(edges, self.edge_vocab_size),
            nodes_counts=x.nodes_counts,
            edges_counts=x.edges_counts,
        )

    @typed
    def __decode_edge(self, z: Edges, g_0) -> Float[Array, "b n n edge_vocab_size"]:
        # Logits are exact if there are no dependencies between dimensions of x
        x_vals_raw = jnp.arange(0, self.edge_vocab_size)[None, None, None]
        # x_vals = jnp.repeat(x_vals, 3, 1)
        x_vals = self.__encode_single(x_vals_raw, self.edge_vocab_size)
        inv_stdev = jnp.exp(-0.5 * g_0[..., None])
        logits = -0.5 * jnp.square((z[..., None] - x_vals) * inv_stdev)
        logprobs = jax.nn.log_softmax(logits)
        return logprobs

    @typed
    def __decode_node(self, z: Nodes, g_0) -> Float[Array, "b n node_vocab_size"]:
        # z.shape = (b, n , 1)
        # x_vals_raw.shape (b, n, vocab_size)

        # Logits are exact if there are no dependencies between dimensions of x
        x_vals_raw = jnp.arange(0, self.node_vocab_size)[None, None]
        # x_vals = jnp.repeat(x_vals, 3, 1)
        x_vals = self.__encode_single(x_vals_raw, self.node_vocab_size)
        # .transpose([1, 0])[
        #     None, None, None, :, :
        # ]
        inv_stdev = jnp.exp(-0.5 * g_0[..., None])
        logits = -0.5 * jnp.square((z - x_vals) * inv_stdev)
        logprobs = jax.nn.log_softmax(logits)
        # jax.debug.breakpoint()
        return logprobs

    @typed
    def decode(self, z: Graph, g_0) -> Graph:
        zn = z.nodes
        ze = z.edges
        return Graph.create(
            nodes=self.__decode_node(zn, g_0),
            edges=self.__decode_edge(ze, g_0),
            nodes_counts=z.nodes_counts,
            edges_counts=z.edges_counts,
        )

    @typed
    def __logprob_nodes(self, x: Nodes, z: Nodes, g_0) -> Float[Array, "b"]:
        # x = x.round().astype("int32")
        # unscale x
        # x_unscaled = ((x + 0.5) * self.node_vocab_size).round().astype("int32")
        x_onehot = jax.nn.one_hot(x.squeeze(-1), self.node_vocab_size)
        logprobs = self.__decode_node(z, g_0)
        logprob = jnp.sum(x_onehot * logprobs, axis=(1, 2))
        # jax.debug.breakpoint()
        return logprob

    @typed
    def __logprob_edges(self, x: Edges, z: Edges, g_0) -> Float[Array, "b"]:
        # x = x.round().astype("int32")
        # x_unscaled = ((x + 0.5) * self.node_vocab_size).round().astype("int32")
        x_onehot = jax.nn.one_hot(x, self.edge_vocab_size)
        logprobs = self.__decode_edge(z, g_0)
        logprob = jnp.sum(x_onehot * logprobs, axis=(1, 2, 3))
        return logprob

    @typed
    def logprob(self, x: Graph, z: Graph, g_0) -> Float[Array, "b"]:
        xn = x.nodes
        xg = x.edges
        zn = z.nodes
        zg = z.edges
        logprob_nodes = self.__logprob_nodes(xn, zn, g_0)
        logprob_edges = self.__logprob_edges(xg, zg, g_0)
        return logprob_nodes + logprob_edges
