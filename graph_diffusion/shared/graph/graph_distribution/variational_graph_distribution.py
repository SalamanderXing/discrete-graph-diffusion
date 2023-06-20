from jax import numpy as np
from jax import random, Array
import jax
from jaxtyping import Float, Int
from .graph_distribution import (
    GraphDistribution,
    pseudo_assert,
    NodeDistribution,
    EdgeDistribution,
    EdgeCountType,
    SBool,
)
from mate.jax import typed, Key
import ipdb

# TODO: idea: make the absence of node/edge as a special case -- when model is too uncertain about any other class, that should be decoded as "no node/edge"


class VariationalGraphDistribution(GraphDistribution):
    @classmethod
    @typed
    def create(
        cls,
        nodes: NodeDistribution,
        edges: EdgeDistribution,
        edges_counts: EdgeCountType,
        nodes_counts: EdgeCountType,
        _safe: SBool = True,
    ) -> "VariationalGraphDistribution":
        # is_nodes_dist = is_dist(nodes) or (not _safe)
        # if not is_nodes_dist:
        #    ipdb.set_trace()
        # pseudo_assert(is_nodes_dist)
        # is_edges_dist = is_dist(edges) or (not _safe)
        # if not is_nodes_dist:
        #     ipdb.set_trace()

        # pseudo_assert(is_edges_dist)

        # makes sure that the edges are symmetric
        pseudo_assert((edges == edges.transpose((0, 2, 1, 3))).all())
        # checks that the edges are not self-loops
        pseudo_assert((edges.argmax(-1).diagonal(axis1=1, axis2=2) == 0).all())

        return cls(
            nodes=nodes,
            edges=edges,
            edges_counts=edges_counts,
            nodes_counts=nodes_counts,
            _created_internally=True,
        )

    @classmethod
    @typed
    def from_graph_ditribution(
        cls, g: GraphDistribution
    ) -> "VariationalGraphDistribution":
        return cls.create(
            nodes=g.nodes,
            edges=g.edges,
            edges_counts=g.edges_counts,
            nodes_counts=g.nodes_counts,
        )

    @staticmethod
    @typed
    def encode_single(value: Array) -> Array:
        return value * 2 - 1

    @typed
    def encode(self) -> "EncodedGraphDistribution":
        return EncodedGraphDistribution.create(
            nodes=VariationalGraphDistribution.encode_single(self.nodes),
            edges=VariationalGraphDistribution.encode_single(self.edges),
            edges_counts=self.edges_counts,
            nodes_counts=self.nodes_counts,
        )

    @typed
    def logprob(self, z: "EncodedGraphDistribution", g_0) -> Float[Array, "b"]:
        logprob_nodes = self.__logprob_nodes(self.nodes, z.nodes, g_0)
        logprob_edges = self.__logprob_edges(self.edges, z.edges, g_0)
        return logprob_nodes + logprob_edges

    @property
    def node_vocab_size(self):
        return self.nodes.shape[-1]

    @property
    def edge_vocab_size(self):
        return self.edges.shape[-1]

    @typed
    def __decode_node(
        self, z: NodeDistribution, g_0: Float[Array, "k"]
    ) -> Float[Array, "b n node_vocab_size node_vocab_size"]:
        # z.shape = (b, n , 1)
        # x_vals_raw.shape (b, n, vocab_size)
        x_vals = VariationalGraphDistribution.encode_single(
            (np.arange(self.node_vocab_size) / self.node_vocab_size)
        )[None, None, None, :]
        # Logits are exact if there are no dependencies between dimensions of x
        inv_stdev = np.exp(-0.5 * g_0)
        logits = -0.5 * np.square((z[..., None] - x_vals) * inv_stdev)
        logprobs = jax.nn.log_softmax(logits)
        return logprobs

    @typed
    def __decode_edge(
        self, z: EdgeDistribution, g_0: Float[Array, "k"]
    ) -> Float[Array, "b n n edge_vocab_size edge_vocab_size"]:
        # z.shape = (b, n , 1)
        # x_vals_raw.shape (b, n, vocab_size)
        x_vals = VariationalGraphDistribution.encode_single(
            (np.arange(self.edge_vocab_size) / self.edge_vocab_size)
        )[None, None, None, None, :]
        # Logits are exact if there are no dependencies between dimensions of x
        inv_stdev = np.exp(-0.5 * g_0)
        logits = -0.5 * np.square((z[..., None] - x_vals) * inv_stdev)
        logprobs = jax.nn.log_softmax(logits)
        return logprobs

    @typed
    def __logprob_edges(self, x: EdgeDistribution, z: EdgeDistribution, g_0) -> Float[Array, "b"]:
        x_unscaled = ((x + 1) / 2).astype("int32")
        #x_onehot = jax.nn.one_hot(x_unscaled, self.edge_vocab_size)
        logprobs = self.__decode_edge(z, g_0)
        # logprob = np.einsum(
        #     "bijkl -> b",
        #     x_onehot * logprobs,
        # )
        # return logprob
        return logprobs.min(-1).mean(-1).sum(-1).sum(-1)


    @typed
    def __logprob_nodes(self, x: NodeDistribution, z: NodeDistribution, g_0):  # -> Float[Array, "b"]:
        x_unscaled = ((x + 1) / 2).astype("int32")
        x_onehot = jax.nn.one_hot(x_unscaled, self.node_vocab_size)
        logprobs = self.__decode_node(z, g_0)
        masked = (x_onehot * logprobs).min(-1).mean(-1).sum(-1)
        # logprob = np.einsum("bijk -> b", masked)
        return masked  # logprob


class EncodedGraphDistribution(GraphDistribution):
    @classmethod
    @typed
    def create(
        cls,
        nodes: NodeDistribution,
        edges: EdgeDistribution,
        edges_counts: EdgeCountType,
        nodes_counts: EdgeCountType,
        _safe: SBool = True,
    ) -> "EncodedGraphDistribution":
        # is_nodes_dist = is_dist(nodes) or (not _safe)
        # if not is_nodes_dist:
        #    ipdb.set_trace()
        # pseudo_assert(is_nodes_dist)
        # is_edges_dist = is_dist(edges) or (not _safe)
        # if not is_nodes_dist:
        #     ipdb.set_trace()

        # pseudo_assert(is_edges_dist)

        pseudo_assert((edges == edges.transpose((0, 2, 1, 3))).all())
        return cls(
            nodes=nodes,
            edges=edges,
            edges_counts=edges_counts,
            nodes_counts=nodes_counts,
            _created_internally=True,
        )

    @typed
    def noise_like(self, key: Key) -> "EncodedGraphDistribution":
        edge_noise = random.normal(key, self.edges.shape)
        edge_noise = GraphDistribution.to_symmetric(edge_noise)
        edge_noise = np.where(np.eye(self.n)[None, :, :, None], 0, edge_noise)
        return EncodedGraphDistribution.create(
            nodes=random.normal(key, self.nodes.shape),
            edges=edge_noise,
            nodes_counts=self.nodes_counts,
            edges_counts=self.edges_counts,
        )

    @classmethod
    @typed
    def noise(
        cls,
        key: Key,
        num_node_features: int,
        num_edge_features: int,
        num_nodes: int,
    ) -> "EncodedGraphDistribution":
        batch_size = 2
        nodes_counts = np.ones(batch_size, int)
        edges_counts = np.ones(batch_size, int)
        edges = random.normal(
            key, (batch_size, num_nodes, num_nodes, num_edge_features)
        )
        edges = GraphDistribution.to_symmetric(edges)
        # sets the diagonal to 0
        edges = np.where(np.eye(num_nodes)[None, :, :, None], 0, edges)
        return cls.create(
            nodes=random.normal(key, (batch_size, num_nodes, num_node_features)),
            edges=edges,
            nodes_counts=nodes_counts,
            edges_counts=edges_counts,
        )

    def decode(self) -> "VariationalGraphDistribution":
        return VariationalGraphDistribution.create(
            nodes=(self.nodes + 1) / 2,
            edges=(self.edges + 1) / 2,
            edges_counts=self.edges_counts,
            nodes_counts=self.nodes_counts,
        )

    def decode_no_probs(self) -> "VariationalGraphDistribution":
        return VariationalGraphDistribution.create(
            nodes=(self.nodes + 1) / 2,
            edges=(self.edges + 1) / 2,
            edges_counts=self.edges_counts,
            nodes_counts=self.nodes_counts,
        )
