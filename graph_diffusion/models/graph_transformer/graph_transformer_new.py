from .graph_transformer import GraphTransformer, DDense
from ...shared.graph.graph_distribution import (
    EncodedGraphDistribution,
    VariationalGraphDistribution,
)
import ipdb
import flax.linen as nn
from jaxtyping import Float
import jax.numpy as np
import jax
from flax.core.frozen_dict import FrozenDict
from mate.jax import typed, Key
from jax import Array
from rich import print
import einops as e
from jaxtyping import Float, Array


class Conv(nn.Module):
    @nn.compact
    def __call__(self, x: Array, deterministic):
        to_add_features_edges = e.rearrange(x, "b n1 n2 ee -> b ee n1 n2")
        res = nn.Conv(features=10, kernel_size=(3, 3))(x)
        res = nn.Dropout(0.2)(res, deterministic)
        res = nn.relu(res)
        res = nn.Conv(features=1, kernel_size=(3, 3))(res)
        res = nn.Dropout(0.2)(res, deterministic)
        res = nn.sigmoid(res)
        return res


# def compute_spectral_features(adj_matrix):
#     degree = compute_degree(adj_matrix)
#     laplacian = np.diag(degree) - adj_matrix
#     return np.linalg.eigvals(laplacian)
#
#
# def compute_degree(adj_matrix):
#     return np.sum(adj_matrix, axis=-1)


def compute_degree(adj_matrix):
    return np.sum(adj_matrix, axis=-1)


def compute_spectral_features_single(adj_matrix):
    degree = compute_degree(adj_matrix)
    laplacian = np.diag(degree) - adj_matrix
    return np.linalg.eigh(laplacian)


def compute_spectral_features(edges):
    alls = (edges.argmax(axis=-1) > 0).astype(float)
    comp = jax.vmap(compute_spectral_features_single)
    v, ev = comp(alls)
    return nn.sigmoid(v[..., None]), nn.sigmoid(ev[..., None])


class GraphTransformerNew(nn.Module):
    depth: int
    edge_dim: int = -1
    dim_head: int = 64
    heads: int = 8
    gate_residual: bool = False
    with_feedforward: bool = False
    norm_edges: bool = False

    @typed
    @nn.compact
    def __call__(
        self, g: EncodedGraphDistribution, embedding: Array, deterministic: bool = False
    ) -> EncodedGraphDistribution:
        embedding_to_nodes = nn.Sequential((DDense(5), nn.tanh))
        embedding_to_edges = nn.Sequential((DDense(5), nn.tanh))
        nodes_embedding = embedding_to_nodes(embedding, deterministic)
        edges_embedding = embedding_to_edges(embedding, deterministic)
        n = g.nodes.shape[1]
        nodes_embedding = e.repeat(nodes_embedding, "b ten -> b n ten", n=n)
        edges_embedding = e.repeat(edges_embedding, "b tee -> b n1 n2 tee", n1=n, n2=n)

        mask = g.node_mask()

        spec_nodes, spec_edges = compute_spectral_features(g.edges)
        spec_nodes = nn.tanh(nn.Dense(5)(spec_nodes))
        spec_edges = nn.tanh(nn.Dense(5)(spec_edges))
        conv_features = Conv()(g.edges, deterministic)
        nodes = np.concatenate([g.nodes, nodes_embedding, spec_nodes], axis=-1)
        edges = np.concatenate(
            [g.edges, edges_embedding, spec_edges, conv_features], axis=-1
        )
        new_nodes, new_edges = GraphTransformer(
            depth=self.depth,
            edge_dim=self.edge_dim,
            dim_head=self.dim_head,
            heads=self.heads,
            gate_residual=self.gate_residual,
            with_feedforward=self.with_feedforward,
            norm_edges=self.norm_edges,
        )(
            nodes,
            edges,
            mask,
            edge_time_embedding=np.zeros((nodes.shape[0], n, n, 0)),
            node_time_embedding=np.zeros((nodes.shape[0], n, 0)),
            deterministic=deterministic,
        )
        new_nodes = nn.Dense(g.nodes.shape[-1])(new_nodes)
        new_edges = nn.Dense(g.edges.shape[-1])(new_edges)
        # symmetrize the edges
        new_edges = EncodedGraphDistribution.to_symmetric(new_edges)
        new_nodes = jax.nn.softmax(new_nodes, axis=-1)
        new_edges = jax.nn.softmax(new_edges, axis=-1)
        return EncodedGraphDistribution.create(
            nodes=(new_nodes * 2) - 1,
            edges=(new_edges * 2) - 1,
            edges_counts=g.edges_counts,
            nodes_counts=g.nodes_counts,
        )

    @classmethod
    @typed
    def initialize(
        cls,
        key: Key,
        number_of_nodes: int,
        in_node_features: int,
        in_edge_features: int,
        out_node_features: int = -1,
        out_edge_features: int = -1,
        num_layers: int = 3,
    ) -> tuple[nn.Module, FrozenDict]:
        out_node_features = (
            out_node_features if out_node_features > 0 else in_node_features
        )
        out_edge_features = (
            out_edge_features if out_edge_features > 0 else in_edge_features
        )
        model = cls(
            depth=num_layers,
        )

        key_nodes, key_edges = jax.random.split(key, num=2)
        dummy_graph = EncodedGraphDistribution.noise(
            key_nodes,
            num_node_features=in_node_features,
            num_edge_features=in_edge_features,
            num_nodes=number_of_nodes,
        )
        # print(f"[orange]Init nodes[/orange]: {nodes.shape}")
        # print(f"Init edges: {edges.shape}")
        params = model.init(
            key,
            dummy_graph,
            np.zeros((2, 129)),
            deterministic=True,
        )
        return model, params
