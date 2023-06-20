import jax
from jaxtyping import Array, Float, Int, Bool
from jax import numpy as np
from jax import random
from flax import linen as nn
import ipdb
from flax.core.frozen_dict import FrozenDict
from typing import Callable
from mate.jax import typed, SBool

from einops import rearrange, repeat


class DDense(nn.Module):
    out_features: int
    use_bias: bool = True
    dropout_rate: float = 0.3

    def setup(self):
        self.dense = nn.Dense(self.out_features, self.use_bias)
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x, deterministic: SBool):
        x = self.dense(x)
        x = self.dropout(x, deterministic)
        return x


class PreNorm(nn.Module):
    fn: Callable

    @typed
    @nn.compact
    def __call__(self, x: Array, *args, **kwargs) -> Array:
        x = nn.LayerNorm()(x)
        return self.fn(x, *args, **kwargs)


class Residual(nn.Module):
    @typed
    @nn.compact
    def __call__(self, x: Array, res: Array) -> Array:
        return x + res


class GatedResidual(nn.Module):
    @typed
    @nn.compact
    def __call__(self, x: Array, res: Array, deterministic: SBool) -> Array:
        gated_input = np.concatenate((x, res, x - res), axis=-1)
        gate = nn.sigmoid(DDense(1, use_bias=False)(gated_input, deterministic))
        return x * gate + res * (1 - gate)


class Attention(nn.Module):
    dim_head: int = 64
    heads: int = 8

    @nn.compact
    @typed
    def __call__(
        self,
        nodes: Float[Array, "b n ne"],
        edges: Float[Array, "b n n ee"],
        mask: Bool[Array, "b k"],
        deterministic: SBool,
    ) -> Array:
        h = self.heads
        inner_dim = self.dim_head * self.heads
        scale = self.dim_head**-0.5
        to_q = DDense(inner_dim)
        to_kv = DDense(inner_dim * 2)
        edges_to_kv = DDense(inner_dim)
        to_out = DDense(nodes.shape[-1])

        q = to_q(nodes, deterministic)
        k, v = np.split(to_kv(nodes, deterministic), 2, axis=-1)

        e_kv = edges_to_kv(edges, deterministic)

        q, k, v, e_kv = map(
            lambda t: rearrange(t, "b ... (h d) -> (b h) ... d", h=h), (q, k, v, e_kv)
        )

        ek, ev = e_kv, e_kv

        k, v = map(lambda t: rearrange(t, "b j d -> b () j d"), (k, v))

        k += ek
        v += ev

        sim = np.einsum("b i d, b i j d -> b i j", q, k) * scale

        mask = rearrange(mask, "b i -> b i ()") & rearrange(mask, "b j -> b () j")
        mask = repeat(mask, "b i j -> (b h) i j", h=h)
        max_neg_value = -np.finfo(sim.dtype).max
        # sim.masked_fill_(~mask, max_neg_value)
        sim = np.where(~mask, max_neg_value, sim)

        attn = nn.softmax(sim, axis=-1)
        out = np.einsum("b i j, b i j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return to_out(out, deterministic)


class FeedForward(nn.Module):
    dim: int
    ff_mult: int = 4

    @nn.compact
    @typed
    def __call__(self, x: Array, deterministic: SBool) -> Array:
        d1 = DDense(self.dim * self.ff_mult)(x, deterministic)
        d1 = nn.gelu(d1)
        d2 = DDense(self.dim)(d1, deterministic)
        return d2


# def feedforward(dim: int, ff_mult: int = 4) -> nn.Module:
#     return nn.Sequential(
#         (
#             DDense(dim * ff_mult),
#             nn.gelu,
#             DDense(dim),
#         )
#     )
#


class NodeEdgeLayerPair(nn.Module):
    dim_head: int
    heads: int
    with_feedforward: bool = True

    @nn.compact
    def __call__(
        self, nodes, edges, mask, deterministic: SBool
    ):  # , node_edges_mask: tuple[Array, Array, Array]):
        # nodes, edges, mask = node_edges_mask
        attn = PreNorm(
            Attention(
                dim_head=self.dim_head,
                heads=self.heads,
            ),
        )
        attn_residual = GatedResidual()

        nodes = attn_residual(
            attn(nodes, edges, mask, deterministic), nodes, deterministic
        )

        if self.with_feedforward:
            ff = PreNorm(FeedForward(self.dim))
            ff_residual = GatedResidual()
            nodes = ff_residual(ff(nodes, deterministic), nodes, deterministic)

        return nodes, edges, mask, deterministic


class GraphTransformer(nn.Module):
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
        self,
        nodes: Array,
        edges: Array,
        mask: Array,
        node_time_embedding: Array,
        edge_time_embedding: Array,
        deterministic: bool,
    ) -> tuple[Array, Array]:
        nodes = np.concatenate((nodes, node_time_embedding), axis=-1)
        edges = np.concatenate((edges, edge_time_embedding), axis=-1)
        for i in range(self.depth):
            # print(f"depth {i} nodes {nodes.shape} edges {edges.shape}")
            # if i > 0 and i < self.depth - 1:
            #     # nodes = np.concatenate((nodes, node_time_embedding), axis=-1)
            #     # edges = np.concatenate((edges, edge_time_embedding), axis=-1)
            #     nodes += node_time_embedding
            #     edges += edge_time_embedding

            nodes, edges, mask, deterministic = NodeEdgeLayerPair(
                dim_head=self.dim_head,
                heads=self.heads,
                with_feedforward=self.with_feedforward,
            )(nodes, edges, mask, deterministic)
        # nodes, edges, _, _ = nn.Sequential(
        #     [
        #         NodeEdgeLayerPair(
        #             dim_head=self.dim_head,
        #             heads=self.heads,
        #             with_feedforward=self.with_feedforward,
        #         )
        #         for _ in range(self.depth)
        #     ]
        # )(nodes, edges, mask, deterministic)
        return nodes, edges

    @classmethod
    @typed
    def initialize(
        cls,
        key: Array,
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
        nodes_shape = (2, number_of_nodes, in_node_features)
        edges_shape = (2, number_of_nodes, number_of_nodes, in_edge_features)
        nodes = jax.random.normal(key_nodes, nodes_shape)
        edges = jax.random.normal(key_edges, edges_shape)
        mask = np.ones((2, number_of_nodes), dtype=bool)
        print(f"Init {nodes.shape=}")
        print(f"Init {edges.shape=}")
        params = model.init(
            key,
            nodes,
            edges,
            mask,
            deterministic=True,
        )
        return model, params
