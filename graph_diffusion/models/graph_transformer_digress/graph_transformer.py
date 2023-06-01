from jax import numpy as np
from jax import Array
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
import ipdb

from mate.jax import typed, Key
import jax
from .xey_transformer_layer import XEYTransformerLayer
from .utils import PlaceHolder  # , assert_correctly_masked
from .config import GraphTransformerConfig, initializers
from ...shared.graph_distribution import GraphDistribution
from rich import print
from flax.struct import dataclass


@dataclass
class Dimensions:
    X: int
    E: int
    y: int


@dataclass
class HiddenDimensions:
    dx: int
    de: int
    dy: int
    n_head: int
    dim_ffX: int
    dim_ffE: int
    dim_ffy: int


"""
hidden_mlp_dims: {'X': 256, 'E': 128, 'y': 128}

# The dimensions should satisfy dx % n_head == 0
hidden_dims : {'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128}
"""


class GraphTransformer(nn.Module):
    """
    n_layers: int -- number of layers
    dims: dict -- dimensions for each feature type
    """

    # config: GraphTransformerConfig
    out_node_features: int
    out_edge_features: int
    num_layers: int = 5
    num_heads: int = 8
    # hidden_node_features: int
    # hidden_edge_features: int
    initializer: str = "xavier_uniform"
    hidden_d_node_features: int = 256
    hidden_d_edge_features: int = 64
    hidden_d_y_features: int = 64
    hidden_ff_node_features: int = 256
    hidden_ff_edge_features: int = 128
    hidden_ff_y_features: int = 128

    hidden_mlp_node_features: int = 256
    hidden_mlp_edge_features: int = 128
    hidden_mlp_y_features: int = 128

    # hidden_mlp_dims: Dimensions
    # hidden_dims: HiddenDimensions
    # output_dims: Dimensions

    @typed
    def setup(
        self,
        act_fn_in=nn.relu,
        act_fn_out=nn.relu,
    ):
        dx = self.hidden_d_node_features
        de = self.hidden_d_edge_features
        dy = self.hidden_d_y_features

        self.mlp_in_x = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_mlp_node_features,
                    use_bias=False,
                ),
                act_fn_in,
                nn.Dense(
                    dx,
                    use_bias=False,
                    kernel_init=initializers[self.initializer],
                ),
            ]
        )
        self.mlp_in_e = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_mlp_edge_features,
                    use_bias=False,
                    kernel_init=initializers[self.initializer],
                ),
                act_fn_in,
                nn.Dense(
                    self.hidden_d_edge_features,
                    use_bias=False,
                    kernel_init=initializers[self.initializer],
                ),
            ]
        )
        self.mlp_in_y = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_mlp_y_features,
                    use_bias=False,
                    kernel_init=initializers[self.initializer],
                ),
                act_fn_in,
                nn.Dense(
                    dy,
                    use_bias=False,
                    kernel_init=initializers[self.initializer],
                ),
            ]
        )

        self.layers = [
            XEYTransformerLayer(
                dx=dx,
                dy=dy,
                de=de,
                n_head=self.num_heads,
                initializer=self.initializer,
            )
            for _ in range(self.num_layers)
        ]

        self.mlp_out_x = nn.Sequential(
            [
                nn.Dense(
                    dx,
                    use_bias=False,
                    kernel_init=initializers[self.initializer],
                ),
                act_fn_out,
                nn.Dense(
                    self.out_node_features,
                    use_bias=False,
                    kernel_init=initializers[self.initializer],
                ),
            ]
        )

        self.mlp_out_e = nn.Sequential(
            [
                nn.Dense(
                    de,
                    use_bias=False,
                    kernel_init=initializers[self.initializer],
                ),
                act_fn_out,
                nn.Dense(
                    self.out_edge_features,
                    use_bias=False,
                    kernel_init=initializers[self.initializer],
                ),
            ]
        )
        self.mlp_out_y = nn.Sequential(
            [
                nn.Dense(
                    dy,
                    use_bias=False,
                    kernel_init=initializers[self.initializer],
                ),
                act_fn_out,
                nn.Dense(
                    1,
                    use_bias=False,
                    kernel_init=initializers[self.initializer],
                ),
            ]
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
        num_layers: int = 3,
        initializer: str = "xavier_uniform",
    ) -> tuple[nn.Module, FrozenDict]:
        out_node_features = (
            out_node_features if out_node_features > 0 else in_node_features
        )
        out_edge_features = (
            out_edge_features if out_edge_features > 0 else in_edge_features
        )
        model = cls(
            initializer=initializer,
            out_node_features=out_node_features,
            out_edge_features=out_edge_features,
            num_layers=num_layers,
        )

        key_nodes, key_edges = jax.random.split(key, num=2)
        nodes_shape = (2, n, in_node_features)
        edges_shape = (2, n, n, in_edge_features)
        nodes = jax.random.normal(key_nodes, nodes_shape)
        edges = jax.random.normal(key_edges, edges_shape)
        # node_mask = np.ones((2, n), dtype=bool)
        dummy_graph = GraphDistribution.create(
            nodes,
            edges,
            edges_counts=np.ones(nodes.shape[0], int),
            nodes_counts=np.ones(nodes.shape[0], int),
        )
        print(f"[orange]Init nodes[/orange]: {nodes.shape}")
        print(f"Init edges: {edges.shape}")
        params = model.init(
            key,
            dummy_graph,
            np.zeros((2, 129)),
            deterministic=True,
        )
        return model, params

    @typed
    def __call__(
        self,
        g: GraphDistribution,
        y: Array,
        deterministic: bool = False,
    ) -> GraphDistribution:
        x, e, node_mask = g.nodes, g.edges, g.node_mask()
        bs, n = x.shape[0], x.shape[1]

        diag_mask = np.broadcast_to(
            ~np.eye(n, dtype=bool)[None, :, :, None],
            (bs, n, n, 1),
        )

        x_to_out = x[..., : self.out_node_features]
        e_to_out = e[..., : self.out_edge_features]
        y_to_out = y[..., :1]

        new_e = self.mlp_in_e(e)
        new_e = (new_e + new_e.transpose((0, 2, 1, 3))) / 2

        after_in = PlaceHolder(x=self.mlp_in_x(x), e=new_e, y=self.mlp_in_y(y)).mask(
            node_mask
        )
        x, e, y = after_in.x, after_in.e, after_in.y

        for layer in self.layers:  # TODO: replace with a nn.Sequential
            x, e, y = layer(x, e, y, node_mask, deterministic=deterministic)

        x = self.mlp_out_x(x)
        e = self.mlp_out_e(e)
        y = self.mlp_out_y(y)

        x = x + x_to_out
        e = (e + e_to_out) * diag_mask
        y = y + y_to_out
        e = 1 / 2 * (e + e.transpose((0, 2, 1, 3)))

        # return PlaceHolder(x=x, e=e, y=y).mask(node_mask)
        return GraphDistribution.create(
            nodes=x, e=e, nodes_counts=g.nodes_counts, edges_counts=g.edges_counts
        )
