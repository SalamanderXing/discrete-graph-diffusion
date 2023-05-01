from jax import numpy as np
from jax import Array
from flax import linen as nn
import ipdb

from mate.jax import typed
from .xey_transformer_layer import XEYTransformerLayer
from .utils import PlaceHolder  # , assert_correctly_masked
from .config import GraphTransformerConfig, initializers


class GraphTransformer(nn.Module):
    """
    n_layers: int -- number of layers
    dims: dict -- dimensions for each feature type
    """

    config: GraphTransformerConfig

    @typed
    def setup(
        self,
        act_fn_in=nn.relu,
        act_fn_out=nn.relu,
    ):
        dx = self.config.hidden_dims.dx
        de = self.config.hidden_dims.de
        dy = self.config.hidden_dims.dy

        self.mlp_in_x = nn.Sequential(
            [
                nn.Dense(
                    self.config.hidden_mlp_dims.X,
                    use_bias=False,
                ),
                act_fn_in,
                nn.Dense(
                    dx,
                    use_bias=False,
                    kernel_init=initializers[self.config.initializer],
                ),
            ]
        )
        self.mlp_in_e = nn.Sequential(
            [
                nn.Dense(
                    self.config.hidden_mlp_dims.E,
                    use_bias=False,
                    kernel_init=initializers[self.config.initializer],
                ),
                act_fn_in,
                nn.Dense(
                    self.config.hidden_dims.de,
                    use_bias=False,
                    kernel_init=initializers[self.config.initializer],
                ),
            ]
        )
        self.mlp_in_y = nn.Sequential(
            [
                nn.Dense(
                    self.config.hidden_mlp_dims.y,
                    use_bias=False,
                    kernel_init=initializers[self.config.initializer],
                ),
                act_fn_in,
                nn.Dense(
                    dy,
                    use_bias=False,
                    kernel_init=initializers[self.config.initializer],
                ),
            ]
        )

        self.layers = [
            XEYTransformerLayer(
                dx=dx,
                dy=dy,
                de=de,
                n_head=self.config.hidden_dims.n_head,
                initializer=self.config.initializer,
            )
            for _ in range(self.config.n_layers)
        ]

        self.mlp_out_x = nn.Sequential(
            [
                nn.Dense(
                    dx,
                    use_bias=False,
                    kernel_init=initializers[self.config.initializer],
                ),
                act_fn_out,
                nn.Dense(
                    self.config.output_dims.X,
                    use_bias=False,
                    kernel_init=initializers[self.config.initializer],
                ),
            ]
        )

        self.mlp_out_e = nn.Sequential(
            [
                nn.Dense(
                    de,
                    use_bias=False,
                    kernel_init=initializers[self.config.initializer],
                ),
                act_fn_out,
                nn.Dense(
                    self.config.output_dims.E,
                    use_bias=False,
                    kernel_init=initializers[self.config.initializer],
                ),
            ]
        )
        self.mlp_out_y = nn.Sequential(
            [
                nn.Dense(
                    dy,
                    use_bias=False,
                    kernel_init=initializers[self.config.initializer],
                ),
                act_fn_out,
                nn.Dense(
                    self.config.output_dims.y,
                    use_bias=False,
                    kernel_init=initializers[self.config.initializer],
                ),
            ]
        )

    @typed
    def __call__(
        self,
        x: Array,
        e: Array,
        y: Array,
        node_mask: Array,
        deterministic: bool = False,
    ):
        bs, n = x.shape[0], x.shape[1]

        diag_mask = np.broadcast_to(
            ~np.eye(n, dtype=bool)[None, :, :, None],
            (bs, n, n, 1),
        )

        x_to_out = x[..., : self.config.output_dims.X]
        e_to_out = e[..., : self.config.output_dims.E]
        y_to_out = y[..., : self.config.output_dims.y]

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

        return PlaceHolder(x=x, e=e, y=y).mask(node_mask)
