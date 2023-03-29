from jax import numpy as np
from jax import Array
from flax import linen as nn

from .xey_transformer_layer import XEYTransformerLayer
from .utils import PlaceHolder, assert_correctly_masked


class GraphTransformer(nn.Module):
    """
    n_layers: int -- number of layers
    dims: dict -- dimensions for each feature type
    """

    def setup(
        self,
        n_layers: int,
        input_dims: dict,
        hidden_mlp_dims: dict,
        hidden_dims: dict,
        output_dims: dict,
        act_fn_in=nn.relu,
        act_fn_out=nn.relu,
    ):
        self.n_layers = n_layers
        self.out_dim_x = output_dims["X"]
        self.out_dim_e = output_dims["E"]
        self.out_dim_y = output_dims["y"]

        self.mlp_in_x = nn.Sequential(
            [
                nn.Dense(
                    hidden_mlp_dims["X"],
                    use_bias=False,
                ),
                act_fn_in,
                nn.Dense(
                    hidden_mlp_dims["dx"],
                    use_bias=False,
                ),
            ]
        )
        self.mlp_in_e = nn.Sequential(
            [
                nn.Dense(
                    hidden_mlp_dims["E"],
                    use_bias=False,
                ),
                act_fn_in,
                nn.Dense(
                    hidden_mlp_dims["de"],
                    use_bias=False,
                ),
            ]
        )
        self.mlp_in_y = nn.Sequential(
            [
                nn.Dense(
                    hidden_mlp_dims["y"],
                    use_bias=False,
                ),
                act_fn_in,
                nn.Dense(
                    hidden_mlp_dims["dy"],
                    use_bias=False,
                ),
            ]
        )

        self.layers = [XEYTransformerLayer(**hidden_dims) for _ in range(n_layers)]

        self.mlp_out_x = nn.Sequential(
            [
                nn.Dense(
                    hidden_mlp_dims["dx"],
                    use_bias=False,
                ),
                act_fn_out,
                nn.Dense(
                    self.out_dim_x,
                    use_bias=False,
                ),
            ]
        )

        self.mlp_out_e = nn.Sequential(
            [
                nn.Dense(
                    hidden_mlp_dims["de"],
                    use_bias=False,
                ),
                act_fn_out,
                nn.Dense(
                    self.out_dim_e,
                    use_bias=False,
                ),
            ]
        )

        self.mlp_out_y = nn.Sequential(
            [
                nn.Dense(
                    hidden_mlp_dims["dy"],
                    use_bias=False,
                ),
                act_fn_out,
                nn.Dense(
                    self.out_dim_y,
                    use_bias=False,
                ),
            ]
        )

    def __call__(self, x: Array, e: Array, y: Array, node_mask: Array):
        bs, n = x.shape[0], x.shape[1]

        diag_mask = np.broadcast_to(
            ~np.eye(n, dtype=bool)[None, :, :, None],
            (bs, n, n, 1),
        )

        x_to_out = x[..., : self.out_dim_x]
        e_to_out = e[..., : self.out_dim_e]
        y_to_out = y[..., : self.out_dim_y]

        new_e = self.mlp_in_e(e)
        new_e = (new_e + new_e.transpose((0, 2, 1))) / 2

        after_in = PlaceHolder(x=x, e=new_e, y=self.mlp_in_y(y)).mask(node_mask)
        x, e, y = after_in.x, after_in.e, after_in.y

        for layer in self.tf_layers:
            x, e, y = layer(x, e, y, node_mask)

        x = self.mlp_out_x(x)
        e = self.mlp_out_e(e)
        y = self.mlp_out_y(y)

        x = x + x_to_out
        e = (e + e_to_out) * diag_mask
        y = y + y_to_out

        e = 1 / 2 * (e + e.transpose((0, 2, 1)))

        return PlaceHolder(x=x, e=e, y=y).mask(node_mask)
