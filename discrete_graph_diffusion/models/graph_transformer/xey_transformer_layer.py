from flax import linen as nn
import jax.numpy as np
from jax import Array
from .node_edge_block import NodeEdgeBlock
from .config import initializers


class XEYTransformerLayer(nn.Module):
    """
    Transformer that updates node, edge and global features
    dx: node features
    dy: edge features
    dz: global features
    n_heads: number of attention heads
    dim_feedforward: dimension of feedforward layer after self-attention
    dropout: dropout rate
    layer_norm_eps: epsilon for layer normalization
    """

    dx: int
    dy: int
    de: int
    n_head: int
    initializer: str
    dim_ffx: int = 256
    dim_ffe: int = 64
    dim_ffy: int = 64
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6

    @nn.compact
    def __call__(
        self,
        x: Array,
        e: Array,
        y: Array,
        node_mask: Array,
        deterministic: bool,
    ):
        """Pass the input through the encoder layer.
        X: (bs, n, d)
        E: (bs, n, n, d)
        y: (bs, dy)
        node_mask: (bs, n) Mask for the src keys per batch (optional)
        Output: newX, newE, new_y with the same shape.
        """
        new_x, new_e, new_y = NodeEdgeBlock(
            dx=self.dx,
            dy=self.dy,
            de=self.de,
            n_head=self.n_head,
            initializer=self.initializer,
        )(x, e, y, node_mask)

        norm_x = nn.LayerNorm(self.layer_norm_eps)(
            x + nn.Dropout(rate=self.dropout, deterministic=deterministic)(new_x)
        )
        norm_e = nn.LayerNorm(self.layer_norm_eps)(
            e + nn.Dropout(rate=self.dropout, deterministic=deterministic)(new_e)
        )
        norm_y = nn.LayerNorm(self.layer_norm_eps)(
            y + nn.Dropout(rate=self.dropout, deterministic=deterministic)(new_y)
        )
        ff_out_x = nn.LayerNorm(self.layer_norm_eps)(
            x
            + nn.Dropout(self.dropout, deterministic=deterministic)(
                nn.Dense(self.dim_ffx, kernel_init=initializers[self.initializer])(
                    nn.Dropout(self.dropout, deterministic=deterministic)(
                        nn.relu(norm_x)
                    )
                )
            )
        )
        ff_out_e = nn.LayerNorm(self.layer_norm_eps)(
            e
            + nn.Dropout(self.dropout, deterministic=deterministic)(
                nn.Dense(self.dim_ffe, kernel_init=initializers[self.initializer])(
                    nn.Dropout(self.dropout, deterministic=deterministic)(
                        nn.relu(norm_e)
                    )
                )
            )
        )
        ff_out_y = nn.LayerNorm(self.layer_norm_eps)(
            y
            + nn.Dropout(self.dropout, deterministic=deterministic)(
                nn.Dense(self.dim_ffy, kernel_init=initializers[self.initializer])(
                    nn.Dropout(self.dropout, deterministic=deterministic)(
                        nn.relu(norm_y)
                    )
                )
            )
        )
        return ff_out_x, ff_out_e, ff_out_y
