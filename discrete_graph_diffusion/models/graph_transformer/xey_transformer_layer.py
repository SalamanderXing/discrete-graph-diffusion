from flax import linen as nn
import jax.numpy as np
from jax import Array
from .node_edge_block import NodeEdgeBlock

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
    dim_ffx: int = 2048
    dim_ffe: int = 128 
    dim_ffy: int = 2048
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6

    @nn.compact
    def __call__(self, x:Array, e:Array, y:Array, node_mask:Array):
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
        )(x, e, y, node_mask)


        return x, y, z 
