from flax import linen as nn
import jax.numpy as jnp

class XEYTransformerLayer(nn.Module):
    """
    Transformer that updates node, edge and global features
    d_x: node features
    d_y: edge features
    dz: global features
    n_heads: number of attention heads
    dim_feedforward: dimension of feedforward layer after self-attention
    dropout: dropout rate
    layer_norm_eps: epsilon for layer normalization
    """


    d_x: int
    d_y: int
    dz: int
    n_heads: int
    dim_feedforward: int
    dropout: float
    layer_norm_eps: float = 1e-6

    @nn.compact
    def __call__(self, x, y, z, mask_x, mask_y, mask_z):
        """
        x: node features
        y: edge features
        z: global features
        mask_x: mask for node features
        mask_y: mask for edge features
        mask_z: mask for global features
        """

        # self-attention
        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        x = nn.SelfAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_x,
            out_features=self.d_x,
            dropout_rate=self.dropout,
            deterministic=False,
        )(x, mask=mask_x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=False)
        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        x = nn.Dense(features=self.d_x)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=False)

        # cross-attention
        y = nn.LayerNorm(epsilon=self.layer_norm_eps)(y)
        y = nn.SelfAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_y,
            out_features=self.d_y,
            dropout_rate=self.dropout,
            deterministic=False,
        )(y, mask=mask_y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic=False)
        y = nn.LayerNorm(epsilon=self.layer_norm_eps)(y)
        y = nn.Dense(features=self.d_y)(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic=False)

        # feedforward
        z = nn.LayerNorm(epsilon=self.layer_norm_eps)(z)
        z = nn.Dense(features=self.dim_feedforward)(z)
        z = nn.gelu(z)
        z = nn.Dropout(rate=self.dropout)(z, deterministic=False)
        z = nn.Dense(features=self.dz)(z)
        z = nn.Dropout(rate=self.dropout)(z, deterministic=False)

        return x, y, z 
