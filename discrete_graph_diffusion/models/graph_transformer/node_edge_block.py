from flax import linen as nn
from jax import numpy as np
from jax import Array

from utils import assert_correctly_masked
from .layers import XToY, EToY, masked_softmax
from .utils import assert_correctly_masked


class NodeEdgeBlock(nn.Module):
    """Self attention layer that also updates the representations of the edges."""

    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx ({dx}) must be divisible by n_head ({n_head})"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = dx // n_head
        self.n_head = n_head

    def setup(self):
        # Attention
        self.q = nn.Dense(
            self.dx,
            use_bias=False,
        )
        self.k = nn.Dense(
            self.dx,
            use_bias=False,
        )
        self.v = nn.Dense(
            self.dx,
            use_bias=False,
        )

        # FiLM E -> X
        # the following matrix must have size (de, dx)
        self.e_add = nn.Dense(
            self.dx,
            use_bias=False,
        )
        self.e_mul = nn.Dense(
            self.dx,
            use_bias=False,
        )

        # FiLM Y -> E
        # the following matrix must have size (dy, dx)
        self.y_e_mul = nn.Dense(self.dx, use_bias=False)
        self.y_e_add = nn.Dense(self.dx, use_bias=False)
        # FiLM Y -> X
        # the following matrix must have size (dy, dx)
        self.y_x_mul = nn.Dense(self.dx, use_bias=False)
        self.y_x_add = nn.Dense(self.dx, use_bias=False)
        # Process y
        self.y_y = nn.Dense(self.dy, use_bias=False)
        self.x_y = XToY()
        self.e_y = EToY()

    def __call__(self, x: Array, e: Array, y: Array, node_mask: Array):
        """
        :param x: (bs, n, dx)
        :param e: (bs, n, n, de)
        :param y: (bs, dz)
        :param node_mask: (bs, n)
        :return: (bs, n, dx), (bs, n, n, de), (bs, dz)
        """
        bs, n, _ = x.shape
        x_mask = node_mask[..., None]
        e_mask1 = np.expand_dims(x_mask, axis=2)
        e_mask2 = np.expand_dims(x_mask, axis=1)

        # 1. Map X to keys and queries
        q = self.q(x) * x_mask
        k = self.k(x) * x_mask
        assert_correctly_masked(q, node_mask)

        # Compute unnormalized attention scores. Y is (bs, n, n, n_heads, df)
        y = q * k
        y /= np.sqrt(self.df)
        assert_correctly_masked(y, (e_mask1 * e_mask2))

        e1 = (self.e_mul(e) * e_mask1 * e_mask2).reshape(bs, n, n, self.n_head, self.df)
        e2 = (self.e_add(e) * e_mask1 * e_mask2).reshape(bs, n, n, self.n_head, self.df)

        # Incorporate edge features into attention scores
        y = y * (e1 + 1) + e2  # (bs, n, n, n_heads, df)

        # Incorporate y to E
        new_e = y.reshape(bs, n, n, -1)
        ye1 = self.y_e_add(y)[:, None, None, :]
        ye2 = self.y_e_mul(y)[:, None, None, :]
        new_e = ye1 + (ye2 + 1) * new_e

        # Output E
        new_e = self.e_out(new_e) * e_mask1 * e_mask2
        assert_correctly_masked(new_e, (e_mask1 * e_mask2))

        # Compute attentios attn is still (bs, n, n, n_heads, df)
        softmax_mask = np.broadcast_to(e_mask2, (bs, n, n, self.n_head))
        attn = masked_softmax(y, softmax_mask, axis=2)

        v = self.v(x) * x_mask
        v = v.reshape(bs, n, 1, self.n_head, self.df)[:, None]

        # Compute weighted values
        weighted_values = (attn * v).sum(axis=2)

        # Send output to input dim
        weighted_values = weighted_values.reshape(bs, n, -1)

        # Incorporate y to X
        yx1 = self.y_x_add(y)[:, None, :]
        yx2 = self.y_x_mul(y)[:, None, :]
        new_x = yx1 + (yx2 + 1) * weighted_values

        # Output X
        new_x = self.x_out(new_x) * x_mask
        assert_correctly_masked(new_x, x_mask)

        # Process y based on x and e
        y = self.y_y(y)
        e_y = self.e_y(e)
        x_y = self.x_y(x)
        new_y = self.y_out(y + e_y + x_y)

        return new_x, new_e, new_y
