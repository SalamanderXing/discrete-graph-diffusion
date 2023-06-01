from flax import linen as nn
from jax import numpy as np
from jax import Array
import ipdb
from .layers import XToY, EToY, masked_softmax
from .config import initializers


class NodeEdgeBlock(nn.Module):
    """Self attention layer that also updates the representations of the edges."""

    dx: int
    de: int
    dy: int
    n_head: int
    initializer: str

    def setup(self):
        # Attention
        self.q = nn.Dense(
            self.dx,
            use_bias=False,
            kernel_init=initializers[self.initializer],
        )
        self.k = nn.Dense(
            self.dx,
            use_bias=False,
            kernel_init=initializers[self.initializer],
        )
        self.v = nn.Dense(
            self.dx,
            use_bias=False,
            kernel_init=initializers[self.initializer],
        )

        # FiLM E -> X
        # the following matrix must have size (de, dx)
        self.e_add = nn.Dense(
            self.dx,
            use_bias=False,
            kernel_init=initializers[self.initializer],
        )
        self.e_mul = nn.Dense(
            self.dx,
            use_bias=False,
            kernel_init=initializers[self.initializer],
        )

        # FiLM Y -> E
        # the following matrix must have size (dy, dx)
        self.y_e_mul = nn.Dense(
            self.dx,
            use_bias=False,
            kernel_init=initializers[self.initializer],
        )
        self.y_e_add = nn.Dense(
            self.dx, use_bias=False, kernel_init=initializers[self.initializer]
        )
        # FiLM Y -> X
        # the following matrix must have size (dy, dx)
        self.y_x_mul = nn.Dense(
            self.dx,
            use_bias=False,
            kernel_init=initializers[self.initializer],
        )
        self.y_x_add = nn.Dense(
            self.dx, use_bias=False, kernel_init=initializers[self.initializer]
        )
        # Process y
        self.y_y = nn.Dense(
            self.dy, use_bias=False, kernel_init=initializers[self.initializer]
        )
        self.x_y = XToY(dy=self.dy, initializer=self.initializer)
        self.e_y = EToY(dy=self.dy, initializer=self.initializer)
        self.x_out = nn.Dense(
            self.dx, use_bias=False, kernel_init=initializers[self.initializer]
        )
        self.e_out = nn.Dense(
            self.de, use_bias=False, kernel_init=initializers[self.initializer]
        )
        self.y_out = nn.Sequential(
            [
                nn.Dense(
                    self.dy, use_bias=False, kernel_init=initializers[self.initializer]
                ),
                nn.relu,
                nn.Dense(
                    self.dy, use_bias=False, kernel_init=initializers[self.initializer]
                ),
            ]
        )

    def __call__(self, x: Array, e: Array, y: Array, node_mask: Array):
        """
        :param x: (bs, n, dx)
        :param e: (bs, n, n, de)
        :param y: (bs, dz)
        :param node_mask: (bs, n)
        :return: (bs, n, dx), (bs, n, n, de), (bs, dz)
        """
        df = self.dx // self.n_head
        bs, n, _ = x.shape
        x_mask = node_mask[..., None]
        e_mask1 = np.expand_dims(x_mask, axis=2)
        e_mask2 = np.expand_dims(x_mask, axis=1)

        # 1. Map X to keys and queries
        q = self.q(x) * x_mask
        k = self.k(x) * x_mask
        # assert_correctly_masked(q, x_mask)

        q = q.reshape(bs, n, 1, self.n_head, df)
        k = k.reshape(bs, 1, n, self.n_head, df)

        # Compute unnormalized attention scores. Y is (bs, n, n, n_heads, df)
        attn = q * k
        attn /= np.sqrt(attn.shape[-1])
        # assert_correctly_masked(attn, (e_mask1 * e_mask2)[..., None])

        e1 = (self.e_mul(e) * e_mask1 * e_mask2).reshape(bs, n, n, self.n_head, df)
        e2 = (self.e_add(e) * e_mask1 * e_mask2).reshape(bs, n, n, self.n_head, df)

        # Incorporate edge features into attention scores
        attn = attn * (e1 + 1) + e2  # (bs, n, n, n_heads, df)

        # Incorporate y to E
        new_e = attn.reshape(bs, n, n, -1)
        ye1 = self.y_e_add(y)[:, None, None, :]
        ye2 = self.y_e_mul(y)[:, None, None, :]
        # prints all the shapes again
        new_e = ye1 + (ye2 + 1) * new_e

        # Output E
        new_e = self.e_out(new_e) * e_mask1 * e_mask2
        # assert_correctly_masked(new_e, (e_mask1 * e_mask2))

        # Compute attentios attn is still (bs, n, n, n_heads, df)
        softmax_mask = np.broadcast_to(e_mask2, (bs, n, n, self.n_head))
        attn = masked_softmax(attn, softmax_mask, axis=2)

        v = self.v(x) * x_mask
        v = v.reshape(bs, n, 1, self.n_head, df)

        # Compute weighted values
        weighted_values = (attn * v).sum(axis=2)

        # Send output to input dim
        weighted_values = weighted_values.reshape(bs, n, -1)

        # Incorporate y to X
        yx1 = self.y_x_add(y)[:, None, :]
        yx2 = self.y_x_mul(y)[:, None, :]
        # prints all the shapes again

        new_x = yx1 + (yx2 + 1) * weighted_values

        # Output X
        new_x = self.x_out(new_x) * x_mask
        # assert_correctly_masked(new_x, x_mask)

        # Process y based on x and e
        y = self.y_y(y)
        e_y = self.e_y(e)
        x_y = self.x_y(x)
        # prints all the shapes again
        new_y = self.y_out(y + x_y + e_y)

        return new_x, new_e, new_y
