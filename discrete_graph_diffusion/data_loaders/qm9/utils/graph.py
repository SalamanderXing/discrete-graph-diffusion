from dataclasses import dataclass
from jax import Array
from jax import numpy as np


@dataclass(frozen=True)
class Graph:
    x: Array
    e: Array
    y: Array
    mask: Array = np.ones((1, 1), dtype=bool)
    collapse: bool = False

    def __post_init__(self):
        assert self.x.shape[0] == self.e.shape[0] == self.y.shape[0]
        assert self.x.shape[1] == self.e.shape[1] == self.e.shape[2]
        assert self.x.shape[2] == self.y.shape[1]
        if self.mask is not None:
            assert self.mask.shape[0] == self.x.shape[0]
            assert self.mask.shape[1] == self.x.shape[1]
            self.__mask(self.mask)
        else:
            object.__setattr__(self, "mask", np.ones(self.x.shape[:2], dtype=bool))

    def __str__(self):
        return f"Graph(X: {self.x.shape}, E: {self.e.shape}, y: {self.y.shape})"

    def __repr__(self):
        return self.__str__()

    def set_values(
        self,
        x: Array | None,
        e: Array | None,
        y: Array | None,
        mask: Array | None,
    ) -> "Graph":
        """Sets the values of X, E, y."""
        if x is None:
            x = self.x
        if e is None:
            e = self.e
        if y is None:
            y = self.y
        if mask is None:
            mask = self.mask
        return Graph(x=x, e=e, y=y, mask=mask)

    def type_as(self, x: Array) -> "Graph":
        """Changes the device and dtype of X, E, y."""
        dtype = x.dtype
        return Graph(
            x=self.x.astype(dtype),
            e=self.e.astype(dtype),
            y=self.y.astype(dtype),
            mask=self.mask,
        )

    def __mask(self, node_mask: Array):
        x_mask = np.expand_dims(node_mask, -1)  # bs, n, 1
        e_mask1 = np.expand_dims(x_mask, 2)  # bs, n, 1, 1
        e_mask2 = np.expand_dims(x_mask, 1)  # bs, 1, n, e_mask1
        if self.collapse:
            x = np.argmax(self.x, axis=-1)
            e = np.argmax(self.e, axis=-1)

            x[node_mask == 0] = -1
            e[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        else:
            x = self.x * x_mask
            e = self.e * e_mask1 * e_mask2
            assert np.allclose(e, np.transpose(e, (0, 2, 1, 3)))
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "e", e)
