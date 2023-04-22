from jax import numpy as np
from jax import Array
import ipdb

check = lambda x, y="": None


class PlaceHolder:
    def __init__(self, x: Array, e: Array, y: Array):
        self.x = x
        self.e = e
        self.y = y


    def __str__(self):
        return f"X: {self.x.shape}, E: {self.e.shape}, y: {self.y.shape}"

    def __repr__(self):
        return self.__str__()

    def mask(self, node_mask: Array, collapse=False):
        x_mask = np.expand_dims(node_mask, -1)  # bs, n, 1
        e_mask1 = np.expand_dims(x_mask, 2)  # bs, n, 1, 1
        e_mask2 = np.expand_dims(x_mask, 1)  # bs, 1, n, 1

        if collapse:
            self.x = np.argmax(self.x, axis=-1)
            self.e = np.argmax(self.e, axis=-1)

            self.x[node_mask == 0] = -1
            self.e[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        else:
            self.x = self.x * x_mask
            self.e = self.e * e_mask1 * e_mask2
            check(np.allclose(self.e, np.transpose(self.e, (0, 2, 1, 3))))
        return self
