from jax import numpy as np
from jax import Array
import ipdb


def assert_correctly_masked(variable, node_mask):
    condition = (
        np.abs(variable * (1 - node_mask.astype(variable.dtype))).max().item() < 1e-4
    )
    assert condition, "Variables not masked properly."


class PlaceHolder:
    def __init__(self, x: Array, e: Array, y: Array):
        self.x = x
        self.e = e
        self.y = y

    """
    def type_as(self, x: Array):
        ''' Changes the device and dtype of X, E, y. '''
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self
    """
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
            try:
                self.x = self.x * x_mask
            except:
                ipdb.set_trace()
            self.e = self.e * e_mask1 * e_mask2
            assert np.allclose(self.e, np.transpose(self.e, (0, 2, 1, 3)))
        return self
