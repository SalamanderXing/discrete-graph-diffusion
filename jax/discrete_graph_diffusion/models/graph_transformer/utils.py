from jax import numpy as np
from jax import Array


def assert_correctly_masked(variable, node_mask):
    assert (
        variable * (1 - node_mask.long())
    ).abs().max().item() < 1e-4, "Variables not masked properly."


class PlaceHolder:
    def __init__(self, x:Array, e:Array, y:Array):
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

    def mask(self, node_mask:Array, collapse=False):
        x_mask = np.expand_dims(node_mask, -1)          # bs, n, 1
        e_mask1 = np.expand_dims(x_mask, 2)             # bs, n, 1, 1
        e_mask2 = np.expand_dims(x_mask, 1)             # bs, 1, n, 1

        if collapse:
            self.X = np.argmax(self.X, axis=-1)
            self.E = np.argmax(self.E, axis=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert np.allclose(self.E, np.transpose(self.E, (1, 2)))
        return self

