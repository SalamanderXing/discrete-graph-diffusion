from jax import numpy as np
from jax import Array

import ipdb


def assert_correctly_masked(variable, node_mask):
    condition = (
        np.abs(variable * (1 - node_mask.astype(variable.dtype))).max().item() < 1e-4
    )
    assert condition, "Variables not masked properly."





