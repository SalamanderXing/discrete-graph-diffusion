from jax import Array
from jax import numpy as np
import ipdb
import jax_dataclasses as jdc
from mate.jax import typed
from jaxtyping import Float, Bool


@jdc.pytree_dataclass
class DataBatch:
    """
    Data structure mimicking the PyTorch Geometric DataBatch.
    """

    edge_index: Array
    edge_attr: Array
    x: Array
    y: Array
    batch: Array

    @classmethod
    def from_torch(cls, data):
        return cls(
            edge_index=np.array(data.edge_index),
            edge_attr=np.array(data.edge_attr),
            x=np.array(data.x),
            y=np.array(data.y),
            batch=np.array(data.batch),
        )
