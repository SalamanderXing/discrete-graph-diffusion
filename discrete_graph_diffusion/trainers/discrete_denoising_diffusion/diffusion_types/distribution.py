from jax import Array
from jax import numpy as np
import ipdb
import jax_dataclasses as jdc
from mate.jax import typed
from jaxtyping import Float, Bool
from .geometric import to_dense
from .data_batch import DataBatch


@jdc.pytree_dataclass
class Distribution(jdc.EnforcedAnnotationsMixin):
    x: Float[Array, "x"]
    e: Float[Array, "e"]
    y: Float[Array, "y"]
