from dataclasses import dataclass
from jax import Array
from jax import numpy as np
import ipdb
import jax_dataclasses as jdc
from mate.jax import typed
from jaxtyping import Float, Bool
from typing import Annotated


@jdc.pytree_dataclass
class Graph(jdc.EnforcedAnnotationsMixin):
    x: Float[Array, "b n"]
    e: Float[Array, "b n n"]
    y: Float[Array, "b ey"]
    mask: Bool[Array, "b n"]

    collapse: bool = False

    @classmethod
    @typed
    def with_trivial_mask(
        cls,
        x: Float[Array, "b n en"],
        e: Float[Array, "b n n ee"],
        y: Float[Array, "b ey"],
    ):
        return cls(x=x, e=e, y=y, mask=np.ones(x.shape[:2], dtype=bool))
