from dataclasses import dataclass
from jax import Array
from jax import numpy as np
import ipdb
import jax_dataclasses as jdc
from jaxtyping import Float, Bool, Int, jaxtyped
from typeguard import typechecked


@jdc.pytree_dataclass
class Q(jdc.EnforcedAnnotationsMixin):
    x: Float[Array, "b n"]
    e: Float[Array, "b n n"]

    # overrides the square bracket indexing
    def __getitem__(self, key: int | Array) -> "Q":
        return Q(x=self.x[key], e=self.e[key])

    @property
    def shape(self) -> dict[str, tuple[int, ...]]:
        return dict(x=self.x.shape, e=self.e.shape)
