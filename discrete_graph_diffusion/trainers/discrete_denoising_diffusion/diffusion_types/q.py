from dataclasses import dataclass
from jax import Array
from jax import numpy as np
import ipdb
import jax_dataclasses as jdc
from jaxtyping import Float, Bool, jaxtyped
from typeguard import typechecked
from typing import Annotated


@jdc.pytree_dataclass
class Q(jdc.EnforcedAnnotationsMixin):
    x: Float[Array, "b n"]
    e: Float[Array, "b n n"]
    y: Float[Array, "b n"]
