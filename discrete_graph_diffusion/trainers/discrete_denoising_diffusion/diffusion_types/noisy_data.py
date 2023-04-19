from dataclasses import dataclass
from flax.linen import Embed

from jax import Array

from .embedded_graph import EmbeddedGraph

import jax_dataclasses as jdc
from typing import Annotated


@jdc.pytree_dataclass
class NoisyData:
    t: Annotated[
        Array,
        (),
        float,
    ]
    graph: EmbeddedGraph
    t_int: Annotated[
        Array,
        (),
        float,
    ]
    beta_t: Annotated[
        Array,
        (),
        float,
    ]
    alpha_s_bar: Annotated[
        Array,
        (),
        float,
    ]  # Product of (1 - beta_t) from 0 to s
    alpha_t_bar: Annotated[
        Array,
        (),
        float,
    ]
