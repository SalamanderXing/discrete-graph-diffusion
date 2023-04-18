from dataclasses import dataclass

from jax import Array

from .graph import Graph

import jax_dataclasses as jdc
from typing import Annotated


@jdc.pytree_dataclass
class NoisyData:
    t: Annotated[
        Array,
        (),
        float,
    ]
    graph: Graph
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
