from dataclasses import dataclass

from jax import Array

from .graph import Graph


@dataclass(frozen=True)
class NoisyData:
    t: Array | int | None
    graph: Graph
    t_int: Array | int | None = None
    beta_t: Array | int | None = None
    alpha_s_bar: Array | None = None  # Product of (1 - beta_t) from 0 to s
    alpha_t_bar: Array | None = None  #
