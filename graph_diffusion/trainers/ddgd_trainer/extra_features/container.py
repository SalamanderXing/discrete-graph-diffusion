from jax import numpy as np
from jax import Array
from flax.struct import dataclass
from ....shared.graph import graph_distribution as gd


@dataclass
class Container:
    nodes: Array
    edges: Array
    y: Array
