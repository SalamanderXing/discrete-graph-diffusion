from .graph_distribution import (
    GraphDistribution,
    NodeDistribution,
    EdgeDistribution,
    NodeMaskType,
    EdgeMaskType,
    get_masks,
)
from jax import numpy as np
from jaxtyping import jaxtyped
from flax.struct import dataclass
from beartype import beartype
from .one_hot_graph_distribution import OneHotGraph


