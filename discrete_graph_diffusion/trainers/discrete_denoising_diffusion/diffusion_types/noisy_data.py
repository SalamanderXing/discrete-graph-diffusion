from dataclasses import dataclass
from flax.linen import Embed

from jax import Array

from .embedded_graph import GraphDistribution

import jax_dataclasses as jdc
from jaxtyping import Float, Bool 
from mate.jax import SFloat, SInt



@jdc.pytree_dataclass
class NoisyData:
    t: SFloat
    embedded_graph: GraphDistribution
    t_int: SInt
    beta_t: SFloat
    alpha_s_bar: SFloat  # Product of (1 - beta_t) from 0 to s
    alpha_t_bar: SFloat
