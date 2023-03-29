from jax import numpy as np
import flax.linen as nn
from jax import Array

from .metrics import TrainLossDiscrete 
from metrics.train_metrics import TrainLossDiscrete
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL


class Dimensions:
    def __init__(self, x: int, e: int, y: int):
        self.x = x
        self.e = e
        self.y = y


class DiscreteDenoisingDiffusion:
    def __init__(
        self,
        model: nn.Module,
        name: str,
        input_dims: Dimensions,
        output_dims: Dimensions,
        nodes_dist: str,
    ):
        self.model = model
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.nodes_dist = nodes_dist
        self.name = name
        self.input_dims = input_dims
        self.output_dims = output_dims
