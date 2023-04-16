from jax import Array
import jax.numpy as np
from . import AverageMetric


class SumExceptBatchMetric(AverageMetric):
    def __init__(self):
        super().__init__()

    def update(self, values:Array) -> None:
        self._accumulator += np.sum(values)
        self._count += values.shape[0]
