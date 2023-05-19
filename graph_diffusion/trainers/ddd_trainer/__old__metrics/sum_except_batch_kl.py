from jax import numpy as np
from jax import Array

from .average_metric import AverageMetric


class SumExceptBatchKL(AverageMetric):
    def update(self, p: Array, q: Array):
        self._sum += np.sum(p * np.log(p / q), axis=1).sum()  # TODO: check this
        self._count += p.shape[0]
