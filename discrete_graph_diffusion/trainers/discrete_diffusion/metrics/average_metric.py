from jax import Array

from . import Metric


class AverageMetric(Metric):
    def __init__(self):
        super().__init__()
        self._sum = 0
        self._count = 0

    def update(self, value: Array):
        self._sum += value.sum()
        self._count += value.size

    def compute(self):
        return self._sum / self._count
