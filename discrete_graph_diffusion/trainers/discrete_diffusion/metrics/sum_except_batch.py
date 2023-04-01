from . import Metric
import jax.numpy as np


class SumExceptBatchMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_value", default=np.array(0.0))
        self.add_state("total_samples", default=np.array(0.0))

    def update(self, values) -> None:
        self.total_value += np.sum(values)
        self.total_samples += values.shape[0]

    def compute(self):
        return self.total_value / self.total_samples
