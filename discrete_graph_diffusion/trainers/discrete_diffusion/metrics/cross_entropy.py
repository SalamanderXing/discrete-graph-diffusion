from jax import numpy as np
import optax

from . import Metric


class CrossEntropyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_ce", default=np.array(0.0))
        self.add_state("total_samples", default=np.array(0.0))

    def update(self, preds, target):
        target = np.argmax(target, axis=-1)
        output = optax.softmax_cross_entropy(logits=preds, labels=target).sum()
        self["total_ce"] += output
        self["total_samples"] += preds.shape[0]

    def compute(self):
        return self["total_ce"] / self["total_samples"]

    def reset(self):
        self["total_ce"] = np.array(0.0)
        self["total_samples"] = np.array(0.0)
