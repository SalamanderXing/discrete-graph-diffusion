from jax import numpy as np
import optax

from . import AverageMetric


class CrossEntropyMetric(AverageMetric):
    def update(self, preds, target):
        target = np.argmax(target, axis=-1)
        output = optax.softmax_cross_entropy(logits=preds, labels=target).sum()
        self._accumulator += output
        self._count += preds.shape[0]
