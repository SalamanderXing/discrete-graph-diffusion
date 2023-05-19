from jax import numpy as np
from jax.random import PRNGKeyArray
from jax import Array
import jax


class DistributionNodes:
    def __init__(self, histogram: Array | dict, rng: PRNGKeyArray):
        """Compute the distribution of the number of nodes in the dataset, and sample from this distribution.
        historgram: dict. The keys are num_nodes, the values are counts
        """
        self.rng = rng
        prob = np.array([])
        if type(histogram) == dict:
            max_n_nodes = max(histogram.keys())
            prob = np.zeros(max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob = prob.at[num_nodes].set(count)
        elif isinstance(histogram, Array):
            prob = histogram
        else:
            raise TypeError("histogram must be a dict or a jax Array")

        self.prob = prob / prob.sum()

    def sample_n(self, n_samples):
        return jax.random.choice(
            jax.random.PRNGKey(0),
            a=np.arange(len(self.prob)),
            p=self.prob,
            shape=(n_samples,),
        )

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.shape) == 1
        probas = self.prob[batch_n_nodes]
        log_p = np.log(probas + 1e-30)
        return log_p
