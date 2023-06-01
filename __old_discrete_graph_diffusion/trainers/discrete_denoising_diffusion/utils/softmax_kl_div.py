from jax import numpy as np
from jax import Array
from jax.scipy.special import logsumexp


def softmax_kl_div(tensor1: Array, tensor2: Array, reduction: str = "mean") -> Array:
    # Subtract maximum value for numerical stability
    tensor1_max, tensor2_max = (
        tensor1.max(axis=-1, keepdims=True),
        tensor2.max(axis=-1, keepdims=True),
    )
    tensor1_stable, tensor2_stable = tensor1 - tensor1_max, tensor2 - tensor2_max

    # Compute log-sum-exp for both tensors
    log_sum_exp1 = logsumexp(tensor1_stable, axis=-1, keepdims=True)
    log_sum_exp2 = logsumexp(tensor2_stable, axis=-1, keepdims=True)

    # Compute the difference between input tensors and their log-sum-exp values
    tensor1_diff = tensor1_stable - log_sum_exp1
    tensor2_diff = tensor2_stable - log_sum_exp2

    # Calculate the proportional softmax values for tensor1
    proportional_softmax1 = np.exp(tensor1_diff)

    # Normalize the softmax values by dividing by the sum along the last dimension
    normalized_softmax1 = proportional_softmax1 / proportional_softmax1.sum(
        axis=-1, keepdims=True
    )

    # Calculate the KL divergence without explicitly computing the softmax values
    kl_div = normalized_softmax1 * (tensor1_diff - tensor2_diff)

    if reduction == "batchmean":
        kl_div = kl_div.sum(axis=-1).mean()
    elif reduction == "batchsum":
        kl_div = kl_div.sum(axis=-1).sum()
    elif reduction == "none":
        pass  # Keep the element-wise KL divergence values as is
    elif reduction == "mean":
        kl_div = kl_div.mean()
    else:
        raise ValueError(
            f"Invalid reduction mode. Got {reduction}. Choose from ['batchmean', 'batchsum', 'none']"
        )

    return kl_div
