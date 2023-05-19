from jax import numpy as jnp
import jax
import numpy as np


def get_timestep_embedding(timesteps, embedding_dim: int, dtype=jnp.float32):
    """Build sinusoidal embeddings (from Fairseq).

    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".

    Args:
      timesteps: jnp.ndarray: generate embedding vectors at these timesteps
      embedding_dim: int: dimension of the embeddings to generate
      dtype: data type of the generated embeddings

    Returns:
      embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(timesteps.shape) == 1
    timesteps *= 1000.0

    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=dtype) * -emb)
    emb = timesteps.astype(dtype)[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb
