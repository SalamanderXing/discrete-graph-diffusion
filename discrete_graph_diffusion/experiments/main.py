"""
This is just a test experiment to see if the graph transformer works
"""

from ..models.graph_transformer import GraphTransformer, GraphTransformerConfig
import ipdb
from jax import numpy as np
from jax import random
import jax

jax.config.update("jax_platform_name", "cpu")

graph_transformer_config = GraphTransformerConfig.from_dict(
    dict(
        input_dims={
            "X": 9,
            "E": 12,
            "y": 13,
        },
        output_dims={
            "X": 4,
            "E": 5,
            "y": 0,
        },
        hidden_mlp_dims={"X": 256, "E": 128, "y": 128},
        hidden_dims={
            "dx": 256,
            "de": 64,
            "dy": 64,
            "n_head": 8,
            "dim_ffX": 256,
            "dim_ffE": 128,
            "dim_ffy": 128,
        },
        n_layers=5,
    )
)
# creates a test input, that is a random tensor of shape (32, 9, 12)
key = random.PRNGKey(0)
x = random.normal(key, (32, 9, graph_transformer_config.input_dims.X))
e = random.normal(key, (32, 9, 9, graph_transformer_config.input_dims.E))
y = random.normal(key, (32, graph_transformer_config.input_dims.y))
node_mask = np.ones((32, 9, 1))

gt = GraphTransformer(graph_transformer_config)

gt.init(key, x, e, y, node_mask)
