from mate import mate
from ..data_loaders.qm9_p import QM9DataModule, QM9Infos, get_train_smiles
import os
import ipdb
from ..models.graph_transformer import GraphTransformer, GraphTransformerConfig
from ..trainers.discrete_diffusion import train_model
import ipdb
from jax import numpy as np
from jax import random
import jax

remove_h = True
batch_size = 32
data_dir = "/Users/giuliozani/Documents/discrete-graph-diffusion/results/experiments/dataloader_test/qm9/qm9_pyg/"
datamodule = QM9DataModule(
    datadir=data_dir,
    train_batch_size=32,
    val_batch_size=32,
    test_batch_size=32,
    remove_h=remove_h,
)
dataset_infos = QM9Infos(
    datamodule=datamodule,
    remove_h=remove_h,
)
datamodule.prepare_data()
uba = next(iter(datamodule.train_dataloader()))

jax.config.update("jax_platform_name", "cpu")  # run on CPU for now.

config = GraphTransformerConfig.from_dict(
    dict(
        input_dims={
            "X": 9,
            "E": 5,
            "y": 13,
        },
        output_dims={
            "X": 4,
            "E": 5,
            "y": 1,  # TODO: this is 0 in the original code, but not sure what's the utility of even allowing the net to output y (grah-level features)
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

rngs = {"params": random.PRNGKey(0), "dropout": random.PRNGKey(1)}
key = random.PRNGKey(2)
x = random.normal(key, (batch_size, 9, config.input_dims.X))
e = random.normal(key, (batch_size, 9, 9, config.input_dims.E))
y = random.normal(key, (batch_size, config.input_dims.y))
node_mask = np.ones((batch_size, 9))

graph_transformer = GraphTransformer(config)
params = graph_transformer.init(rngs, x, e, y, node_mask)

# this is the forward pass
out = graph_transformer.apply(
    params, x, e, y, node_mask, rngs={"dropout": rngs["dropout"]}
)
