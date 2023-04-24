import tensorflow as tf
from jax import config

tf.config.experimental.set_visible_devices([], "GPU")

from jax.lib import xla_bridge

print(f"Using device: {xla_bridge.get_backend().platform}")

config.update("jax_debug_nans", True)
from mate import mate
from dataclasses import dataclass, asdict
from ..data_loaders.qm9_p import QM9DataModule, QM9Infos, get_train_smiles
import os
from ..models.graph_transformer import GraphTransformer, GraphTransformerConfig
from ..trainers.discrete_denoising_diffusion import train_model, TrainingConfig
import ipdb
from jax import numpy as np
from jax import random

remove_h = True
batch_size = 500
data_dir = os.path.join(mate.save_dir, "qm9/qm9_pyg/")
datamodule = QM9DataModule(
    datadir=data_dir,
    train_batch_size=batch_size,
    val_batch_size=batch_size,
    test_batch_size=batch_size,
    remove_h=remove_h,
)
dataset_infos = QM9Infos(
    datamodule=datamodule,
    remove_h=remove_h,
)
datamodule.prepare_data()
uba = next(iter(datamodule.train_dataloader()))

graph_transformer_config = GraphTransformerConfig.from_dict(
    dict(
        input_dims={
            "X": 4,
            "E": 5,
            "y": 1,  # 13,
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
dataset_dict = dataset_infos.__dict__
dataset_dict["n_nodes"] = np.array(dataset_dict["n_nodes"])
dataset_dict["node_types"] = np.array(dataset_dict["node_types"])
dataset_dict["edge_types"] = np.array(dataset_dict["edge_types"])
dataset_dict["valency_distribution"] = np.array(dataset_dict["valency_distribution"])
training_config = TrainingConfig.from_dict(
    dict(
        dataset=dataset_dict,
        diffusion_steps=500,
        diffusion_noise_schedule="cosine",
        learning_rate=1e-3,
        lambda_train=(5, 0),
        transition="marginal",
        number_chain_steps=50,
        log_every_steps=4,
    )
)

rngs = {"params": random.PRNGKey(0), "dropout": random.PRNGKey(1)}
key = random.PRNGKey(2)
# X.shape=torch.Size([200, 9, 12]), E.shape=torch.Size([200, 9, 9, 5]), y.shape=torch.Size([200, 13]), node_mask.shape=torch.Size([200, 9])
x = random.normal(key, (batch_size, 9, graph_transformer_config.input_dims.X))
e = random.normal(key, (batch_size, 9, 9, graph_transformer_config.input_dims.E))
y = random.normal(key, (batch_size, graph_transformer_config.input_dims.y))
node_mask = np.ones((batch_size, 9))

graph_transformer = GraphTransformer(graph_transformer_config)
print(f"Model init shapes: {x.shape}, {e.shape}, {y.shape}, {node_mask.shape}")
params = graph_transformer.init(rngs, x, e, y, node_mask)

# this is the forward pass
out = graph_transformer.apply(
    params, x, e, y, node_mask, rngs={"dropout": rngs["dropout"]}
)
train_model(
    config=training_config,
    model=graph_transformer,
    params=params,
    rngs=rngs,
    lr=training_config.learning_rate,
    train_loader=datamodule.train_dataloader(),
    val_loader=datamodule.val_dataloader(),
    output_dims=graph_transformer_config.output_dims.__dict__,
    nodes_dist=dataset_infos.nodes_dist,
)
