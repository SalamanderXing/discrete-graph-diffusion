"""
```yaml
```
"""

from mate import mate

from ..data_loaders.tu import load_data

import tensorflow as tf

tf.config.experimental.set_visible_devices([], "GPU")

ds_name = "MUTAG"  # "PTC_MR"  # "MUTAG"
batch_size = 3  # 7 if ds_name == "PTC_MR" else 10
tu_dataset = load_data(
    save_path=mate.save_dir,
    seed=0,
    batch_size=batch_size,
    name=ds_name,
    one_hot=True,
)

# if not platform.system() == "Darwin":


import platform
import jax

# jax.config.update("jax_platform_name", "cpu")  # run on CPU for now.

jax.config.update("jax_debug_nans", True)

import jax
from jax import config
from jax.lib import xla_bridge


import os
from ..models.gat_custom import GAT
from ..trainers.ddd_trainer import run_model, TrainingConfig
import ipdb
from jax import numpy as np
from jax import random
from rich import print


print(f"Dataset: [yellow]{ds_name} [/yellow]")
print(f"Using device: [yellow]{xla_bridge.get_backend().platform} [/yellow]")

data_key = random.PRNGKey(0)

training_config = TrainingConfig.from_dict(
    dict(
        diffusion_steps=100,
        diffusion_noise_schedule="cosine",
        learning_rate=1e-4,
        lambda_train=(5, 0),
        transition="marginal",
        number_chain_steps=50,
        log_every_steps=4,
        max_num_nodes=tu_dataset.n,
        num_node_features=tu_dataset.max_node_feature,
        num_edge_features=tu_dataset.max_edge_feature,
    )
)
rngs = {"params": random.PRNGKey(0), "dropout": random.PRNGKey(1)}
model, params = GAT.initialize(
    key=rngs["params"],
    in_node_features=tu_dataset.node_prior.shape[-1],
    in_edge_features=tu_dataset.edge_prior.shape[-1],
    n=tu_dataset.n,
    num_layers=3,
)
best_val_loss = run_model(
    config=training_config,
    model=model,
    params=params,
    rngs=rngs,
    lr=training_config.learning_rate,
    train_loader=tu_dataset.train_loader,
    val_loader=tu_dataset.test_loader,
    num_epochs=200,
    action=mate.command if mate.command else "train",
    save_path=mate.save_dir,
    ds_name=ds_name,
    nodes_dist=np.array(tu_dataset.nodes_dist),
    nodes_prior=tu_dataset.node_prior,
    edges_prior=tu_dataset.edge_prior,
    bits_per_edge=True,
)
mate.result({f"{ds_name} best_val_loss": best_val_loss})
