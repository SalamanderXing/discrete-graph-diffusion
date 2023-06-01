import platform
import jax

jax.config.update("jax_platform_name", "cpu")  # run on CPU for now.

if not platform.system() == "Darwin":
    import tensorflow as tf

    tf.config.experimental.set_visible_devices([], "GPU")
import jax
from jax import config
from jax.lib import xla_bridge


from mate import mate
from ..data_loaders.tu2 import load_data
import os
from ..models.gat import GAT
from ..trainers.discrete_denoising_diffusion import run_model, TrainingConfig
import ipdb
from jax import numpy as np
from jax import random
from rich import print


print(f"Using device: [yellow]{xla_bridge.get_backend().platform} [/yellow]")
batch_size = 20

data_key = random.PRNGKey(0)
ds_name = "MUTAG"
tu_dataset = load_data(
    save_path=mate.save_dir,
    seed=data_key,
    batch_size=batch_size,
    name=ds_name,
    one_hot=True,
)
training_config = TrainingConfig.from_dict(
    dict(
        diffusion_steps=500,
        diffusion_noise_schedule="cosine",
        learning_rate=1e-3,
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
    batch_size=batch_size,
    in_node_features=tu_dataset.max_node_feature,
    in_edge_features=tu_dataset.max_edge_feature,
    n=tu_dataset.n,
    num_layers=1,
)
mate.wandb()
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
    nodes_dist=tu_dataset.nodes_dist,
    nodes_prior=tu_dataset.node_prior,
    edges_prior=tu_dataset.edge_prior,
)
mate.result({f"{ds_name} best_val_loss": best_val_loss})
