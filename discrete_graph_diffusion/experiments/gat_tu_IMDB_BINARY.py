import platform
import jax

# jax.config.update("jax_platform_name", "cpu")  # run on CPU for now.
jax.config.update("jax_debug_nans", True)
try:
    import tensorflow as tf

    tf.config.experimental.set_visible_devices([], "GPU")
except:
    pass
import jax
from jax import config
from jax.lib import xla_bridge

# jax.config.update("jax_platform_name", "cpu")  # run on CPU for now.

from mate import mate
from ..data_loaders.tu import load_data_no_attributes as load_data
import os
from ..models.gat import GAT
from ..trainers.discrete_denoising_diffusion import run_model, TrainingConfig
import ipdb
from jax import random
from rich import print

print(f"Using device: {xla_bridge.get_backend().platform}")
batch_size = 20
print(f"Batch size: {batch_size}")
data_key = random.PRNGKey(0)
ds_name = "IMDB-BINARY"
print(f"\nRunning on dataset: {ds_name}\n")
train_loader, test_loader, dataset_infos = load_data(
    save_path=mate.save_dir, seed=data_key, batch_size=batch_size, name=ds_name
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
        max_num_nodes=dataset_infos.max_num_nodes,
        num_node_features=dataset_infos.num_node_features,
        num_edge_features=dataset_infos.num_edge_features,
    )
)
rngs = {"params": random.PRNGKey(0), "dropout": random.PRNGKey(1)}
print("Initializing model")
model, params = GAT.initialize(
    key=rngs["params"],
    batch_size=batch_size,
    in_node_features=dataset_infos.num_node_features,
    in_edge_features=dataset_infos.num_edge_features,
    n=dataset_infos.max_num_nodes,
    num_layers=2,
)
print("Model initialized")
mate.wandb()
best_val_loss = run_model(
    config=training_config,
    model=model,
    params=params,
    rngs=rngs,
    lr=training_config.learning_rate,
    train_loader=train_loader,
    val_loader=test_loader,
    num_epochs=2,
    action=mate.command if mate.command else "train",
    save_path=mate.save_dir,
    ds_name=ds_name,
    nodes_dist=dataset_infos.nodes_dist,
)
mate.result({f"{ds_name} best_val_loss": best_val_loss})
