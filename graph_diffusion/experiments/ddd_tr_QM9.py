"""
```yaml
tags:
- research 
```
"""

import platform
import jax

# jax.config.update("jax_platform_name", "cpu")  # run on CPU for now.

jax.config.update("jax_debug_nans", True)

if not platform.system() == "Darwin":
    import tensorflow as tf

    tf.config.experimental.set_visible_devices([], "GPU")
import jax
from jax import config
from jax.lib import xla_bridge


from mate import mate

# from ..data_loaders.tu import load_data
from ..data_loaders.qm92 import load_data
import os
from ..models.graph_transformer import GraphTransformer
from ..trainers.ddd_trainer import run_model, TrainingConfig
import ipdb
from jax import numpy as np
from jax import random
from rich import print

# data_module = QM9DataModule(
#     mate.data_dir,
#     remove_h=True,
#     train_batch_size=32,
#     val_batch_size=32,
#     test_batch_size=32,
# )
# data_module.prepare_data()
#
# ds_infos = QM9Infos(data_module, remove_h=True)
#

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Disable TF info/warnings # nopep8

print(f"Using device: [yellow]{xla_bridge.get_backend().platform} [/yellow]")
batch_size = 5

data_key = random.PRNGKey(0)
ds_name = "QM9"  # "PTC_MR"# "MUTAG"
dataset = load_data(
    save_dir=mate.data_dir,
    batch_size=batch_size,
)

training_config = TrainingConfig.from_dict(
    dict(
        diffusion_steps=15,
        diffusion_noise_schedule="cosine",
        learning_rate=1e-3,
        lambda_train=(5, 0),
        transition="marginal",
        number_chain_steps=50,
        log_every_steps=4,
        max_num_nodes=dataset.n,
        num_node_features=dataset.max_node_feature,
        num_edge_features=dataset.max_edge_feature,
    )
)
rngs = {"params": random.PRNGKey(0), "dropout": random.PRNGKey(1)}
model, params = GraphTransformer.initialize(
    key=rngs["params"],
    in_node_features=dataset.node_prior.shape[-1],
    in_edge_features=dataset.edge_prior.shape[-1],
    number_of_nodes=dataset.n,
    num_layers=3,
)
best_val_loss = run_model(
    config=training_config,
    model=model,
    params=params,
    rngs=rngs,
    lr=training_config.learning_rate,
    train_loader=dataset.train_loader,
    val_loader=dataset.test_loader,
    num_epochs=200,
    action=mate.command if mate.command else "train",
    save_path=mate.save_dir,
    ds_name=ds_name,
    nodes_dist=dataset.nodes_dist,
    nodes_prior=dataset.node_prior,
    edges_prior=dataset.edge_prior,
    bits_per_edge=False,
)
mate.result({f"{ds_name} best_val_loss": best_val_loss})
