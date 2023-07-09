"""
```yaml
tags:
- rocket 
```
"""


import jax

data_key = jax.random.PRNGKey(
    0
)  # has to be done at the beginning to allow JAX to take control of the GPU memory.


gpu = True 
debug_compiles = False

if not gpu:
    jax.config.update("jax_platform_name", "cpu")  # run on CPU for now.

jax.config.update("jax_log_compiles", debug_compiles)
jax.config.update("jax_debug_nans", True)


import jax
from jax import config
from jax.lib import xla_bridge


from mate import mate

#from ..data_loaders.qm92 import load_data
from ..data_loaders.qm9_digress import load_data
import os
from ..models.graph_transformer import GraphTransformerGraphDistribution
from ..trainers.ddgd_trainer import Trainer
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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Disable TF info/warnings # nopep8

print(f"Using device: [yellow]{xla_bridge.get_backend().platform} [/yellow]")
batch_size = 100

ds_name = "QM9"  # "PTC_MR"# "MUTAG"
dataset = load_data(
    save_dir=mate.data_dir,
    batch_size=batch_size,
)
diffusion_steps = 500
import random as pyrandom

rngs = {
    "params": random.PRNGKey(pyrandom.randint(0, 10000)),
    "dropout": random.PRNGKey(pyrandom.randint(0, 10000)),
}
model, params = GraphTransformerGraphDistribution.initialize(
    key=rngs["params"],
    in_node_features=dataset.node_prior.shape[-1],
    in_edge_features=dataset.edge_prior.shape[-1],
    number_of_nodes=dataset.n,
    num_layers=5,
)
save_path = os.path.join(mate.save_dir, f"{diffusion_steps}_diffusion_steps")
os.makedirs(save_path, exist_ok=True)
os.makedirs(os.path.join(save_path, "plots"), exist_ok=True)

# import ipdb;
# ipdb.set_trace()
trainer = Trainer(
    model=model,
    params=params,
    rngs=rngs,
    train_loader=dataset.train_loader,
    val_loader=dataset.test_loader,
    num_epochs=200,
    match_edges=True,
    save_path=save_path,
    nodes_dist=dataset.nodes_dist,
    nodes_prior=dataset.node_prior,
    edges_prior=dataset.edge_prior,
    bits_per_edge=False,
    diffusion_steps=diffusion_steps,
    noise_schedule_type="cosine",
    learning_rate=1e-4,
    log_every_steps=4,
    max_num_nodes=dataset.n,
    num_node_features=dataset.max_node_feature,
    num_edge_features=dataset.max_edge_feature,
)
mate.bind(trainer)
# mate.result({f"{ds_name} best_val_loss": best_val_loss})
