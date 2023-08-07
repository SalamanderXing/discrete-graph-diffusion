"""
```yaml
tags:
- rocket 
```
"""
import jax
from jax import random
from jax.lib import xla_bridge
import ipdb
from rich import print

data_key = jax.random.PRNGKey(0)


gpu = True
debug_compiles = False

if not gpu:
    jax.config.update("jax_platform_name", "cpu")  # run on CPU for now.

jax.config.update("jax_log_compiles", debug_compiles)
jax.config.update("jax_debug_nans", True)

device = xla_bridge.get_backend().platform
print(f"Using device: [yellow]{device} [/yellow]")
if device.lower() == "cpu":
    print("[red]WARNING: :skull:[/red] Running on CPU. This will be slow.")


import jax
from jax import config
from ..data_loaders.tu import load_data
from mate import mate
import os
from ..models.gt_digress import GraphTransformer
from ..trainers.ddgd_trainer import Trainer
from jax import numpy as np
from jax import random


batch_size = 200

dataset = load_data(
    name="ZINC_full",
    seed=32,
    save_path=mate.data_dir,
    train_batch_size=batch_size,
    test_batch_size=batch_size * 2,
    one_hot=True,
    # filter_graphs_by_max_node_count=10,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Disable TF info/warnings # nopep8


diffusion_steps = 500
import random as pyrandom

rngs = {
    "params": random.PRNGKey(pyrandom.randint(0, 10000)),
    "dropout": random.PRNGKey(pyrandom.randint(0, 10000)),
}
# model, params = GraphTransformer.initialize(
#     key=rngs["params"],
#     in_node_features=dataset.node_prior.shape[-1],
#     in_edge_features=dataset.edge_prior.shape[-1],
#     number_of_nodes=dataset.n,
#     num_layers=5,
# )
# model = GraphTransformer(n_layers=5)
save_path = os.path.join(mate.save_dir, f"{diffusion_steps}_diffusion_steps")
os.makedirs(save_path, exist_ok=True)
os.makedirs(os.path.join(save_path, "plots"), exist_ok=True)

# from jaxtyping import Float, Array
#
#
# def dumbtest(a: Float[Array, "2"]) -> Float[Array, "2"]:
#     return a
#
#
# dumbtest(np.array([1.0]))


trainer = Trainer(
    model_class=GraphTransformer,
    rngs=rngs,
    train_loader=dataset.train_loader,
    val_loader=dataset.test_loader,
    num_epochs=300,
    match_edges=True,
    use_extra_features=False,
    save_path=save_path,
    nodes_dist=dataset.nodes_dist,
    feature_nodes_prior=dataset.feature_node_prior,
    feature_edges_prior=dataset.feature_edge_prior,
    bits_per_edge=False,
    diffusion_steps=diffusion_steps,
    noise_schedule_type="cosine",
    learning_rate=0.0002,
    log_every_steps=4,
    max_num_nodes=dataset.n,
    num_node_features=dataset.max_node_feature,
    num_edge_features=dataset.max_edge_feature,
    diffusion_type=Trainer.DiffusionType.simple,
    shuffle_coding_metric=True,
)
mate.bind(trainer)
