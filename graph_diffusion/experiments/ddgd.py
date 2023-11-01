from jax import random

import jax

# has to be done at the beginning to allow JAX to take control of the GPU memory.
data_key = jax.random.PRNGKey(0)


import jax
from jax import config
from jax.lib import xla_bridge
from mate import mate
import os
from jax import numpy as np
from jax import random
from rich import print
import random as pyrandom
from ..data_loaders.qm92 import load_data
from ..models.gt_digress import GraphTransformer
from ..trainers.ddgd_trainer import Trainer
import ipdb

assert mate is not None

diffusion_type = getattr(Trainer.DiffusionType, mate.diffusion_type)
force_cpu = mate.force_cpu
do_jit = mate.do_jit
debug_compiles = mate.debug_compiles


# def run(
#     diffusion_type: Trainer.DiffusionType,
#     force_cpu: bool,
#     do_jit: bool,
#     debug_compiles: bool,
# ):
if force_cpu:
    jax.config.update("jax_platform_name", "cpu")  # run on CPU for now.

jax.config.update("jax_log_compiles", debug_compiles)
jax.config.update("jax_debug_nans", True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Disable TF info/warnings # nopep8

device = xla_bridge.get_backend().platform
print(f"Using device: [yellow]{device} [/yellow]")
if device.lower() == "cpu":
    print("[red]WARNING: :skull:[/red] Running on CPU. This will be slow.")

if not do_jit:
    print("[red]WARNING: :skull:[/red] Running without JIT. This will be slow.")
if device in ["gpu", "cpu"]:
    batch_size = mate.local_batch_size
else:
    batch_size = mate.remote_batch_size
print(f"{batch_size=}")
dataset = load_data(
    save_dir=mate.data_dir,
    batch_size=batch_size,
)

diffusion_steps = mate.diffusion_steps

rngs = {
    "params": random.PRNGKey(pyrandom.randint(0, 10000)),
    "dropout": random.PRNGKey(pyrandom.randint(0, 10000)),
}
save_path = os.path.join(mate.results_dir, f"{diffusion_steps}_diffusion_steps")
os.makedirs(save_path, exist_ok=True)
os.makedirs(os.path.join(save_path, "plots"), exist_ok=True)

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
    structure_nodes_prior=dataset.structure_node_prior,
    structure_edges_prior=dataset.structure_edge_prior,
    dataset_infos=dataset.infos,
    bits_per_edge=False,
    diffusion_steps=diffusion_steps,
    noise_schedule_type="cosine",
    learning_rate=0.0001,
    log_every_steps=4,
    max_num_nodes=dataset.n,
    num_node_features=dataset.max_node_feature,
    num_edge_features=dataset.max_edge_feature,
    train_smiles=dataset.train_smiles,
    diffusion_type=diffusion_type,
)
with jax.disable_jit(not do_jit):
    mate.bind(trainer)


# mate.result({f"{ds_name} best_val_loss": best_val_loss})
