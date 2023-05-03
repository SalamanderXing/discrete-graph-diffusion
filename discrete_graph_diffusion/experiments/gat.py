import tensorflow as tf

tf.config.experimental.set_visible_devices([], "GPU")
import jax
from flax.linen import init
from jax import config
from jax.lib import xla_bridge

# jax.config.update("jax_platform_name", "cpu")  # run on CPU for now.

from mate import mate
from ..data_loaders.qm9_p import QM9DataModule, QM9Infos, get_train_smiles
import os
from ..models.gat import GAT
from ..trainers.discrete_denoising_diffusion import run_model, TrainingConfig
import ipdb
from jax import numpy as np
from jax import random

print(f"Using device: {xla_bridge.get_backend().platform}")
remove_h = True
batch_size = 1000
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
input_dims = {
    "v": 4,
    "e": 5,
    "n": 9,
}
model, params = GAT.initialize(
    key=rngs["params"],
    batch_size=batch_size,
    in_node_features=input_dims["v"],
    in_edge_features=input_dims["e"],
    n=input_dims["n"],
)

run_model(
    config=training_config,
    model=model,
    params=params,
    rngs=rngs,
    lr=training_config.learning_rate,
    train_loader=datamodule.train_dataloader(),
    val_loader=datamodule.val_dataloader(),
    num_epochs=100,
    action=mate.command if mate.command else "train",
    save_path=mate.save_dir,
)
