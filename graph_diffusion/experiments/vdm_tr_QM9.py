"""
```yaml
tags:
    - book
```

"""
# import ipdb;
# ipdb.set_trace()

import jax

key = jax.random.PRNGKey(0)

jax.config.update("jax_debug_nans", True)
# import tensorflow as tf
#
# tf.config.experimental.set_visible_devices([], "GPU")
#
from ..data_loaders.qm92 import load_data


import jax

# jax.config.update("jax_platform_name", "cpu")  # run on CPU for now.

from mate import mate


# train_iter, eval_iter, data_info = load_data(
#     save_path=mate.data_dir, batch_size=10, seed=0
# )
#
batch_size = 5
dataset = load_data(
    save_dir=mate.data_dir,
    batch_size=batch_size,
    onehot=True,
)


import os  # nopep8


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Disable TF info/warnings # nopep8

import jax
import tensorflow as tf
import jax.numpy as jnp
from ..trainers.vdm_trainer import VDMTrainer, VDMTrainingConfig
from ..models.graph_transformer import GraphTransformerNew
from ..models.vdm import VDM, VDMConfig
from ..shared.graph.graph_distribution import VariationalGraphDistribution

# Graph = graph.Graph

# from ..shared.encoder_decoder import EncoderDecoder

print(f"Using device: {jax.lib.xla_bridge.get_backend().platform}")
num_time_embedding = 15

model, params = GraphTransformerNew.initialize(
    key=key,
    in_node_features=dataset.node_prior.shape[-1],
    in_edge_features=dataset.edge_prior.shape[-1],
    number_of_nodes=dataset.n,
    num_layers=3,
)

# train_iter, eval_iter = create_dataset(dataset_config, jax.random.PRNGKey(0))


# upper_bound = 10 * (5 + 13.3) / 255
# import ipdb; ipdb.set_trace()
#
vdm_config = VDMConfig(
    vocab_size=256,
    sample_softmax=False,
    antithetic_time_sampling=True,
    with_fourier_features=True,
    with_attention=False,
    # configurations of the noise schedule
    gamma_type="cosine",  # "fixed",  # "learnable_scalar",  # "fixed",  # "learnable_scalar",  # learnable_scalar / learnable_nnet / fixed
    gamma_min=-7.0,
    gamma_max=2.5,
    sm_n_timesteps=0,
    num_time_embedding=num_time_embedding,
    sm_n_layer=32,
    sm_pdrop=0.1,
)


# gat, params = GAT.initialize(
#     jax.random.PRNGKey(0), n=tu_dataset.n, n_heads_per_layer=(1, 1)
# )

vmd, vmd_params = VDM.create(
    config=vdm_config,
    example_input=VariationalGraphDistribution.create(
        nodes=jnp.zeros((2, dataset.n, dataset.node_prior.shape[0]), dtype=float),
        edges=jnp.zeros(
            (2, dataset.n, dataset.n, dataset.edge_prior.shape[0]), dtype=float
        ),
        nodes_counts=jnp.ones((2,), dtype=int),
        edges_counts=jnp.ones((2,), dtype=int),
        # edge_vocab_size=jnp.array(len(dataset.edge_prior)),
        # node_vocab_size=jnp.array(len(dataset.node_prior)),
    ),
    probability_model=model,
    rng=jax.random.PRNGKey(0),
)


training_config = VDMTrainingConfig(
    lr=1e-3,
    weight_decay=1e-7,
    plot_location=mate.plots_dir,
)

experiment = VDMTrainer(
    config=training_config,
    train_iter=dataset.train_loader,
    eval_iter=dataset.test_loader,
    model=vmd,
    params=vmd_params,
)
mate.wandb()
if mate.command == "train":
    workdir = os.path.join(mate.save_dir, "workdir")
    # logging.info("Training at workdir: " + FLAGS.workdir)
    experiment.train_and_evaluate(workdir)
elif mate.command == "eval":
    experiment.evaluate(mate.checkpoint_dir)
else:
    raise Exception("Unknown FLAGS.mode")
