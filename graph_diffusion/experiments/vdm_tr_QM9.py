"""
```yaml
tags:
    - book
```

"""

from ..shared.graph_distribution import GraphDistribution
from ..data_loaders.qm92 import load_data
import tensorflow as tf

tf.config.experimental.set_visible_devices([], "GPU")


import jax

jax.config.update("jax_platform_name", "cpu")  # run on CPU for now.

from mate import mate


# train_iter, eval_iter, data_info = load_data(
#     save_path=mate.data_dir, batch_size=10, seed=0
# )
#
batch_size = 32
dataset = load_data(
    save_dir=mate.data_dir,
    batch_size=batch_size,
)


import os  # nopep8


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Disable TF info/warnings # nopep8
import jax
import tensorflow as tf
import jax.numpy as jnp
from ..trainers.vdm_trainer import VDMTrainer, VDMTrainingConfig
from ..models.graph_transformer import GraphTransformerSimple
from ..models.vdm import VDM, VDMConfig
from ..shared.graph import Graph
from ..shared.encoder_decoder import EncoderDecoder

print(f"Using device: {jax.lib.xla_bridge.get_backend().platform}")

model, params = GraphTransformerSimple.initialize(
    key=jax.random.PRNGKey(0),
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
    gamma_type="learnable_scalar",  # "fixed",  # "learnable_scalar",  # learnable_scalar / learnable_nnet / fixed
    gamma_min=0.3,
    gamma_max=5.3,
    sm_n_timesteps=0,
    num_time_embedding=10,
    sm_n_layer=32,
    sm_pdrop=0.1,
)


encoder_decoder = EncoderDecoder(
    edge_vocab_size=len(dataset.edge_prior),
    node_vocab_size=len(dataset.node_prior),
)

# gat, params = GAT.initialize(
#     jax.random.PRNGKey(0), n=tu_dataset.n, n_heads_per_layer=(1, 1)
# )

vmd, vmd_params = VDM.create(
    config=vdm_config,
    example_input=GraphDistribution.create(
        nodes=jnp.zeros((2, dataset.n, len(dataset.node_prior)), dtype=float),
        e=jnp.zeros((2, dataset.n, dataset.n, len(dataset.edge_prior)), dtype=float),
        nodes_counts=jnp.ones((2,), dtype=int),
        edges_counts=jnp.ones((2,), dtype=int),
    ),
    probability_model=model,
    rng=jax.random.PRNGKey(0),
    encoder_decoder=encoder_decoder,
)

training_config = VDMTrainingConfig()

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