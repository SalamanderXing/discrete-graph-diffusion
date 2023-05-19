from ..data_loaders.tu import load_data
import tensorflow as tf

tf.config.experimental.set_visible_devices([], "GPU")


import jax

# jax.config.update("jax_platform_name", "cpu")  # run on CPU for now.

from mate import mate


# train_iter, eval_iter, data_info = load_data(
#     save_path=mate.data_dir, batch_size=10, seed=0
# )
#
tu_dataset = load_data(save_path=mate.data_dir, batch_size=150, seed=0)

import os  # nopep8


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Disable TF info/warnings # nopep8
import jax
import tensorflow as tf
import jax.numpy as jnp
from ..trainers.vdm import VDMTrainer, VDMTrainingConfig
from ..models.vdm import VDM, VDMConfig
from ..models.gat import GAT  # , EncoderDecoder
from ..shared.graph import Graph
from ..shared.encoder_decoder import EncoderDecoder

# from ..data_loaders.cifar10 import create_dataset, DatasetConfig

# flags.DEFINE_string("workdir", None, "Work unit directory.")
# flags.DEFINE_string("checkpoint", "", "Checkpoint to evaluate.")
# flags.DEFINE_string("mode", "train", "train / eval")
# flags.DEFINE_string("model", "vdm", "vdm")
# flags.mark_flags_as_required(["config", "workdir"])
# flags.DEFINE_string("log_level", "info", "info/warning/error")


# del argv
# if jax.process_index() == 0:
#     logging.set_verbosity(FLAGS.log_level)
# else:
#     logging.set_verbosity("error")
# logging.warning("=== Start of main() ===")
#
# Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
# it unavailable to JAX. (Not necessary with TPU.)
# tf.config.experimental.set_visible_devices([], "GPU")

# logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
# logging.info("JAX devices: %r", jax.devices())

print(f"Using device: {jax.lib.xla_bridge.get_backend().platform}")


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
    edge_vocab_size=tu_dataset.max_edge_feature,
    node_vocab_size=tu_dataset.max_node_feature,
)

# gat, params = GAT.initialize(
#     jax.random.PRNGKey(0), n=tu_dataset.n, n_heads_per_layer=(1, 1)
# )
gat = GAT(n_heads_per_layer=(2, 2))

vmd, vmd_params = VDM.create(
    config=vdm_config,
    example_input=Graph.create(
        **{
            "nodes": jnp.zeros((2, tu_dataset.n, 1), dtype=float),
            "edges": jnp.zeros((2, tu_dataset.n, tu_dataset.n), dtype=float),
            "nodes_counts": jnp.ones((2,), dtype=int),
            "edges_counts": jnp.ones((2,), dtype=int),
        }
    ),
    probability_model=gat,
    rng=jax.random.PRNGKey(0),
    encoder_decoder=encoder_decoder,
)

training_config = VDMTrainingConfig()

experiment = VDMTrainer(
    config=training_config,
    train_iter=tu_dataset.train_loader,
    eval_iter=tu_dataset.test_loader,
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
