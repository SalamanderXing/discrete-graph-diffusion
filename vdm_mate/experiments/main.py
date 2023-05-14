import tensorflow as tf

tf.config.experimental.set_visible_devices([], "GPU")

import jax
# jax.config.update("jax_platform_name", "cpu")  # run on CPU for now.

import os  # nopep8


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Disable TF info/warnings # nopep8
from mate import mate
import jax
import tensorflow as tf
import jax.numpy as jnp
from ..trainers.vdm import VDMTrainer, VDMTrainingConfig
from ..models.vdm import VDM, VDMConfig
from ..models.score_unet import ScoreUNet, ScoreUNetConfig, EncoderDecoder

# from ..data_loaders.cifar10 import create_dataset, DatasetConfig
from ..data_loaders.cifar10_simple import get_data_loaders

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
train_iter, eval_iter = get_data_loaders(
    train_batch_size=2, test_batch_size=10, save_dir=mate.data_dir
)


vdm_config = VDMConfig(
    vocab_size=256,
    sample_softmax=False,
    antithetic_time_sampling=True,
    with_fourier_features=True,
    with_attention=False,
    # configurations of the noise schedule
    gamma_type="learnable_scalar",  # learnable_scalar / learnable_nnet / fixed
    gamma_min=-13.3,
    gamma_max=5.0,
    # configurations of the score model
    sm_n_timesteps=0,
    sm_n_embd=128,
    sm_n_layer=32,
    sm_pdrop=0.1,
)

score_unet_config = ScoreUNetConfig(
    sm_n_embd=vdm_config.sm_n_embd,
    sm_n_layer=vdm_config.sm_n_layer,
    with_attention=True,
    sm_pdrop=vdm_config.sm_pdrop,
)
score_unet = ScoreUNet(score_unet_config)
encoder_decoder = EncoderDecoder(256)

vmd, vmd_params = VDM.create(
    config=vdm_config,
    example_input={
        "images": jnp.zeros((2, 32, 32, 3), "uint8"),
        "conditioning": jnp.zeros((2,)),
    },
    probability_model=score_unet,
    rng=jax.random.PRNGKey(0),
    encoder_decoder=encoder_decoder,
)

training_config = VDMTrainingConfig()

experiment = VDMTrainer(
    config=training_config,
    train_iter=train_iter,
    eval_iter=eval_iter,
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
