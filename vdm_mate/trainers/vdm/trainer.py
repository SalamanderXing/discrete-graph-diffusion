# Copyright 2022 The VDM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from rich import print
import functools
import os
from typing import Any, Iterable
import wandb
import chex
from clu import periodic_actions
from clu import parameter_overview
from clu import metric_writers
from clu import checkpoint
import ipdb
from flax.core.frozen_dict import unfreeze, FrozenDict
import flax.jax_utils as flax_utils
import flax
from jax._src.random import PRNGKey
import jax
import jax.numpy as jnp
import numpy as np
import optax
from optax._src import base
from flax.struct import dataclass
from jax_tqdm import scan_tqdm

from flax import linen as nn
from mate.jax import Key, typed
from .train_state import TrainState
from . import utils


@dataclass
class TrainingConfig:
    num_steps_eval: int = 1
    steps_per_save: int = 1
    steps_per_eval: int = 1
    steps_per_logging: int = 1
    num_steps_train: int = 10
    seed: int = 0
    lr_decay: float = 0.98
    ckpt_restore_dir: str | None = None
    lr: float = 0.001
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    num_steps_lr_warmup: int = 1000
    optimizer_name: str = "adamw"
    gradient_clip_norm: float | None = None
    substeps: int = 1
    ema_rate: float = 0.999


class Trainer:
    """Boilerplate for training and evaluating VDM models."""

    def __init__(
        self,
        config: TrainingConfig,
        train_iter: Iterable,
        eval_iter: Iterable,
        model: nn.Module,
        params: FrozenDict,
    ):
        self.config = config

        # Set seed before initializing model.
        seed = config.seed
        self.rng = jax.random.PRNGKey(seed)

        # initialize dataset
        self.rng, data_rng = jax.random.split(self.rng)
        # self.train_iter, self.eval_iter = dataset.create_dataset(config, data_rng)
        self.train_iter = train_iter
        self.eval_iter = eval_iter

        # initialize model
        self.rng, model_rng = jax.random.split(self.rng)
        # self.model, params = self.get_model_and_params(model_rng)
        self.model = model
        parameter_overview.log_parameter_overview(params)

        # initialize train state
        print("=== Initializing train state ===")
        self.state: TrainState = TrainState.create(
            apply_fn=self.model.apply,
            variables=params,
            optax_optimizer=self.get_optimizer,
        )
        self.lr_schedule = self.get_lr_schedule()

        # Restore from checkpoint
        if self.config.ckpt_restore_dir is not None:
            ckpt_restore = checkpoint.Checkpoint(self.config.ckpt_restore_dir)
            checkpoint_to_restore = ckpt_restore.get_latest_checkpoint_to_restore_from()
            assert checkpoint_to_restore
            state_restore_dict = ckpt_restore.restore_dict(checkpoint_to_restore)
            self.state = restore_partial(self.state, state_restore_dict)
            del state_restore_dict, ckpt_restore, checkpoint_to_restore

        # initialize train/eval step
        print("=== Initializing train/eval step ===")
        self.rng, train_rng = jax.random.split(self.rng)

        # @scan_tqdm(1000)
        # def train_step(base_rng, state, batch):
        #     return self.train_step(base_rng, state, batch)

        self.p_train_step = functools.partial(self.train_step, train_rng)
        self.train_rng = train_rng
        # self.p_train_step = functools.partial(jax.lax.scan, self.p_train_step)
        # self.p_train_step = jax.pmap(self.p_train_step, "batch")

        self.rng, eval_rng, sample_rng = jax.random.split(self.rng, 3)
        self.eval_rng = eval_rng
        self.p_eval_step = functools.partial(self.eval_step, eval_rng)
        # self.p_eval_step = jax.pmap(self.p_eval_step, "batch")
        # self.p_sample = functools.partial(
        #     self.sample_fn,
        #     dummy_inputs=next(self.eval_iter)["images"][0],  # FIXME: no images maybe
        #     rng=sample_rng,
        # )
        # self.p_sample = utils.dist(
        #     self.p_sample, accumulate="concat", axis_name="batch"
        # )

        print("=== Done with Experiment.__init__ ===")

    def get_lr_schedule(self):
        learning_rate = self.config.lr
        # Create learning rate schedule
        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=learning_rate,
            transition_steps=self.config.num_steps_lr_warmup,
        )

        if self.config.lr_decay:
            decay_fn = optax.linear_schedule(
                init_value=learning_rate,
                end_value=0,
                transition_steps=self.config.num_steps_train
                - self.config.num_steps_lr_warmup,
            )
            schedule_fn = optax.join_schedules(
                schedules=[warmup_fn, decay_fn],
                boundaries=[self.config.num_steps_lr_warmup],
            )
        else:
            schedule_fn = warmup_fn

        return schedule_fn

    def get_optimizer(self, lr: float) -> base.GradientTransformation:
        """Get an optax optimizer. Can be overided."""

        def decay_mask_fn(params):
            flat_params = flax.traverse_util.flatten_dict(unfreeze(params))
            flat_mask = {
                path: (
                    path[-1] != "bias"
                    and path[-2:]
                    not in [("layer_norm", "scale"), ("final_layer_norm", "scale")]
                )
                for path in flat_params
            }
            return FrozenDict(flax.traverse_util.unflatten_dict(flat_mask))

        if self.config.optimizer_name == "adamw":
            optimizer = optax.adamw(
                learning_rate=lr,
                mask=decay_mask_fn,
                # **config.args,
            )
            if self.config.gradient_clip_norm is not None:
                clip = optax.clip_by_global_norm(self.config.gradient_clip_norm)
                optimizer = optax.chain(clip, optimizer)
        else:
            raise Exception("Unknow optimizer.")

        return optimizer

    @typed
    def train_and_evaluate(self, workdir: str):
        print("=== Experiment.train_and_evaluate() ===")
        # if jax.process_index() == 0:
        #  if not tf.io.gfile.exists(workdir):
        #    tf.io.gfile.mkdir(workdir)

        print(f"num_steps_train={self.config.num_steps_train}")

        # Get train state
        state = self.state

        # Set up checkpointing of the model and the input pipeline.
        checkpoint_dir = os.path.join(workdir, "checkpoints")
        ckpt = checkpoint.MultihostCheckpoint(checkpoint_dir, max_to_keep=5)
        checkpoint_to_restore = ckpt.get_latest_checkpoint_to_restore_from()
        # if checkpoint_to_restore:
        #     state = ckpt.restore_or_initialize(state, checkpoint_to_restore)
        initial_step = int(state.step)

        # Distribute training.

        # state = flax_utils.replicate(state)

        # Create logger/writer
        # writer = utils.create_custom_writer(workdir, jax.process_index())
        # if initial_step == 0:
        #     writer.write_hparams(dict(self.config))

        hooks = []
        # report_progress = periodic_actions.ReportProgress(
        #     num_train_steps=self.config.num_steps_train, writer=writer
        # )
        # if jax.process_index() == 0:
        #     hooks += [report_progress]
        #     if config.profile:
        #         hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]

        step = initial_step
        substeps = self.config.substeps

        print("=== Start of training ===")
        # the step count starts from 1 to num_steps_train
        while step < self.config.num_steps_train:
            is_last_step = step + substeps >= self.config.num_steps_train

            # batch = jax.tree_map(jnp.asarray, next(self.train_iter))
            # state, _train_metrics = self.p_train_step(state, batch)
            batch = jax.tree_map(jnp.asarray, next(self.train_iter))
            state, _train_metrics = self.train_step(self.train_rng, state, batch)
            # ipdb.set_trace()
            new_step = int(state.step)
            assert new_step == step + substeps
            step = new_step

            if step % self.config.steps_per_logging == 0 or is_last_step:
                print("=== Writing scalars ===")
                metrics = _train_metrics[
                    "scalars"
                ]  # flax_utils.unreplicate(_train_metrics["scalars"])
                metrics = {
                    key: np.mean(val)
                    for key, val in metrics.items()
                    if key == "train_bdb"
                }
                wandb.log(metrics, step)
                print(metrics)

            if step % self.config.steps_per_eval == 0 or is_last_step or step == 1000:
                print("=== Running eval ===")
                eval_metrics = {}
                for eval_step in range(self.config.num_steps_eval):
                    batch = self.eval_iter.next()
                    batch = jax.tree_map(jnp.asarray, batch)
                    metrics = self.eval_step(
                        self.eval_rng, state.ema_params, batch, eval_step
                    )
                    for key, val in metrics["scalars"].items():
                        eval_metrics[key] = eval_metrics.get(key, []) + [val]

                eval_metrics = {
                    key: np.mean(val)
                    for key, val in eval_metrics.items()
                    if key == "eval_bdb"
                }
                # average over eval metrics
                # eval_metrics = utils.get_metrics(eval_metrics)

                print(eval_metrics)
                wandb.log(eval_metrics, step)

                # print out a batch of images
                # metrics = flax_utils.unreplicate(metrics)
                # images = metrics["images"]
                # samples = self.p_sample(params=state.ema_params)
                # samples = utils.generate_image_grids(samples)[None, :, :, :]
                # images["samples"] = samples.astype(np.uint8)
                # wandb.log(step, images)

            # if step % self.config.steps_per_save == 0 or is_last_step:
            # ckpt.save(flax_utils.unreplicate(state))

    def evaluate(self, checkpoint_dir):
        """Perform one evaluation."""
        print("=== Experiment.evaluate() ===")
        ckpt = checkpoint.Checkpoint(checkpoint_dir)
        state_dict = ckpt.restore_dict()
        params = flax.core.FrozenDict(state_dict["ema_params"])
        step = int(state_dict["step"])

        # Distribute training.
        params = flax_utils.replicate(params)

        eval_metrics = []

        for eval_step in range(self.config.num_steps_eval):
            batch = self.eval_iter.next()
            batch = jax.tree_map(jnp.asarray, batch)
            metrics = self.p_eval_step(params, batch, flax_utils.replicate(eval_step))
            eval_metrics.append(metrics["scalars"])

        # average over eval metrics
        eval_metrics = utils.get_metrics(eval_metrics)
        # eval_metrics = jax.tree_map(jnp.mean, eval_metrics)

        # writer.write_scalars(step, eval_metrics)
        wandb.log(eval_metrics, step)

        # sample a batch of images
        samples = self.p_sample(params=params)
        samples = utils.generate_image_grids(samples)[None, :, :, :]
        samples = {"samples": samples.astype(np.uint8)}
        # writer.write_images(step, samples)

    def train_step(self, base_rng, state, batch):
        batch = {
            "images": batch[0],
            "conditioning": batch[1].squeeze(1),
        }
        # rng = jax.random.fold_in(base_rng, jax.lax.axis_index("batch"))
        rng = jax.random.fold_in(base_rng, state.step)
        # print('about to compute gradssss')
        grad_fn = jax.value_and_grad(self.loss_fn, has_aux=True)
        (_, metrics), grads = grad_fn(state.params, batch, rng=rng, is_train=True)
        # grads = jax.lax.pmean(grads, "batch")

        learning_rate = self.lr_schedule(state.step)
        new_state = state.apply_gradients(
            grads=grads, lr=learning_rate, ema_rate=self.config.ema_rate
        )
        # ipdb.set_trace()
        # metrics["scalars"] = jax.tree_map(lambda x: jnp.mean(x, -1), metrics["scalars"])
        metrics["scalars"] = {"train_" + k: v for (k, v) in metrics["scalars"].items()}

        # metrics["images"] = jax.tree_map(
        #     lambda x: utils.generate_image_grids(x)[None, :, :, :], metrics["images"]
        # )

        return new_state, metrics

    def eval_step(self, base_rng, params, batch, eval_step=0):
        batch = {
            "images": batch[0],
            "conditioning": batch[1].squeeze(1),
        }
        # rng = jax.random.fold_in(base_rng, jax.lax.axis_index("batch"))
        rng = jax.random.fold_in(base_rng, eval_step)

        _, metrics = self.loss_fn(params, batch, rng=rng, is_train=False)

        # summarize metrics
        # metrics["scalars"] = jax.lax.pmean(metrics["scalars"], axis_name="batch")
        metrics["scalars"] = {"eval_" + k: v for (k, v) in metrics["scalars"].items()}

        # metrics["images"] = jax.tree_map(
        #     lambda x: utils.generate_image_grids(x)[None, :, :, :], metrics["images"]
        # )

        return metrics

    @typed
    def loss_fn(self, params, inputs, rng, is_train):
        rng, sample_rng = jax.random.split(rng)
        rngs = {"sample": sample_rng}
        if is_train:
            rng, dropout_rng = jax.random.split(rng)
            rngs["dropout"] = dropout_rng

        # sample time steps, with antithetic sampling
        outputs = self.state.apply_fn(
            variables={"params": params},
            **inputs,
            rngs=rngs,
            deterministic=not is_train,
        )
        rescale_to_bpd = 1.0 / (np.prod(inputs["images"].shape[1:]) * np.log(2.0))
        bpd_latent = jnp.mean(outputs.loss_klz) * rescale_to_bpd
        bpd_recon = jnp.mean(outputs.loss_recon) * rescale_to_bpd
        bpd_diff = jnp.mean(outputs.loss_diff) * rescale_to_bpd
        bpd = bpd_recon + bpd_latent + bpd_diff
        scalar_dict = {
            "bpd": bpd,
            "bpd_latent": bpd_latent,
            "bpd_recon": bpd_recon,
            "bpd_diff": bpd_diff,
            "var0": outputs.var_0,
            "var": outputs.var_1,
        }
        # img_dict = {"inputs": inputs["images"]}
        metrics = {"scalars": scalar_dict}  # , "images": img_dict}

        return bpd, metrics

    @typed
    def sample_fn(self, *, dummy_inputs, rng, params):
        rng = jax.random.fold_in(rng, jax.lax.axis_index("batch"))

        if self.model.config.sm_n_timesteps > 0:
            T = self.model.config.sm_n_timesteps
        else:
            T = 1000

        conditioning = jnp.zeros((dummy_inputs.shape[0],), dtype="uint8")

        # sample z_0 from the diffusion model
        rng, sample_rng = jax.random.split(rng)
        z_init = jax.random.normal(sample_rng, dummy_inputs.shape)

        def body_fn(i, z_t):
            return self.state.apply_fn(
                variables={"params": params},
                i=i,
                T=T,
                z_t=z_t,
                conditioning=conditioning,
                rng=rng,
                method=self.model.sample,
            )

        z_0 = jax.lax.fori_loop(lower=0, upper=T, body_fun=body_fn, init_val=z_init)

        samples = self.state.apply_fn(
            variables={"params": params},
            z_0=z_0,
            method=self.model.generate_x,
        )

        return samples


def copy_dict(dict1, dict2):
    if not isinstance(dict1, dict):
        assert not isinstance(dict2, dict)
        return dict2
    for key in dict1.keys():
        if key in dict2:
            dict1[key] = copy_dict(dict1[key], dict2[key])

    return dict1


def restore_partial(state, state_restore_dict):
    state_dict = flax.serialization.to_state_dict(state)
    state_dict = copy_dict(state_dict, state_restore_dict)
    state = flax.serialization.from_state_dict(state, state_dict)

    return state
