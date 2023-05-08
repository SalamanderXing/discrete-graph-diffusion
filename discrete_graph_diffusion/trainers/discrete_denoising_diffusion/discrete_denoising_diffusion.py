"""
Entrypoint of the discrete denoising diffusion model.
The function `train_model` is the main entrypoint.
"""
import os
from typing import Callable, cast
from jax import numpy as np, Array
from flax.training import train_state
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# import orbax.checkpoint
import flax.linen as nn

# from flax.training import orbax_utils
from jax.debug import print as jprint  # type: ignore
from jaxtyping import Float, Int, Bool
from jax.experimental.checkify import checkify
from time import time
import jax_dataclasses as jdc
import jax
import optax
import wandb
import ipdb

# from rich import print as rprint
from tqdm import tqdm
from dataclasses import dataclass
from flax.core.frozen_dict import FrozenDict
from mate.jax import SFloat, SInt, SBool, SUInt, typed, Key, jit


from . import diffusion_functions as df
from .config import TrainingConfig
from .diffusion_types import (
    GraphDistribution,
    TransitionModel,
)


class TrainState(train_state.TrainState):
    key: jax.random.KeyArray  # type: ignore


@jdc.pytree_dataclass
class GetProbabilityFromState(jdc.EnforcedAnnotationsMixin):
    state: TrainState
    dropout_rng: Key
    transition_model: TransitionModel

    def __call__(
        self, g: GraphDistribution, t: Int[Array, "batch_size"]
    ) -> GraphDistribution:
        temporal_embeddings = self.transition_model.temporal_embeddings[t][:, None]
        pred_x, pred_e = self.state.apply_fn(
            self.state.params,
            g.x + temporal_embeddings,
            g.e,
            g.mask,  # .astype(float),
            rngs={"dropout": self.dropout_rng},
            # deterministic=True,
        )
        pred = GraphDistribution.masked(
            x=jax.nn.softmax(pred_x, -1),
            e=jax.nn.softmax(pred_e, -1),
            mask=g.mask,
        )
        return pred


@jdc.pytree_dataclass
class GetProbabilityFromParams(jdc.EnforcedAnnotationsMixin):
    params: FrozenDict
    state: TrainState
    dropout_rng: Key
    transition_model: TransitionModel
    deterministic: SBool = False

    def __call__(
        self, g: GraphDistribution, t: Int[Array, "batch_size"]
    ) -> GraphDistribution:
        temporal_embeddings = self.transition_model.temporal_embeddings[t][:, None]
        pred_x, pred_e = self.state.apply_fn(
            self.params,
            g.x + temporal_embeddings,
            g.e,
            g.mask,  # .astype(float),
            rngs={"dropout": self.dropout_rng},
            # deterministic=False,
        )
        pred = GraphDistribution.masked(
            x=jax.nn.softmax(pred_x, -1),
            e=jax.nn.softmax(pred_e, -1),
            mask=g.mask,
        )
        return pred


@typed
def compute_train_loss(
    pred_graph: GraphDistribution,
    q_g_s: GraphDistribution,
    lambda_train_e: SFloat = 0.0,
):
    loss_fn = optax.softmax_cross_entropy  # (logits=x, labels=y).sum()
    true_x = q_g_s.x.reshape(-1, q_g_s.x.shape[-1])  # (bs*n, dx)
    true_e = q_g_s.e.reshape(-1, q_g_s.e.shape[-1])  # (bs*n*n, de)
    pred_x = pred_graph.x.reshape(-1, pred_graph.x.shape[-1])  # (bs*n, dx)
    pred_e = pred_graph.e.reshape(-1, pred_graph.e.shape[-1])  # (bs*n*n, de)

    # Remove masked rows
    mask_x = (true_x != 0).any(axis=-1)
    mask_e = (true_e != 0).any(axis=-1)

    loss_x = np.where(mask_x, loss_fn(pred_x, true_x).mean(-1), 0).sum() / mask_x.sum()
    loss_e = np.where(mask_e, loss_fn(pred_e, true_e).mean(-1), 0).sum() / mask_e.sum()

    # return loss_x + lambda_train_e * loss_e  # + lambda_train_y * loss_y
    return loss_x + loss_e


@typed
def compute_val_loss(
    *,
    diffusion_steps: SInt,
    target: GraphDistribution,
    transition_model: TransitionModel,
    get_probability: Callable[
        [GraphDistribution, Int[Array, "batch_size"]], GraphDistribution
    ],
    rng_key: Key,
):
    # TODO: ask jamie, can you guess why there were other terms in this loss in the original code?
    # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
    kl_prior = df.kl_prior(
        target=target,
        diffusion_steps=diffusion_steps,
        transition_model=transition_model,
    )
    # 3. Diffusion loss
    loss_all_t = df.compute_lt(
        rng=rng_key,
        g=target,
        n_t_samples=1,
        diffusion_steps=diffusion_steps,
        transition_model=transition_model,
        get_probability=get_probability,
    )
    # 4. Reconstruction loss
    # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
    reconstruction_logp = df.reconstruction_logp(
        rng_key=rng_key,
        g=target,
        transition_model=transition_model,
        get_probability=get_probability,
        n_samples=1,
    )
    # jprint(
    #     "{kl_prior} {loss_all_t} {reconstruction_logp}",
    #     kl_prior=kl_prior,
    #     loss_all_t=loss_all_t,
    #     reconstruction_logp=reconstruction_logp,
    # )
    return kl_prior + loss_all_t - reconstruction_logp


def print_gradient_analysis(grads: FrozenDict):
    there_are_nan = np.any(
        np.array([np.any(np.isnan(v)) for v in jax.tree_util.tree_flatten(grads)[0]])
    )
    max_value = np.max(
        np.array([np.max(v) for v in jax.tree_util.tree_flatten(grads)[0]])
    )
    min_value = np.min(
        np.array([np.min(v) for v in jax.tree_util.tree_flatten(grads)[0]])
    )
    jprint("gradient analysis")
    jprint(
        "max_val={max_value} min_val={min_value}",
        max_value=max_value,
        min_value=min_value,
    )
    jprint("there are nans: {there_are_nan}", there_are_nan=there_are_nan)


@typed
def train_step(
    *,
    g: GraphDistribution,
    i: SInt,
    state: TrainState,
    rng: Key,
    diffusion_steps: SInt,
    transition_model: TransitionModel,
):  # -> tuple[TrainState, Float[Array, "batch_size"]]:
    dropout_train_key = jax.random.fold_in(key=rng, data=state.step)
    # t, z_t = df.apply_random_noise(
    #     graph=g,
    #     test=np.array(False),
    #     rng=rng,
    #     diffusion_steps=diffusion_steps,
    #     transition_model=transition_model,
    # )
    # temporal_embeddings = transition_model.temporal_embeddings[t][:, None]
    # z_t = z_t.set("x", z_t.x + temporal_embeddings)
    # q_s = transition_model.q_bars[t - 1]
    # q_g_s = g @ q_s

    def loss_fn(params: FrozenDict):
        # pred_x, pred_e = state.apply_fn(  # TODO: suffix dist for distributions
        #     params,
        #     z_t.x,
        #     z_t.e,
        #     z_t.mask,
        #     rngs={"dropout": dropout_train_key},
        # )
        # pred = GraphDistribution.masked(
        #     x=pred_x,
        #     e=pred_e,
        #     mask=z_t.mask,
        # )  # consider changing the name of this class to something that suggests ditribution

        # loss = compute_train_loss(
        #     pred_graph=pred,
        #     q_g_s=q_g_s,
        # )
        get_probability = GetProbabilityFromParams(
            state=state,
            params=params,
            transition_model=transition_model,
            dropout_rng=dropout_train_key,
        )
        loss = df.compute_lt(
            rng=rng,
            g=g,
            n_t_samples=1,
            diffusion_steps=diffusion_steps,
            transition_model=transition_model,
            get_probability=get_probability,
        )

        # jprint("loss {loss}", loss=loss)
        return loss.mean(), None

    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = gradient_fn(state.params)
    # print_gradient_analysis(grads)
    # grads = jax.tree_map(
    #     lambda x: np.where(np.isnan(x), 0.0, x), grads
    # )  # remove the nans :(
    state = state.apply_gradients(grads=grads)
    return state, loss


@typed
def run_model(
    *,
    model: nn.Module,
    train_loader,
    val_loader,
    params: FrozenDict,
    num_epochs=1,
    rngs: dict[str, Key],
    lr: float,
    config: TrainingConfig,
    action: str,  # 'train' or 'test'
    save_path: str,
    ds_name: str,
) -> float:
    state, transition_model = setup(
        model=model,
        params=params,
        lr=lr,
        config=config,
        rngs=rngs,
        epochs=num_epochs,
    )
    if action == "test":
        best_val_loss, val_time = val_epoch(
            rng=rngs["params"],
            state=state,
            val_loader=val_loader,
            config=config,
            transition_model=transition_model,
        )
        print(f"Validation loss: {best_val_loss:.4f} time: {val_time:.4f}")
    elif action == "train":
        val_loss, val_time = val_epoch(
            rng=rngs["params"],
            state=state,
            val_loader=val_loader,
            config=config,
            transition_model=transition_model,
        )
        print(f"Validation loss: {val_loss:.4f} time: {val_time:.4f}")
        best_val_loss = train_all_epochs(
            num_epochs=num_epochs,
            rngs=rngs,
            state=state,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            transition_model=transition_model,
            save_path=save_path,
            ds_name=ds_name,
        )
    elif action == "sample":
        get_probability = GetProbabilityFromState(
            state=state, dropout_rng=rngs["dropout"], transition_model=transition_model
        )
        df.sample_batch(
            rng_key=rngs["params"],
            diffusion_steps=config.diffusion_steps,
            get_probability=get_probability,
            batch_size=10,
            n=9,
            node_embedding_size=config.node_embedding_size,
            edge_embedding_size=config.edge_embedding_size,
        )
    else:
        raise ValueError(f"Unknown action {action}")

    return best_val_loss


def train_all_epochs(
    num_epochs: int,
    rngs: dict[str, Key],
    state: TrainState,
    train_loader,
    val_loader,
    config: TrainingConfig,
    transition_model: TransitionModel,
    save_path: str,
    ds_name: str,
):
    val_losses = []
    train_losses = []
    for epoch_idx in range(1, num_epochs + 1):
        rng, _ = jax.random.split(rngs["params"])
        print(f"Epoch {epoch_idx}")
        state, train_loss, train_time = train_epoch(
            rng=rng,
            state=state,
            train_loader=train_loader,
            epoch=epoch_idx,
            diffusion_steps=config.diffusion_steps,
            transition_model=transition_model,
        )
        print(f"Train loss: {train_loss:.4f} time: {train_time:.4f}")
        val_loss, val_time = val_epoch(
            rng=rng,
            state=state,
            val_loader=val_loader,
            config=config,
            transition_model=transition_model,
        )

        # wandb.log(
        #     {
        #         "train_loss": train_loss,
        #         "val_loss": val_loss,
        #         "epoch": epoch_idx,
        #     }
        # )
        wandb.log("train_loss", train_loss, epoch_idx)
        wandb.log("val_loss", val_loss, epoch_idx)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"Validation loss: current={val_loss:.4f} best={min(val_losses):.5f} time: {val_time:.4f}"
        )
        import json

        with open(os.path.join(save_path, "val_losses.json"), "w") as f:
            json.dump(val_losses, f)

        with open(os.path.join(save_path, "train_losses.json"), "w") as f:
            json.dump(train_losses, f)

        # makes a plot separated horizontally into two subplots
        # the first plot is the train loss
        # the second plot is the val loss
        # the x axis is the epoch

        _, axs = plt.subplots(2, 1)
        axs[0].plot(train_losses)
        axs[0].set_title("Train loss")
        axs[1].plot(val_losses)
        axs[1].set_title("Val loss")
        plt.savefig(os.path.join(save_path, "losses.png"))
        plt.close()

        if val_loss == min(val_losses):
            print("Saving model")
            import pickle

            pickle.dump(
                state.params, open(os.path.join(save_path, "checkpoint.pickle"), "wb")
            )


def setup(
    model: nn.Module,
    params: FrozenDict,
    rngs: dict[str, Key],
    lr: float,
    config: TrainingConfig,
    epochs: int,
):
    cosine_decay_scheduler = optax.cosine_decay_schedule(
        0.0001, decay_steps=epochs, alpha=0.95
    )
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(cosine_decay_scheduler),
        key=rngs["dropout"],
    )
    """
    The following is the prior of the data. It is used to generate matrices Q_t, 
    and also to compute the prior loss term in the ELBO.
    """
    # Key idea proposed in the DiGress paper: use the prior distribution to gnerate Q_t
    # These priors are extracted from the dataset.
    # transition_model = TransitionModel.create(
    #     x_priors=np.array(
    #         [
    #             0.7230000495910645,
    #             0.11510000377893448,
    #             0.15930001437664032,
    #             0.0026000002399086952,
    #         ]
    #     ),
    #     e_priors=np.array(
    #         [
    #             0.7261000275611877,
    #             0.23839999735355377,
    #             0.027400000020861626,
    #             0.008100000210106373,
    #             0.0,
    #         ]
    #     ),
    #     diffusion_steps=config.diffusion_steps,
    #     temporal_embedding_dim=4,  # FIXME: same as the node embedding. Should improve this parametrization
    # )
    transition_model = TransitionModel.create(
        x_priors=np.ones(config.num_node_features) / config.num_node_features,
        e_priors=np.ones(config.num_edge_features) / config.num_edge_features,
        diffusion_steps=config.diffusion_steps,
        temporal_embedding_dim=config.num_node_features,  # FIXME: same as the node embedding. Should improve this parametrization
    )
    return state, transition_model


# @jit
@typed
def val_step(
    *,
    dense_data: GraphDistribution,
    state: TrainState,
    diffusion_steps: SInt,
    rng: Key,
    transition_model: TransitionModel,
) -> Float[Array, "batch_size"]:
    dropout_test_key = jax.random.fold_in(key=rng, data=state.step)

    get_probability = GetProbabilityFromState(
        state=state, dropout_rng=dropout_test_key, transition_model=transition_model
    )
    return compute_val_loss(
        target=dense_data,
        diffusion_steps=diffusion_steps,
        transition_model=transition_model,
        rng_key=rng,
        get_probability=get_probability,
    )


@typed
def val_epoch(
    *,
    val_loader,
    state: TrainState,
    config: TrainingConfig,
    rng: Key,
    transition_model: TransitionModel,
) -> tuple[float, float]:
    run_loss = []
    t0 = time()
    for i, (x, e, mask) in enumerate(tqdm(val_loader)):
        loss = val_step(
            dense_data=GraphDistribution.masked(x, e, mask.astype(bool)),
            state=state,
            diffusion_steps=config.diffusion_steps,
            rng=rng,
            transition_model=transition_model,
        )
        run_loss.extend(loss.reshape(-1).tolist())
    t1 = time()
    total_time = t1 - t0
    avg_loss = np.mean(np.array(run_loss)).tolist()
    return avg_loss, total_time


@typed
def train_epoch(
    *,
    train_loader,
    epoch: int,
    state: TrainState,
    diffusion_steps: SInt,
    rng: Key,
    transition_model: TransitionModel,
):
    run_losses = []
    tot_len: int = 0
    t0 = time()
    for batch_index, (x, e, mask) in enumerate(tqdm(train_loader)):
        # dense_data = GraphDistribution.from_sparse_torch(batch)
        dense_data = GraphDistribution.masked(x, e, mask.astype(bool))
        state, loss = train_step(
            g=dense_data,
            i=np.array(epoch),
            state=state,
            diffusion_steps=diffusion_steps,
            rng=rng,
            transition_model=transition_model,
        )
        run_losses.append(loss)
    t1 = time()
    tot_time = t1 - t0
    avg_loss = np.mean(np.array(run_losses)).tolist()
    return state, avg_loss, tot_time
