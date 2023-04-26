"""
Entrypoint of the discrete denoising diffusion model.
The function `train_model` is the main entrypoint.
"""
from typing import Callable
from jax import numpy as np, Array
from flax.training import train_state
import flax.linen as nn
from jax.debug import print as jprint
from jaxtyping import Float, Int, Bool
from jax.experimental.checkify import checkify
import jax
import optax
import ipdb
from rich import print
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
from .extra_features import extra_features


class TrainState(train_state.TrainState):
    key: jax.random.KeyArray


@typed
def train_loss(
    pred_graph: GraphDistribution,
    true_graph: GraphDistribution,
    lambda_train_e: SFloat = 5.0,
    lambda_train_y: SFloat = 0.0,
    loss_type: SInt = 1,  # 1 in l2, 2 = softmax_cross_entropy
):
    loss_fn = optax.l2_loss  # optax.softmax_cross_entropy  # (logits=x, labels=y).sum()
    true_x = true_graph.x.reshape(-1, true_graph.x.shape[-1])  # (bs*n, dx)
    true_e = true_graph.e.reshape(-1, true_graph.e.shape[-1])  # (bs*n*n, de)
    pred_x = pred_graph.x.reshape(-1, pred_graph.x.shape[-1])  # (bs*n, dx)
    pred_e = pred_graph.e.reshape(-1, pred_graph.e.shape[-1])  # (bs*n*n, de)

    # Remove masked rows
    mask_x = (true_x != 0).any(axis=-1)
    mask_e = (true_e != 0).any(axis=-1)

    loss_x = np.where(mask_x, loss_fn(pred_x, true_x).mean(-1), 0).sum() / mask_x.sum()
    loss_e = np.where(mask_e, loss_fn(pred_e, true_e).mean(-1), 0).sum() / mask_e.sum()

    return loss_x + lambda_train_e * loss_e  # + lambda_train_y * loss_y


@typed
def val_loss(
    *,
    T: SInt,
    target: GraphDistribution,
    nodes_dist: Float[Array, "n"],
    transition_model: TransitionModel,
    state: TrainState,
    get_probability: Callable[[GraphDistribution], GraphDistribution],
    rng_key: Key,
):
    # TODO: this still has to be fully ported
    x = target.x
    e = target.e
    y = target.y
    mask = target.mask

    # 1.
    n = mask.sum(1).astype(int)
    log_pN = np.log(nodes_dist[n] + 1e-30)
    # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
    kl_prior = df.kl_prior(
        target=target,
        diffusion_steps=T,
        transition_model=transition_model,
    )
    # 3. Diffusion loss
    loss_all_t = df.compute_lt(
        rng=rng_key,
        g=target,
        n_t_samples=1,
        n_g_samples=1,
        diffusion_steps=T,
        transition_model=transition_model,
        get_probability=get_probability,
    )
    # 4. Reconstruction loss
    # TODO
    # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
    prob0 = df.reconstruction_logp(
        rng_key=rng_key, state=state, graph=target, t=t, noise_schedule=noise_schedule
    )

    loss_term_0 = np.sum(x * prob0.X.log(), axis=1) + np.sum(e * prob0.E.log(), axis=1)

    # Combine terms
    nlls = -log_pN + kl_prior + loss_all_t - loss_term_0
    assert len(nlls.shape) == 1, f"{nlls.shape} has more than only batch dim."

    nll = np.mean(nlls, 1)

    print(
        {
            "kl prior": kl_prior.mean(),
            "Estimator loss terms": loss_all_t.mean(),
            "log_pn": log_pN.mean(),
            "loss_term_0": loss_term_0,
        },
    )
    return nll


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
    print(f"{max_value=} {min_value=}")
    print(f"there are nans: {there_are_nan}")


def train_step(
    *,
    dense_data: GraphDistribution,
    i: SInt,
    state: TrainState,
    rng: Key,
    diffusion_steps: SInt,
    transition_model: TransitionModel,
) -> tuple[TrainState, SFloat]:
    dropout_train_key = jax.random.fold_in(key=rng, data=state.step)
    # if data.edge_index.size == 0:
    #     print("Found a batch with no edges. Skipping.")
    #     return state, 0.0

    z_t = df.apply_random_noise(
        graph=dense_data,
        test=np.array(False),
        rng=rng,
        diffusion_steps=diffusion_steps,
        transition_model=transition_model,
    ).set("y", np.ones((dense_data.y.shape[0], 1)))

    # concatenates the extra features to the graph
    # FIXME: I disabled temporarely the extra features
    # graph_dist_with_extra_data = z_t  | extra_features.compute(z_t)
    def loss_fn(params: FrozenDict):
        raw_pred = state.apply_fn(  # TODO: suffix dist for distributions
            params,
            z_t.x,
            z_t.e,
            z_t.y,
            z_t.mask.astype(float),
            rngs={"dropout": dropout_train_key},
        )
        pred = GraphDistribution.masked(
            x=raw_pred.x,
            e=raw_pred.e,
            y=raw_pred.y,
            mask=z_t.mask,
        )  # consider changing the name of this class to something that suggests ditribution

        loss = train_loss(pred_graph=pred, true_graph=dense_data)
        jprint("loss {loss}", loss=loss)
        return loss, pred

    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = gradient_fn(state.params)
    grads = jax.tree_map(
        lambda x: np.where(np.isnan(x), 0.0, x), grads
    )  # remove the nans :(
    state = state.apply_gradients(grads=grads)
    return state, loss


@typed
def train_model(
    *,
    model: nn.Module,
    train_loader,
    val_loader,
    params: FrozenDict,
    num_epochs=1,
    rngs: dict[str, Key],
    lr: float,
    config: TrainingConfig,
    output_dims: dict[str, int],
    nodes_dist_torch,
):
    rng = rngs["params"]
    nodes_dist = np.array(nodes_dist_torch.prob)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(lr),
        key=rngs["dropout"],
    )
    """
    The following is the prior of the data. It is used to generate matrices Q_t, 
    and also to compute the prior loss term in the ELBO.
    """
    # Key idea proposed in the DiGress paper: use the prior distribution to gnerate Q_t
    # These priors are extracted from the dataset.
    transition_model = TransitionModel.create(
        x_priors=np.array(
            [
                0.7230000495910645,
                0.11510000377893448,
                0.15930001437664032,
                0.0026000002399086952,
            ]
        ),
        e_priors=np.array(
            [
                0.7261000275611877,
                0.23839999735355377,
                0.027400000020861626,
                0.008100000210106373,
                0.0,
            ]
        ),
        y_classes=output_dims["y"],
        diffusion_steps=config.diffusion_steps,
    )
    # Track best eval accuracy
    for epoch_idx in range(1, num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        # state, _ = train_epoch(
        #     rng=rng,
        #     state=state,
        #     train_loader=train_loader,
        #     epoch=epoch_idx,
        #     diffusion_steps=np.array(config.diffusion_steps),
        #     transition_model=transition_model,
        # )
        val_loss = val_epoch(
            rng=rng,
            state=state,
            val_loader=val_loader,
            config=config,
            transition_model=transition_model,
            nodes_dist=nodes_dist,
        )


def get_probability_from_state(
    state: TrainState, droput_rng: Key
) -> Callable[[GraphDistribution], GraphDistribution]:
    def get_probability(g: GraphDistribution) -> GraphDistribution:
        y = np.ones((g.y.shape[0], 1)) if g.y.size == 0 else g.y
        raw_pred = state.apply_fn(
            state.params,
            g.x,
            g.e,
            y,
            g.mask.astype(float),
            rngs={"dropout": droput_rng},
        )
        pred = GraphDistribution.masked(
            x=jax.nn.softmax(raw_pred.x, -1),
            e=jax.nn.softmax(raw_pred.e, -1),
            y=jax.nn.softmax(raw_pred.y, -1),
            mask=g.mask,
        )
        return pred

    return get_probability


def val_step(
    *,
    dense_data: GraphDistribution,
    state: TrainState,
    diffusion_steps: SInt,
    rng: Key,
    transition_model: TransitionModel,
    nodes_dist: Float[Array, "n"],
) -> SFloat:
    dropout_test_key = jax.random.fold_in(key=rng, data=state.step)
    # X, E, node_mask = to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
    X, E = dense_data.x, dense_data.e
    z_t = df.apply_random_noise(
        graph=dense_data,
        test=False,
        rng=rng,
        diffusion_steps=diffusion_steps,
        transition_model=transition_model,
    ).set("y", np.ones((dense_data.y.shape[0], 1)))
    # skip extra features for now
    graph_dist_with_extra_data = z_t  #  | extra_features.compute(z_t)

    def loss_fn(params: FrozenDict):
        get_probability = get_probability_from_state(state, dropout_test_key)
        loss = train_loss(
            pred_graph=get_probability(z_t),
            true_graph=dense_data,
        )
        loss = val_loss(
            state=state,
            target=dense_data,
            T=diffusion_steps,
            transition_model=transition_model,
            rng_key=rng,
            nodes_dist=nodes_dist,
            get_probability=get_probability,
        )
        return loss, pred

    loss, _ = loss_fn(state.params)
    print(f"{loss=}")
    return loss


@typed
def val_epoch(
    *,
    val_loader,
    state: TrainState,
    config: TrainingConfig,
    rng: Key,
    transition_model: TransitionModel,
    nodes_dist: Float[Array, "n"],
):
    run_loss = 0.0
    for batch in tqdm(val_loader):
        loss = val_step(
            dense_data=GraphDistribution.from_sparse_torch(batch),
            state=state,
            diffusion_steps=int(config.diffusion_steps),
            rng=rng,
            transition_model=transition_model,
            nodes_dist=nodes_dist,
        )
        run_loss += loss
    avg_loss = run_loss / len(val_loader)
    # prints the average loss for the epoch
    print(f"Validation loss: {avg_loss:.4f}")
    return avg_loss


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
    run_loss = 0.0
    for batch_index, batch in enumerate(train_loader):
        dense_data = GraphDistribution.from_sparse_torch(batch)
        state, loss = train_step(
            dense_data=dense_data,
            i=np.array(epoch),
            state=state,
            diffusion_steps=diffusion_steps,
            rng=rng,
            transition_model=transition_model,
        )
        run_loss += loss
        # if batch_index == 3:
        #     break
    avg_loss = run_loss / len(train_loader)
    # prints the average loss for the epoch
    print(f"Epoch {epoch} loss: {avg_loss:.4f}")
    return state, avg_loss
