"""
Entrypoint of the discrete denoising diffusion model.
The function `train_model` is the main entrypoint.
"""
from jax import numpy as np
from jax.numpy import array
from flax.training import train_state
from jax import jit
import flax.linen as nn
from jaxtyping import Float, Int, Bool
import jax
import optax
import ipdb
from rich import print
from tqdm import tqdm
from dataclasses import dataclass
from flax.core.frozen_dict import FrozenDict
from mate.jax import SFloat, SInt, SBool, SUInt, typed, Key
from .noise_schedule import PredefinedNoiseScheduleDiscrete
from .transition_model import (
    DiscreteUniformTransition,
    MarginalUniformTransition,
    TransitionModel,
)
from . import diffusion_functions as df
from .config import TrainingConfig
from .diffusion_types import EmbeddedGraph, NoisyData, Distribution
from .nodes_distribution import NodesDistribution
from .extra_features import extra_features
from .noise_schedule import PredefinedNoiseScheduleDiscrete


class TrainState(train_state.TrainState):
    key: jax.random.KeyArray


@typed
def train_loss(
    pred_graph: EmbeddedGraph,
    true_graph: EmbeddedGraph,
    lambda_train_x: SFloat = array(1.0),
    lambda_train_e: SFloat = array(5.0),
    lambda_train_y: SFloat = array(0.0),
):
    """Compute train metrics
    masked_pred_X : tensor -- (bs, n, dx)
    masked_pred_E : tensor -- (bs, n, n, de)
    pred_y : tensor -- (bs, )
    true_X : tensor -- (bs, n, dx)
    true_E : tensor -- (bs, n, n, de)
    true_y : tensor -- (bs, )
    log : boolean."""
    loss_fn = optax.softmax_cross_entropy  # (logits=x, labels=y).sum()
    true_x = true_graph.x.reshape(-1, true_graph.x.shape[-1])  # (bs*n, dx)
    true_e = true_graph.e.reshape(-1, true_graph.e.shape[-1])  # (bs*n*n, de)
    masked_pred_x = pred_graph.x.reshape(-1, pred_graph.x.shape[-1])  # (bs*n, dx)
    masked_pred_e = pred_graph.e.reshape(-1, pred_graph.e.shape[-1])  # (bs*n*n, de)

    # Remove masked rows
    mask_x = (true_x != 0).any(axis=-1)
    mask_e = (true_e != 0).any(axis=-1)

    flat_true_x = true_x[mask_x]
    flat_true_e = true_e[mask_e]

    flat_pred_x = masked_pred_x[mask_x]
    flat_pred_e = masked_pred_e[mask_e]

    # Compute metrics
    loss_x = loss_fn(flat_pred_x, flat_true_x).mean() if true_x.size else 0
    loss_e = loss_fn(flat_pred_e, flat_true_e).mean() if true_e.size else 0
    loss_y = loss_fn(pred_graph.y, true_graph.y).mean()
    return lambda_train_x * loss_x + lambda_train_e * loss_e + lambda_train_y * loss_y


@typed
def val_loss(
    *,
    T: SInt,
    target: EmbeddedGraph,
    pred: EmbeddedGraph,
    nodes_dist: NodesDistribution,
    noise_schedule: PredefinedNoiseScheduleDiscrete,
    transition_model: TransitionModel,
    state: TrainState,
    limit_dist: Distribution,
    rng_key: Key,
):
    # TODO: this still has to be fully ported
    x = target.x
    e = target.e
    y = target.y
    mask = target.mask

    # 1.
    N = mask.sum(1).astype(int)
    log_pN = nodes_dist.log_prob(N)
    # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
    kl_prior = df.kl_prior(
        target=target,
        T=T,
        noise_schedule=noise_schedule,
        transition_model=transition_model,
        limit_dist=limit_dist,
    )

    # 3. Diffusion loss
    loss_all_t = df.compute_Lt(
        target=target,
        pred=pred,
        noisy_data=noisy_data,
        T=T,
        transition_model=transition_model,
    )

    # 4. Reconstruction loss
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


@typed
def forward(
    params: FrozenDict,
    graph: EmbeddedGraph,
    state: TrainState,
    extra_data: EmbeddedGraph,
    dropout_key: Key,
) -> EmbeddedGraph:
    X = np.concatenate((graph.x, extra_data.x), axis=2)
    E = np.concatenate((graph.e, extra_data.e), axis=3)
    y = np.hstack((graph.y, extra_data.y))
    pred = state.apply_fn(
        params,
        X,
        E,
        y,
        graph.mask.astype(float),
        rngs={"dropout": dropout_key},
    )
    return EmbeddedGraph(x=pred.x, e=pred.e, y=pred.y, mask=graph.mask)


@typed
def training_step(
    *,
    data,
    i: SInt,
    state: TrainState,
    rng: Key,
    config: TrainingConfig,
    noise_schedule: PredefinedNoiseScheduleDiscrete,
    transition_model: TransitionModel,
) -> tuple[TrainState, float]:
    dropout_train_key = jax.random.fold_in(key=rng, data=state.step)
    if data.edge_index.size == 0:
        print("Found a batch with no edges. Skipping.")
        return state, 0.0
    # X, E, node_mask = to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
    # dense_data = EmbeddedGraph(x=X, e=E, y=data.y, mask=node_mask)
    dense_data = EmbeddedGraph.from_sparse_torch(data)
    X, E = dense_data.x, dense_data.e
    if i == 0:
        ipdb.set_trace()

    noisy_data = df.apply_random_noise(
        graph=dense_data,
        training=np.array(True),
        rng=rng,
        T=array(config.diffusion_steps),
        noise_schedule=noise_schedule,
        transition_model=transition_model,
    )
    # prints the shapes of the noisy data's X, E, y
    extra_data = extra_features(noisy_data.graph)

    def loss_fn(params: FrozenDict):
        pred = forward(
            params, noisy_data.graph, state, extra_data, dropout_key=dropout_train_key
        )
        loss = train_loss(pred_graph=pred, true_graph=dense_data)
        return loss, pred

    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = gradient_fn(state.params)
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
    nodes_dist,
):
    rng = rngs["params"]
    nodes_dist = NodesDistribution.from_torch(nodes_dist, rng)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(lr),
        key=rngs["dropout"],
    )
    noise_schedule = PredefinedNoiseScheduleDiscrete(
        "cosine", diffusion_steps=config.diffusion_steps
    )
    """
    transition_model = DiscreteUniformTransition(
        x_classes=output_dims["X"],
        e_classes=output_dims["E"],
        y_classes=output_dims["y"],
    )
    """
    # TODO: replace hardcoded values with computed ones
    limit_dist = Distribution(
        x=np.array(
            [
                0.7230000495910645,
                0.11510000377893448,
                0.15930001437664032,
                0.0026000002399086952,
            ]
        ),
        e=np.array(
            [
                0.7261000275611877,
                0.23839999735355377,
                0.027400000020861626,
                0.008100000210106373,
                0.0,
            ]
        ),
        y=np.ones(output_dims["y"]) / output_dims["y"],
    )
    transition_model = MarginalUniformTransition(
        x_marginals=limit_dist.x,
        e_marginals=limit_dist.e,
        y_classes=output_dims["y"],
    )
    # Track best eval accuracy
    for epoch_idx in range(1, num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        # state, train_loss = training_epoch(
        #     rng=rng,
        #     state=state,
        #     train_loader=train_loader,
        #     epoch=epoch_idx,
        #     config=config,
        #     transition_model=transition_model,
        #     noise_schedule=noise_schedule,
        # )
        val_loss = val_epoch(
            rng=rng,
            state=state,
            val_loader=val_loader,
            config=config,
            transition_model=transition_model,
            noise_schedule=noise_schedule,
            limit_dist=limit_dist,
            nodes_dist=nodes_dist,
        )


@typed
def val_step(
    *,
    data,
    state: TrainState,
    config: TrainingConfig,
    rng: Key,
    noise_schedule: PredefinedNoiseScheduleDiscrete,
    transition_model: TransitionModel,
    limit_dist: Distribution,
    nodes_dist: NodesDistribution,
) -> SFloat:
    dropout_test_key = jax.random.fold_in(key=rng, data=state.step)
    if data.edge_index.size == 0:
        print("Found a batch with no edges. Skipping.")
        return array(0.0)
    # X, E, node_mask = to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
    # dense_data = EmbeddedGraph(x=X, e=E, y=data.y, mask=node_mask)
    dense_data = EmbeddedGraph.from_sparse_torch(data)
    X, E = dense_data.x, dense_data.e
    noisy_data = df.apply_random_noise(
        graph=dense_data,
        training=np.array(False),
        rng=rng,
        T=np.array(config.diffusion_steps),
        noise_schedule=noise_schedule,
        transition_model=transition_model,
    )
    extra_data = extra_features(noisy_data.graph)

    def loss_fn(params: FrozenDict):
        pred = forward(
            params, noisy_data.graph, state, extra_data, dropout_key=dropout_test_key
        )
        loss = train_loss(pred_graph=pred, true_graph=dense_data)  # just for testing
        # loss = val_loss(
        #     state=state,
        #     target=dense_data,
        #     pred=pred,
        #     T=array(config.diffusion_steps),
        #     noise_schedule=noise_schedule,
        #     transition_model=transition_model,
        #     rng_key=rng,
        #     limit_dist=limit_dist,
        #     nodes_dist=nodes_dist,
        # )
        return loss, pred

    loss, _ = loss_fn(state.params)
    return loss


@typed
def val_epoch(
    *,
    val_loader,
    state: TrainState,
    config: TrainingConfig,
    rng: Key,
    noise_schedule: PredefinedNoiseScheduleDiscrete,
    transition_model: TransitionModel,
    limit_dist: Distribution,
    nodes_dist: NodesDistribution,
):
    run_loss = 0.0
    for batch in tqdm(val_loader):
        loss = val_step(
            data=batch,
            state=state,
            config=config,
            rng=rng,
            noise_schedule=noise_schedule,
            transition_model=transition_model,
            limit_dist=limit_dist,
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
    config: TrainingConfig,
    rng: Key,
    noise_schedule: PredefinedNoiseScheduleDiscrete,
    transition_model: TransitionModel,
):
    run_loss = 0.0
    for batch in tqdm(train_loader):
        state, loss = training_step(
            data=batch,
            i=np.array(epoch),
            state=state,
            config=config,
            rng=rng,
            noise_schedule=noise_schedule,
            transition_model=transition_model,
        )
        run_loss += loss
    avg_loss = run_loss / len(train_loader)
    # prints the average loss for the epoch
    print(f"Epoch {epoch} loss: {avg_loss:.4f}")
    return state, avg_loss
