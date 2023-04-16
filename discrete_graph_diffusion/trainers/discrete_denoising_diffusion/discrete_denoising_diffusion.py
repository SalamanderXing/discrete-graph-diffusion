"""
Entrypoint of the discrete denoising diffusion model.
The function `train_model` is the main entrypoint.
"""
from jax import numpy as np
from flax.training import train_state
from jax import jit
import flax.linen as nn
from jax import Array
from jax.random import PRNGKeyArray
import jax
import optax
import ipdb
from rich import print
from tqdm import tqdm
from dataclasses import dataclass
from flax.core.frozen_dict import FrozenDict

from .noise_schedule import PredefinedNoiseScheduleDiscrete
from .transition_model import (
    DiscreteUniformTransition,
    TransitionModel,
)
from .diffusion_functions import apply_noise, compute_Lt
from .config import TrainingConfig
from .diffusion_types import Graph, NoisyData
from .utils.geometric import to_dense
from .diffusion_functions import kl_prior as kl_prior_fn, reconstruction_logp
from .nodes_distribution import NodesDistribution
from .extra_features import extra_features
from .noise_schedule import PredefinedNoiseScheduleDiscrete


@dataclass(frozen=True)
class DataBatch:
    """
    Data structure mimicking the PyTorch Geometric DataBatch.
    """

    edge_index: Array
    edge_attr: Array
    x: Array
    y: Array
    batch: Array

    @classmethod
    def from_pytorch(cls, data):
        return cls(
            edge_index=np.array(data.edge_index),
            edge_attr=np.array(data.edge_attr),
            x=np.array(data.x),
            y=np.array(data.y),
            batch=np.array(data.batch),
        )


class TrainState(train_state.TrainState):
    key: jax.random.KeyArray


def train_loss(
    masked_pred_x: Array,
    masked_pred_e: Array,
    pred_y: Array,
    true_x: Array,
    true_e: Array,
    true_y: Array,
    lambda_train_x: float = 1,
    lambda_train_e: float = 5,
    lambda_train_y: float = 0,
):
    """Compute train metrics
    masked_pred_X : tensor -- (bs, n, dx)
    masked_pred_E : tensor -- (bs, n, n, de)
    pred_y : tensor -- (bs, )
    true_X : tensor -- (bs, n, dx)
    true_E : tensor -- (bs, n, n, de)
    true_y : tensor -- (bs, )
    log : boolean."""

    loss_fn = lambda x, y: optax.softmax_cross_entropy(logits=x, labels=y).sum()
    true_x = true_x.reshape(-1, true_x.shape[-1])  # (bs*n, dx)
    true_e = true_e.reshape(-1, true_e.shape[-1])  # (bs*n*n, de)
    masked_pred_x = masked_pred_x.reshape(-1, masked_pred_x.shape[-1])  # (bs*n, dx)
    masked_pred_e = masked_pred_e.reshape(-1, masked_pred_e.shape[-1])  # (bs*n*n, de)

    # Remove masked rows
    mask_x = (true_x != 0).any(axis=-1)
    mask_e = (true_e != 0).any(axis=-1)

    flat_true_x = true_x[mask_x]
    flat_true_e = true_e[mask_e]

    flat_pred_x = masked_pred_x[mask_x]
    flat_pred_e = masked_pred_e[mask_e]

    # Compute metrics
    loss_x = loss_fn(flat_pred_x, flat_true_x) if true_x.size else 0
    loss_e = loss_fn(flat_pred_e, flat_true_e) if true_e.size else 0
    loss_y = loss_fn(pred_y, true_y)

    return lambda_train_x * loss_x + lambda_train_e * loss_e + lambda_train_y * loss_y


def compute_val_loss(
    *,
    T: int,
    target: Graph,
    pred: Graph,
    noisy_data: NoisyData,
    node_dist: NodesDistribution,
    noise_schedule: PredefinedNoiseScheduleDiscrete,
    transition_model: TransitionModel,
    state: FrozenDict,
    limit_dist: Graph,
    rng_key: PRNGKeyArray,
):
    # TODO: this still has to be fully ported
    t = noisy_data.t
    x = target.x
    e = target.e
    y = target.y
    mask = target.mask

    # 1.
    N = mask.sum(1).astype(int)
    log_pN = node_dist.log_prob(N)
    # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
    kl_prior = kl_prior_fn(
        x=x,
        e=e,
        mask=mask,
        T=T,
        noise_schedule=noise_schedule,
        transition_model=transition_model,
        limit_dist=limit_dist,
    )

    # 3. Diffusion loss
    loss_all_t = compute_Lt(
        target=target,
        pred=pred,
        noisy_data=noisy_data,
        T=T,
        transition_model=transition_model,
    )

    # 4. Reconstruction loss
    # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
    prob0 = reconstruction_logp(
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


def forward(
    params: FrozenDict,
    graph_x: Array,
    graph_e: Array,
    graph_y: Array,
    node_mask: Array,
    state: TrainState,
    extra_data: Graph,
    dropout_key: PRNGKeyArray,
):
    X = np.concatenate((graph_x, extra_data.x), axis=2)
    E = np.concatenate((graph_e, extra_data.e), axis=3)
    y = np.hstack((graph_y, extra_data.y))
    return state.apply_fn(
        params,
        X,
        E,
        y,
        node_mask,
        rngs={"dropout": dropout_key},
    )


def training_step(
    *,
    data: DataBatch,
    i: int,
    state: TrainState,
    rng: PRNGKeyArray,
    config: TrainingConfig,
    noise_schedule: PredefinedNoiseScheduleDiscrete,
    transition_model: TransitionModel,
) -> tuple[TrainState, float]:
    dropout_train_key = jax.random.fold_in(key=rng, data=state.step)
    if data.edge_index.size == 0:
        print("Found a batch with no edges. Skipping.")
        return state, 0.0
    X, E, node_mask = to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
    dense_data = Graph(x=X, e=E, y=data.y, mask=node_mask)
    X, E, y = dense_data.x, dense_data.e, dense_data.y
    if i == 0:
        ipdb.set_trace()

    noisy_data = apply_noise(
        x=X,
        e=E,
        y=y,
        training=True,
        node_mask=node_mask,
        rng=rng,
        T=config.diffusion_steps,
        noise_schedule=noise_schedule,
        transition_model=transition_model,
    )
    # prints the shapes of the noisy data's X, E, y
    extra_data = extra_features(noisy_data)
    nX, nE, ny, n_mask = (
        noisy_data.graph.x,
        noisy_data.graph.e,
        noisy_data.graph.y,
        noisy_data.graph.mask,
    )
    n_mask = n_mask.astype(float)

    def loss_fn(params: FrozenDict):
        pred = forward(
            params, nX, nE, ny, n_mask, state, extra_data, dropout_key=dropout_train_key
        )
        loss = train_loss(
            masked_pred_x=pred.x,
            masked_pred_e=pred.e,
            pred_y=pred.y,
            true_x=X,
            true_e=E,
            true_y=data.y,
        )
        return loss, pred

    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = gradient_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def train_model(
    *,
    model: nn.Module,
    train_loader,
    params: FrozenDict,
    val_loader,
    num_epochs=1,
    rngs: dict[str, PRNGKeyArray],
    lr: float,
    config: TrainingConfig,
    output_dims: dict[str, int],
):
    optimizer = optax.adam(lr)
    rng = rngs["params"]
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        key=rngs["dropout"],
    )
    noise_schedule = PredefinedNoiseScheduleDiscrete(
        "cosine", diffusion_steps=config.diffusion_steps
    )
    transition_model = DiscreteUniformTransition(
        x_classes=output_dims["X"],
        e_classes=output_dims["E"],
        y_classes=output_dims["y"],
    )
    # Track best eval accuracy
    for epoch_idx in range(1, num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        state, train_loss = training_epoch(
            rng=rng,
            state=state,
            train_loader=train_loader,
            epoch=epoch_idx,
            config=config,
            transition_model=transition_model,
            noise_schedule=noise_schedule,
        )


def training_epoch(
    *,
    train_loader,
    epoch: int,
    state: TrainState,
    config: TrainingConfig,
    rng: PRNGKeyArray,
    noise_schedule: PredefinedNoiseScheduleDiscrete,
    transition_model: TransitionModel,
):
    run_loss = 0.0
    for batch in tqdm(train_loader):
        batch = DataBatch.from_pytorch(batch)
        state, loss = training_step(
            data=batch,
            i=epoch,
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


def validation_step(model: nn.Module, data: Array, i: int):
    # TODO: this is still not ported.
    dense_data, node_mask = utils.to_dense(
        data.x, data.edge_index, data.edge_attr, data.batch
    )
    dense_data = dense_data.mask(node_mask)
    noisy_data = apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
    extra_data = compute_extra_data(noisy_data)
    pred = model(noisy_data, extra_data, node_mask)
    nll = compute_val_loss(
        pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, test=False
    )
    return {"loss": nll}
