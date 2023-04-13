from jax import numpy as np
from flax.training import train_state
from jax import jit
from jax import random
import flax.linen as nn
from jax import Array
from jax.random import PRNGKeyArray
import jax
import optax
import ipdb
from rich import print
from torch import dropout_
from tqdm import tqdm
from typing import Iterable
from dataclasses import dataclass
from typing import FrozenSet
from flax.core.frozen_dict import FrozenDict
from .utils import NoisyData
from .train_loss import train_loss

# from .train_loss import TrainLossDiscrete
# from .metrics import AverageMetric
# from .metrics.sum_except_batch import SumExceptBatchMetric
# from .metrics.sum_except_batch_kl import SumExceptBatchKL
from .noise_schedule import PredefinedNoiseScheduleDiscrete
from .transition_model import (
    DiscreteUniformTransition,
    MarginalUniformTransition,
    TransitionModel,
)
from .diffusion import apply_noise, compute_Lt, kl_prior, compute_extra_data
from .config import TrainingConfig
from .utils import Graph
from .utils.geometric import to_dense
from .sample import sample_discrete_features
from .diffusion import kl_prior
from .nodes_distribution import NodesDistribution
from .extra_features import extra_features


@dataclass(frozen=True)
class DataBatch:
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


class Logger:
    def add_scalar(self, name: str, val: float | int, epoch: int):
        print(f"{name} {val} {epoch}")


def compute_val_loss(
    *,
    target: Graph,
    pred: Graph,
    noisy_data: Array,
    node_mask: Array,
    node_dist: NodesDistribution,
    test=False,
    state: FrozenDict,
):
    """Computes an estimator for the variational lower bound.
    pred: (batch_size, n, total_features)
    noisy_data: dict
    X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
    node_mask : (bs, n)
    Output: nll (size 1)
    """
    t = noisy_data["t"]

    # 1.
    N = node_mask.sum(1).astype(int)
    log_pN = node_dist.log_prob(N)

    # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
    kl_prior = kl_prior(X, E, node_mask)

    # 3. Diffusion loss
    loss_all_t = compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

    # 4. Reconstruction loss
    # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
    prob0 = reconstruction_logp(t, X, E, node_mask)

    loss_term_0 = val_X_logp(X * prob0.X.log()) + val_E_logp(E * prob0.E.log())

    # Combine terms
    nlls = -log_pN + kl_prior + loss_all_t - loss_term_0
    assert len(nlls.shape) == 1, f"{nlls.shape} has more than only batch dim."

    # Update NLL metric object and return batch nll
    nll = (test_nll if test else val_nll)(nlls)  # Average over the batch

    print(
        {
            "kl prior": kl_prior.mean(),
            "Estimator loss terms": loss_all_t.mean(),
            "log_pn": log_pN.mean(),
            "loss_term_0": loss_term_0,
            "batch_test_nll" if test else "val_nll": nll,
        },
    )
    return nll


def forward(
    params: FrozenDict,
    nX: Array,
    nE: Array,
    y: Array,
    node_mask: Array,
    state: TrainState,
    extra_data: Graph,
    dropout_key: PRNGKeyArray,
):
    X = np.concatenate((nX, extra_data.x), axis=2).astype(float)
    E = np.concatenate((nE, extra_data.e), axis=3).astype(float)
    y = np.hstack((y, extra_data.y)).astype(float)
    # print(f"{noisy_data.graph=}")
    # print(f"{extra_data=}")
    # print(
    #    f"Forward shapes: {X.shape=}, {E.shape=}, {y.shape=} {noisy_data.graph.mask.shape=}"
    # )
    # __import__('sys').exit()
    return state.apply_fn(
        params,
        X,
        E,
        y,
        node_mask,
        # training=True,
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
        return
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

    # extra data shapes:  torch.Size([200, 9, 8]) torch.Size([200, 9, 9, 0]) torch.Size([200, 13])
    # assert extra_data.x.shape[2] == 8, f"{extra_data.x.shape=}"
    # assert extra_data.e.shape[3] == 0, f"{extra_data.e.shape=}"
    # assert extra_data.y.shape[1] == 13, f"{extra_data.y.shape=}"
    # extra_data = Graph(x=extra_data.x[:, :, :8], e=extra_data.e, y=extra_data.y[:, :13])
    # print(f"{extra_data=}")
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

    # loss, pred = loss_fn(state)
    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, pred), grads = gradient_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


'''
@jit
def apply_model(state: TrainState, images, labels):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, images)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = np.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = np.mean(np.argmax(logits, -1) == labels)
    return grads, loss, accuracy
'''


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
    # We first need to create optimizer and the scheduler for the given number of epochs
    logger = Logger()
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
        # _, test_loss = apply_model(state, val_loader)


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
    # prints the average loss for the epoch
    print(f"Epoch {epoch} loss: {run_loss / len(train_loader)}")


def validation_step(model: nn.Module, data: Array, i: int):
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


"""
class DiscreteDenoisingDiffusion:
    def __init__(
        self,
        model: nn.Module,
        cfg: GeneralConfig,
        sampling_metrics: AverageMetric,
    ):
        self.model = model
        self.cfg = cfg
        self.train_loss = TrainLossDiscrete(self.cfg.train.lambda_train)

        self.val_nll = AverageMetric()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()

        self.test_nll = AverageMetric()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            cfg.train.diffusion_noise_schedule, timesteps=cfg.train.diffusion_steps
        )
        self.sampling_metrics = sampling_metrics
        if cfg.train.transition == "uniform":
            self.transition_model = DiscreteUniformTransition(
                x_classes=self.cfg.dataset.out_dims.X,
                e_classes=self.cfg.dataset.out_dims.E,
                y_classes=self.cfg.dataset.out_dims.y,
            )
            x_limit = np.ones(self.cfg.dataset.out_dims.X) / self.cfg.dataset.out_dims.X
            e_limit = np.ones(self.cfg.dataset.out_dims.E) / self.cfg.dataset.out_dims.E
            y_limit = np.ones(self.cfg.dataset.out_dims.y) / self.cfg.dataset.out_dims.y
            self.limit_dist = Graph(x=x_limit, e=e_limit, y=y_limit)

        elif cfg.train.transition == "marginal":
            node_types = self.cfg.dataset.node_types.astype(float)
            x_marginals = node_types / np.sum(node_types)

            edge_types = self.cfg.dataset.edge_types.astype(float)
            e_marginals = edge_types / np.sum(edge_types)
            print(
                f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges"
            )
            self.transition_model = MarginalUniformTransition(
                x_marginals=x_marginals,
                e_marginals=e_marginals,
                y_classes=self.cfg.dataset.out_dims.y,
            )
            self.limit_dist = Graph(
                x=x_marginals,
                e=e_marginals,
                y=np.ones(self.cfg.dataset.out_dims.y) / self.cfg.dataset.out_dims.y,
            )

        # self.save_hyperparameters(ignore=[train_metrics, sampling_metrics]) # TODO: implement this maybe?
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.train.log_every_steps
        self.number_chain_steps = cfg.train.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.sampling_metrics.reset()
"""
