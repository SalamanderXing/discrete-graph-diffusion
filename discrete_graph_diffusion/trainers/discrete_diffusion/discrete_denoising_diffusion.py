from jax import numpy as np
from flax.training.train_state import TrainState
from jax import jit
import flax.linen as nn
from jax import Array
from jax.random import PRNGKeyArray
import jax
import optax
import ipdb
from rich import print
from typing import Iterable
from dataclasses import dataclass
from typing import FrozenSet
from flax.core.frozen_dict import FrozenDict

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
from .diffusion import apply_noise, compute_Lt, kl_prior
from .config import TrainingConfig 
from .utils import Graph
from .utils.geometric import to_dense
from .sample import sample_discrete_features
from .diffusion import kl_prior
from .nodes_distribution import NodesDistribution


@dataclass(frozen=True)
class DataBatch:
    edge_index: Array
    edge_attr: Array
    x: Array
    y: Array
    batch: Array


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


def training_step(*, model: nn.Module, data: DataBatch, i: int, state: TrainState):
    if data.edge_index.size == 0:
        print("Found a batch with no edges. Skipping.")
        return
    dense_data, node_mask = to_dense(
        data.x, data.edge_index, data.edge_attr, data.batch
    )
    dense_data = dense_data.mask(node_mask)
    X, E = dense_data.X, dense_data.E
    if i == 0:
        ipdb.set_trace()
    noisy_data = apply_noise(X, E, data.y, node_mask)
    extra_data = compute_extra_data(noisy_data)
    pred = model(noisy_data, extra_data, node_mask)
    loss = train_loss(
        masked_pred_X=pred.X,
        masked_pred_E=pred.E,
        pred_y=pred.y,
        true_X=X,
        true_E=E,
        true_y=data.y,
    )

    train_metrics(
        masked_pred_X=pred.X,
        masked_pred_E=pred.E,
        true_X=X,
        true_E=E,
        log=i % self.log_every_steps == 0,
    )

    return {"loss": loss}


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


def train_model(
    *,
    model: nn.Module,
    train_loader,
    params: FrozenDict,
    val_loader,
    num_epochs=1,
    rngs: dict[str, PRNGKeyArray],
    lr: float,
    config:TrainingConfig
):
    optimizer = optax.adam(lr)
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    # aaaaaaaaaa Train model for defined number of epochs
    # We first need to create optimizer and the scheduler for the given number of epochs
    logger = Logger()
    # Track best eval accuracy
    for epoch_idx in tqdm(range(1, num_epochs + 1)):
        rng, input_rng = jax.random.split(rng)
        state, train_loss = training_epoch(
            state=state, train_dataloader=train_dataloader, batch_size=batch_size, input_rng=input_rng
        )
        _, test_loss = apply_model(
            state, test_dataloader
        )



def pytorch_geometric_databatch_to_jax(pytorch_geometric_databatch):
    ipdb.set_trace()
    return


def training_epoch(model: nn.Module, train_loader, epoch: int, state: TrainState):
    for batch in train_loader:
        batch = pytorch_geometric_databatch_to_jax(pytorch_geometric_databatch_to_jax)
        training_step(model=model, data=batch, i=epoch, state=state)


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
