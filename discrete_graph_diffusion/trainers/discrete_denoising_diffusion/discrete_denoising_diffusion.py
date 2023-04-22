"""
Entrypoint of the discrete denoising diffusion model.
The function `train_model` is the main entrypoint.
"""
from jax import numpy as np
from jax.numpy import array
from flax.training import train_state
import flax.linen as nn
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
    Distribution,
    TransitionModel,
    TransitionModel,
)
from .nodes_distribution import NodesDistribution
from .extra_features import extra_features


class TrainState(train_state.TrainState):
    key: jax.random.KeyArray


@typed
def train_loss(
    pred_graph: GraphDistribution,
    true_graph: GraphDistribution,
    lambda_train_e: SFloat = array(5.0),
    lambda_train_y: SFloat = array(0.0),
):
    loss_fn = optax.softmax_cross_entropy  # (logits=x, labels=y).sum()
    true_x = true_graph.x.reshape(-1, true_graph.x.shape[-1])  # (bs*n, dx)
    true_e = true_graph.e.reshape(-1, true_graph.e.shape[-1])  # (bs*n*n, de)
    pred_x = pred_graph.x.reshape(-1, pred_graph.x.shape[-1])  # (bs*n, dx)
    pred_e = pred_graph.e.reshape(-1, pred_graph.e.shape[-1])  # (bs*n*n, de)

    # Remove masked rows
    mask_x = (true_x != 0).any(axis=-1)
    mask_e = (true_e != 0).any(axis=-1)

    """
    flat_true_x = true_x[mask_x]
    flat_true_e = true_e[mask_e]

    flat_pred_x = masked_pred_x[mask_x]
    flat_pred_e = masked_pred_e[mask_e]
    """

    loss_x = np.where(mask_x, loss_fn(pred_x, true_x), 0).sum() / mask_x.sum()
    loss_e = np.where(mask_e, loss_fn(pred_e, true_e), 0).sum() / mask_e.sum()

    # Compute metrics
    # loss_x = loss_fn(pred_x, true_x).mean() if true_x.size else 0
    # loss_e = loss_fn(pred_e, true_e).mean() if true_e.size else 0
    loss_y = loss_fn(pred_graph.y, true_graph.y).mean()
    return loss_x + lambda_train_e * loss_e + lambda_train_y * loss_y


@typed
def val_loss(
    *,
    T: SInt,
    target: GraphDistribution,
    pred: GraphDistribution,
    nodes_dist: NodesDistribution,
    transition_model: TransitionModel,
    state: TrainState,
    prior_dist: Distribution,
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
        transition_model=transition_model,
        prior_dist=prior_dist,
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


@jit
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
    """
    if data.edge_index.size == 0:
        print("Found a batch with no edges. Skipping.")
        return state, 0.0
    """
    # X, E, node_mask = to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
    # dense_data = EmbeddedGraph(x=X, e=E, y=data.y, mask=node_mask)
    # dense_data =   # TODO: cache this!
    z_t = df.apply_random_noise(
        graph=dense_data,
        test=False,
        rng=rng,
        T=array(diffusion_steps),
        transition_model=transition_model,
    )
    # concatenates the extra features to the graph
    graph_dist_with_extra_data = z_t | extra_features.compute(z_t)

    def loss_fn(params: FrozenDict):
        raw_pred = state.apply_fn(  # TODO: suffix dist for distributions
            params,
            graph_dist_with_extra_data.x,
            graph_dist_with_extra_data.e,
            graph_dist_with_extra_data.y,
            z_t.mask.astype(float),
            rngs={"dropout": dropout_train_key},
        )
        pred = GraphDistribution(
            x=raw_pred.x,
            e=raw_pred.e,
            y=raw_pred.y,
            mask=z_t.mask,
        )  # consider changing the name of this class to something that suggests ditribution
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
    """
    The following is the prior of the data. It is used to generate matrices Q_t, 
    and also to compute the prior loss term in the ELBO.
    """
    # TODO: replace hardcoded values with computed ones
    prior_dist = Distribution(
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
    # Key idea proposed in the DiGress paper: use the prior distribution to gnerate Q_t
    transition_model = TransitionModel.create(
        x_marginals=prior_dist.x,
        e_marginals=prior_dist.e,
        y_classes=output_dims["y"],
        diffusion_steps=config.diffusion_steps,
    )
    # Track best eval accuracy
    for epoch_idx in range(1, num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        state, _ = train_epoch(
            rng=rng,
            state=state,
            train_loader=train_loader,
            epoch=epoch_idx,
            diffusion_steps=np.array(config.diffusion_steps),
            transition_model=transition_model,
        )
        """
        val_loss = val_epoch(
            rng=rng,
            state=state,
            val_loader=val_loader,
            config=config,
            transition_model=transition_model,
        )
        """


@jit
def val_step(
    *,
    dense_data: GraphDistribution,
    state: TrainState,
    diffusion_steps: SInt,
    rng: Key,
    transition_model: TransitionModel,
) -> SFloat:
    dropout_test_key = jax.random.fold_in(key=rng, data=state.step)
    # X, E, node_mask = to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
    X, E = dense_data.x, dense_data.e
    z_t = df.apply_random_noise(
        graph=dense_data,
        test=np.array(False),
        rng=rng,
        T=np.array(diffusion_steps),
        transition_model=transition_model,
    )
    # err.throw()
    graph_dist_with_extra_data = z_t | extra_features.compute(z_t)

    def loss_fn(params: FrozenDict):
        raw_pred = state.apply_fn(  # TODO: suffix dist for distributions
            params,
            graph_dist_with_extra_data.x,
            graph_dist_with_extra_data.e,
            graph_dist_with_extra_data.y,
            z_t.mask.astype(float),
            rngs={"dropout": dropout_test_key},
        )
        pred = GraphDistribution(
            x=raw_pred.x,
            e=raw_pred.e,
            y=raw_pred.y,
            mask=z_t.mask,
        )  # consider changing the name of this class to something that suggests distribution
        loss = train_loss(
            pred_graph=pred,
            true_graph=dense_data,
        )

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
    transition_model: TransitionModel,
):
    run_loss = 0.0
    for batch in tqdm(val_loader):
        loss = val_step(
            dense_data=GraphDistribution.from_sparse_torch(batch),
            state=state,
            diffusion_steps=int(config.diffusion_steps),
            rng=rng,
            transition_model=transition_model,
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
    for batch in tqdm(train_loader):
        state, loss = train_step(
            dense_data=GraphDistribution.from_sparse_torch(batch),
            i=np.array(epoch),
            state=state,
            diffusion_steps=diffusion_steps,
            rng=rng,
            transition_model=transition_model,
        )
        run_loss += loss
    avg_loss = run_loss / len(train_loader)
    # prints the average loss for the epoch
    print(f"Epoch {epoch} loss: {avg_loss:.4f}")
    return state, avg_loss
