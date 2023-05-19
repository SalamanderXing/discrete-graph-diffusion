"""
Entrypoint of the discrete denoising diffusion model.
The function `train_model` is the main entrypoint.
"""
import os
from typing import Callable, cast
from jax import numpy as np, Array
from flax.training import train_state
import matplotlib.pyplot as plt
import flax.linen as nn
from jax.debug import print as jprint  # type: ignore
from jaxtyping import Float, Int, Bool
from time import time
import jax_dataclasses as jdc
import jax
import optax
import wandb
from typing import Iterable, cast
import ipdb

from rich import print as print
from tqdm import tqdm
from flax.core.frozen_dict import FrozenDict
from mate.jax import SFloat, SInt, SBool, SUInt, typed, Key, jit

from ...shared.graph import SimpleGraphDist
from . import diffusion_functions as df
from .config import TrainingConfig
from .types import (
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

    @jit
    def __call__(
        self, g: GraphDistribution, t: Int[Array, "batch_size"]
    ) -> GraphDistribution:
        temporal_embeddings = self.transition_model.temporal_embeddings[t][:, None]
        # print(f"[blue]Init nodes[/blue]: {g.x.shape}")
        # print(f"[orange]Init nodes[/orange]: {g.e.shape}")
        pred_x, pred_e = self.state.apply_fn(
            self.state.params,
            g.x + temporal_embeddings,
            g.e,
            # g.mask,  # .astype(float),
            rngs={"dropout": self.dropout_rng},
            # deterministic=True,
        )
        pred = GraphDistribution.masked(
            x=jax.nn.softmax(pred_x, -1),
            e=jax.nn.softmax(pred_e, -1),
            edges_counts=g.edges_counts,
            nodes_counts=g.nodes_counts,
        )
        return pred


@jdc.pytree_dataclass
class GetProbabilityFromParams(jdc.EnforcedAnnotationsMixin):
    params: FrozenDict
    state: TrainState
    dropout_rng: Key
    transition_model: TransitionModel
    deterministic: SBool = False

    @jit
    def __call__(
        self, g: GraphDistribution, t: Int[Array, "batch_size"]
    ) -> GraphDistribution:
        temporal_embeddings = self.transition_model.temporal_embeddings[t][:, None]
        pred_x, pred_e = self.state.apply_fn(
            self.params,
            g.x + temporal_embeddings,
            g.e,
            # g.mask,  # .astype(float),
            rngs={"dropout": self.dropout_rng},
            # deterministic=False,
        )
        pred = GraphDistribution.masked(
            x=jax.nn.softmax(pred_x, -1),
            e=jax.nn.softmax(pred_e, -1),
            edges_counts=g.edges_counts,
            nodes_counts=g.nodes_counts,
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
    nodes_dist: Array,
    rng_key: Key,
    full: SBool = False,
) -> dict[str, Float[Array, "batch_size"]]:
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
        n_t_samples=1 if not full else 0,
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
    # print(f"{kl_prior=} {loss_all_t=} {reconstruction_logp=}")
    # log_n = np.log()
    base = 2
    current_node_sizes = (target.x[:, :, -1] == 0).sum(-1)
    assert (current_node_sizes <= target.x.shape[1]).all()
    log_probs = np.log(nodes_dist[current_node_sizes]) / np.log(base)
    # n_edges = (target.e[..., -1] == 0).sum((1, 2))
    # print(f"{n_edges=}")
    # n_edges = target.e.shape[1] * (target.e.shape[1] - 1)
    edges_counts = target.edges_counts * 2
    tot_loss = (-log_probs + kl_prior + loss_all_t - reconstruction_logp) / edges_counts
    return {
        "kl_prior": kl_prior / edges_counts,
        "diffusion_loss": loss_all_t / edges_counts,
        "rec_logp": reconstruction_logp / edges_counts,
        "bpe": tot_loss,
    }


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
    nodes_dist: Array,
    nodes_prior: Array,
    edges_prior: Array,
) -> float:
    train_loader = cast(Iterable[SimpleGraphDist], train_loader)
    val_loader = cast(Iterable[SimpleGraphDist], val_loader)
    state, transition_model = setup(
        model=model,
        params=params,
        lr=lr,
        config=config,
        rngs=rngs,
        epochs=num_epochs,
        nodes_prior=nodes_prior,
        edges_prior=edges_prior,
    )
    if action == "test":
        best_val_loss, val_time = val_epoch(
            rng=rngs["params"],
            state=state,
            val_loader=val_loader,
            config=config,
            transition_model=transition_model,
            nodes_dist=nodes_dist,
        )
        print(f"Validation loss: {best_val_loss:.4f} time: {val_time:.4f}")
    elif action == "train":
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
            nodes_dist=nodes_dist,
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


def prettify(val: dict[str, Float[Array, ""]]) -> dict[str, float]:
    return {k: float(f"{v.tolist():.4f}") for k, v in val.items()}


@typed
def train_all_epochs(
    num_epochs: int,
    rngs: dict[str, Key],
    state: TrainState,
    train_loader: Iterable[SimpleGraphDist],
    val_loader: Iterable[SimpleGraphDist],
    config: TrainingConfig,
    transition_model: TransitionModel,
    save_path: str,
    ds_name: str,
    nodes_dist: Array,
):
    val_losses = []
    train_losses = []
    for epoch_idx in range(1, num_epochs + 1):
        rng, _ = jax.random.split(rngs["params"])
        print(
            f"[green bold]Epoch[/green bold]: {epoch_idx} \n\n[underline]Validating[/underline]"
        )
        val_loss, val_time = val_epoch(
            rng=rng,
            state=state,
            val_loader=val_loader,
            config=config,
            transition_model=transition_model,
            nodes_dist=nodes_dist,
        )
        val_losses.append(val_loss)
        print(
            f"[green underline]Validation[/green underline] \ncurrent={prettify(val_loss)}\nbest={prettify(min(val_losses, key=lambda x: x['eval_bpe']))}\ntime={val_time:.4f}"
        )
        state, train_loss, train_time = train_epoch(
            rng=rng,
            state=state,
            train_loader=train_loader,
            epoch=epoch_idx,
            diffusion_steps=config.diffusion_steps,
            transition_model=transition_model,
        )
        train_losses.append(train_loss)
        print(
            f"[red underline]Train[/red underline]\nloss={train_loss:.5f} best={min(train_losses):.5f} time={train_time:.4f}"
        )
        wandb.log({"train_loss": train_loss, "val_loss": val_loss}, epoch_idx)

        # import json
        #
        # with open(os.path.join(save_path, "val_losses.json"), "w") as f:
        #     json.dump(val_losses, f)
        #
        # with open(os.path.join(save_path, "train_losses.json"), "w") as f:
        #     json.dump(train_losses, f)
        #
        # if val_loss == min(val_losses):
        #     print("Saving model")
        #     import pickle
        #
        #     pickle.dump(
        #         state.params, open(os.path.join(save_path, "checkpoint.pickle"), "wb")
        #     )

    rng, _ = jax.random.split(rngs["params"])
    val_loss, val_time = val_epoch(
        rng=rng,
        state=state,
        val_loader=val_loader,
        config=config,
        transition_model=transition_model,
        full=True,
        nodes_dist=nodes_dist,
    )
    print(f"Final loss: {val_loss:.4f} best={min(val_losses):.5f} time: {val_time:.4f}")


def setup(
    model: nn.Module,
    params: FrozenDict,
    rngs: dict[str, Key],
    lr: float,
    config: TrainingConfig,
    epochs: int,
    nodes_prior: Array,
    edges_prior: Array,
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
        x_priors=nodes_prior,
        e_priors=edges_prior,
        diffusion_steps=config.diffusion_steps,
        temporal_embedding_dim=config.num_node_features
        - 1,  # FIXME: same as the node embedding. Should improve this parametrization
    )
    return state, transition_model


@typed
def val_step(
    *,
    dense_data: GraphDistribution,
    state: TrainState,
    diffusion_steps: SInt,
    rng: Key,
    transition_model: TransitionModel,
    nodes_dist: Array,
    full=False,
) -> dict[str, Float[Array, "batch_size"]]:
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
        nodes_dist=nodes_dist,
        full=full,
    )


@typed
def val_epoch(
    *,
    val_loader: Iterable[SimpleGraphDist],
    state: TrainState,
    config: TrainingConfig,
    rng: Key,
    transition_model: TransitionModel,
    nodes_dist: Array,
    full=False,
) -> tuple[dict[str, Float[Array, ""]], float]:
    run_losses: list[dict[str, Float[Array, "batch_size"]]] = []
    t0 = time()
    n_val_steps = 10
    for i in tqdm(range(n_val_steps)):
        simple_graph_dist = next(val_loader)
        graph_dist = GraphDistribution.from_simple(simple_graph_dist)
        losses = val_step(
            dense_data=graph_dist,  # , mask.astype(bool)),
            state=state,
            diffusion_steps=config.diffusion_steps,
            rng=rng,
            transition_model=transition_model,
            nodes_dist=nodes_dist,
            full=full,
        )
        run_losses.append(losses)
    t1 = time()
    total_time = t1 - t0
    avg_losses = {
        f"eval_{k}": np.mean(np.array([r[k] for r in run_losses]))
        for k in run_losses[0].keys()
    }
    # avg_loss = np.mean(np.array(run_loss)).tolist()
    return avg_losses, total_time


@typed
def train_epoch(
    *,
    train_loader: Iterable[SimpleGraphDist],
    epoch: int,
    state: TrainState,
    diffusion_steps: SInt,
    rng: Key,
    transition_model: TransitionModel,
):
    run_losses = []
    tot_len: int = 0
    t0 = time()
    n_train_steps = 100
    for batch_index in tqdm(range(n_train_steps)):
        simple_graph_dist = next(train_loader)
        graph_dist = GraphDistribution.from_simple(simple_graph_dist)
        # dense_data = GraphDistribution.from_sparse_torch(batch)
        # dense_data = GraphDistribution.masked(x, e)  # , mask.astype(bool))
        state, loss = train_step(
            g=graph_dist,
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
