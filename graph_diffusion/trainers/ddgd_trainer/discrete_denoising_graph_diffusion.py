"""
Entrypoint of the discrete denoising diffusion model.
The function `train_model` is the main entrypoint.
"""
import sys
from memory_profiler import profile
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
from orbax import checkpoint


from ...shared.graph import SimpleGraphDist
from ...shared.graph_distribution import GraphDistribution
from . import diffusion_functions as df
from .config import TrainingConfig
from .types import (
    TransitionModel,
)


class TrainState(train_state.TrainState):
    key: jax.random.KeyArray  # type: ignore
    lr: SFloat
    last_loss: SFloat


@jdc.pytree_dataclass
class GetProbabilityFromState(jdc.EnforcedAnnotationsMixin):
    state: TrainState
    dropout_rng: Key
    transition_model: TransitionModel

    @typed
    def __call__(
        self, g: GraphDistribution, t: Int[Array, "batch_size"]
    ) -> GraphDistribution:
        temporal_embeddings = self.transition_model.temporal_embeddings[t]
        # print(f"[blue]Init nodes[/blue]: {g.x.shape}")
        # print(f"[orange]Init nodes[/orange]: {g.e.shape}")
        pred_graph = self.state.apply_fn(
            self.state.params,
            g,
            temporal_embeddings,
            deterministic=True,
            rngs={"dropout": self.dropout_rng},
        )
        pred = GraphDistribution.create(
            nodes=jax.nn.softmax(pred_graph.nodes, -1),
            edges=jax.nn.softmax(pred_graph.edges, -1),
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

    @typed
    def __call__(
        self, g: GraphDistribution, t: Int[Array, "batch_size"]
    ) -> GraphDistribution:
        temporal_embeddings = self.transition_model.temporal_embeddings[t]
        pred_graph = self.state.apply_fn(
            self.params,
            g,
            temporal_embeddings,
            deterministic=False,
            rngs={"dropout": self.dropout_rng},
            # deterministic=False,
        )
        pred = GraphDistribution.create(
            nodes=jax.nn.softmax(pred_graph.nodes, -1),
            edges=jax.nn.softmax(pred_graph.edges, -1),
            edges_counts=g.edges_counts,
            nodes_counts=g.nodes_counts,
        )

        return pred


@typed
def compute_train_loss(
    g: GraphDistribution,
    rng: Key,
    transition_model: TransitionModel,
    get_probability: Callable[
        [GraphDistribution, Int[Array, "batch_size"]], GraphDistribution
    ],
):
    t = jax.random.randint(rng, (g.batch_size,), 1, len(transition_model.q_bars))
    q_bars = transition_model.q_bars[t]
    z = (g @ q_bars).sample_one_hot(rng)
    pred_graph = get_probability(z, t)
    loss_fn = optax.softmax_cross_entropy  # (logits=x, labels=y).sum()
    true_x = g.nodes.reshape(-1, g.nodes.shape[-1])  # (bs*n, dx)
    true_e = g.edges.reshape(-1, g.edges.shape[-1])  # (bs*n*n, de)
    pred_x = pred_graph.nodes.reshape(-1, pred_graph.nodes.shape[-1])  # (bs*n, dx)
    pred_e = pred_graph.edges.reshape(-1, pred_graph.edges.shape[-1])  # (bs*n*n, de)

    # Remove masked rows
    mask_x, mask_e = g.masks()

    # loss_x = np.where(mask_x, loss_fn(pred_x, true_x).mean(-1), 0).sum() / mask_x.sum()
    # loss_e = np.where(mask_e, loss_fn(pred_e, true_e).mean(-1), 0).sum() / mask_e.sum()
    n = g.nodes.shape[1]
    uba = loss_fn(pred_x, true_x).reshape(g.batch_size, n, -1).mean(-1)
    alla = loss_fn(pred_e, true_e).reshape(g.batch_size, n, n, -1).mean(-1)
    loss_x = (uba * mask_x).sum()
    loss_e = (alla * mask_e).sum()
    rec_loss = loss_x + loss_e
    reconstruction_logp = df.reconstruction_logp(
        rng_key=rng,
        g=g,
        transition_model=transition_model,
        get_probability=get_probability,
        n_samples=1,
        base=np.e,
    )
    return -reconstruction_logp.mean() + rec_loss


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
    bits_per_edge: SBool = False,
) -> dict[str, Float[Array, "batch_size"]]:
    base = jax.lax.select(bits_per_edge, 2.0, np.e)

    # 1.  log_prob of the target graph under the nodes distribution (based on # of nodes)
    log_probs = np.log(nodes_dist[target.nodes_counts]) / np.log(base)

    # 2. The KL between q(z_T | x) and p(z_T) = (simply an Empirical prior). Should be close to zero.
    kl_prior = df.kl_prior(
        target=target,
        transition_model=transition_model,
        bits_per_edge=bits_per_edge,
    )
    # 3. Diffusion loss
    loss_all_t = df.compute_lt(
        rng=rng_key,
        g=target,
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
        base=base,
    )
    # assert (current_node_sizes <= target.x.shape[1]).all()

    edges_counts = jax.lax.select(
        bits_per_edge, target.edges_counts * 2, np.ones(len(target.edges_counts), int)
    )
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
    nodes_dist: Array,
    bits_per_edge: bool = True,
):  # -> tuple[TrainState, Float[Array, "batch_size"]]:
    dropout_train_key = jax.random.fold_in(key=rng, data=state.step)

    def loss_fn(params: FrozenDict):
        get_probability = GetProbabilityFromParams(
            state=state,
            params=params,
            transition_model=transition_model,
            dropout_rng=dropout_train_key,
        )
        # losses = compute_val_loss(
        #     target=g,
        #     diffusion_steps=diffusion_steps,
        #     transition_model=transition_model,
        #     rng_key=rng,
        #     get_probability=get_probability,
        #     nodes_dist=nodes_dist,
        #     full=False,
        #     bits_per_edge=bits_per_edge,
        # )
        losses = compute_train_loss(
            g=g,
            transition_model=transition_model,
            rng=rng,
            get_probability=get_probability,
        )

        # jprint("loss {loss}", loss=loss)
        # return losses["bpe"].mean(), None
        return losses, None

    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = gradient_fn(state.params)
    # print_gradient_analysis(grads)
    # grads = jax.tree_map(
    #     lambda x: np.where(np.isnan(x), 0.0, x), grads
    # )  # remove the nans :(
    # state = state.replace()
    state = state.apply_gradients(grads=grads)
    return state, loss * g.batch_size


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
    bits_per_edge: bool,
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
    elif action in ("train", "restart"):
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
            bits_per_edge=bits_per_edge,
            restart=action == "restart",
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

    elif action == "plot_noise":
        os.makedirs(os.path.join(save_path, "plots"), exist_ok=True)
        plot_noised_graphs(
            val_loader, transition_model, os.path.join(save_path, "plots")
        )
    else:
        raise ValueError(f"Unknown action {action}")

    return best_val_loss


def noise(transition_model, g, t, rng):
    q = transition_model.q_bars[np.array(t)[None]]
    probs = g @ q
    noised = probs.sample_one_hot(rng)
    return noised


@typed
def plot_noised_graphs(train_loader, transition_model: TransitionModel, save_path: str):
    print(f"Plotting noised graphs to {save_path}")
    T = len(transition_model.q_bars)
    simple = next(iter(train_loader))
    g = GraphDistribution.from_simple(simple)[np.array([0])].repeat(
        len(transition_model.q_bars)
    )
    rng = jax.random.PRNGKey(0)
    # new_file_name = os.path.join(save_path, f"noise_original.png")
    # g.plot(location=new_file_name)
    # q = transition_model.q_bars[np.array(45)[None]]
    # probs_1 = g @ q
    # q = transition_model.q_bars[np.array(20)[None]]
    # probs_2 = g @ q
    # noised = probs_1.sample_one_hot(rng)
    # file_name = os.path.join(save_path, f"noise_test.png")
    # noised_graph = noised.plot(location=file_name)
    timesteps = np.arange(len(transition_model.alpha_bars))
    q_bars = transition_model.q_bars[timesteps]
    probs = (g @ q_bars).sample_one_hot(rng)
    probs.plot()


def prettify(val: dict[str, Float[Array, ""]]) -> dict[str, float]:
    return {k: float(f"{v.tolist():.4f}") for k, v in val.items()}


def get_optimizer(lr):
    return optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adamw(learning_rate=lr, weight_decay=1e-5),
    )


def save_stuff():
    pass


# @typed
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
    bits_per_edge: bool,
    from_epoch: int = 0,
    restart: bool = False,
):
    orbax_checkpointer = checkpoint.PyTreeCheckpointer()
    options = checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_manager = checkpoint.CheckpointManager(
        save_path, orbax_checkpointer, options
    )
    if restart:
        print("Restarting training")
        if checkpoint_manager.latest_step() is not None:
            state_dict = checkpoint_manager.restore(checkpoint_manager.latest_step())
            state = TrainState(
                tx=get_optimizer(lr=state_dict["lr"]),
                apply_fn=state.apply_fn,
                **{k: v for k, v in state_dict.items()},
            )
            print(
                f"[yellow]Restored from epoch {checkpoint_manager.latest_step()}[/yellow]"
            )
        else:
            print("[yellow]No checkpoint found, starting from scratch[/yellow]")

    val_losses = []
    train_losses = []
    patience = 3
    current_patience = patience
    rng, _ = jax.random.split(rngs["params"])
    for epoch_idx in range(from_epoch, num_epochs):
        rng, rng_this_epoch = jax.random.split(rng)  # TODO use that
        print(
            "\n".join(
                f"{name} : {sys.getsizeof(value):.2f} B"
                for name, value in locals().items()
            )
        )
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
            f"""[green underline]Validation[/green underline]
            current={prettify(val_loss)}
            best={prettify(min(val_losses, key=lambda x: x['eval_bpe']))}
            time={val_time:.4f}"""
        )
        avg_loss = val_losses[-1]["eval_bpe"]
        if state.last_loss < avg_loss:
            if current_patience == 0:
                new_lr = state.lr * 0.1
                if new_lr >= 1e-05:
                    print(
                        f"[red] learning rate did not decrease. Reducing lr to {new_lr} [/red]"
                    )
                    state = state.replace(
                        lr=new_lr, tx=get_optimizer(new_lr), last_loss=avg_loss
                    )
                    current_patience = patience
            else:
                current_patience -= 1
        if avg_loss < state.last_loss:
            state = state.replace(last_loss=avg_loss)

        state, train_loss, train_time = train_epoch(
            rng=rng,
            state=state,
            train_loader=train_loader,
            epoch=epoch_idx,
            diffusion_steps=config.diffusion_steps,
            transition_model=transition_model,
            nodes_dist=nodes_dist,
            bits_per_edge=bits_per_edge,
        )
        train_losses.append(train_loss)

        wandb.log({"train_loss": train_loss, "val_loss": val_loss}, epoch_idx)
        print(
            f"[red underline]Train[/red underline]\nloss={train_loss:.5f} best={min(train_losses):.5f} time={train_time:.4f}"
        )
        print(f"[yellow] Saving checkpoint[/yellow]")
        checkpoint_manager.save(epoch_idx, state)
        print(f"[yellow] Saved! [/yellow]")

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
    # scheduler = optax.cosine_decay_schedule(
    #     0.001, decay_steps=epochs, alpha=0.95
    # )
    # scheduler = optax.linear_schedule(0.001, 0.000001, 1)
    initial_lr = lr
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=get_optimizer(initial_lr),
        key=rngs["dropout"],
        last_loss=1000000,
        lr=initial_lr,
    )

    transition_model = TransitionModel.create(
        x_priors=nodes_prior,
        e_priors=edges_prior,
        diffusion_steps=config.diffusion_steps,
        temporal_embedding_dim=128,
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
    bits_per_edge: bool = False,
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
        bits_per_edge=bits_per_edge,
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
    nodes_dist: Array,
    bits_per_edge: bool = True,
):
    run_losses = []
    tot_len: int = 0
    t0 = time()
    n_train_steps = 20
    print(f"[pink] {state.lr=} [/pink]")
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
            nodes_dist=nodes_dist,
            bits_per_edge=bits_per_edge,
        )

        run_losses.append(loss)
    avg_loss = np.mean(np.array(run_losses)).tolist()
    t1 = time()
    tot_time = t1 - t0
    return state, avg_loss, tot_time
