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


# from ...shared.graph import SimpleGraphDist
from ...shared.graph import graph_distribution as gd
from . import diffusion_functions as df
from .types import (
    TransitionModel,
)

GraphDistribution = gd.GraphDistribution


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
        return pred_graph


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
        return pred_graph


@typed
def compute_train_loss(
    g: GraphDistribution,
    rng: Key,
    transition_model: TransitionModel,
    get_probability: Callable[
        [GraphDistribution, Int[Array, "batch_size"]], GraphDistribution
    ],
    match_edges: SBool = True,
    stepwise: SBool = False,
):
    t = jax.random.randint(rng, (g.batch_size,), 1, len(transition_model.q_bars))
    q_bars = transition_model.q_bars[t]
    z = (g @ q_bars).sample_one_hot(rng)
    pred_graph = get_probability(z, t)
    # print(f"{pred_graph.edges.argmax(-1)}")
    # print(f"{z.edges.argmax(-1)}")
    # ce = gd.cross_entropy(pred_graph, z)  # .mean()

    # print(ce.shape)
    # mse = (z - pred_graph) ** 2

    # print(pred_graph.nodes.argmax(-1)[0])
    # print(pred_graph.edges.argmax(-1)[0])
    target = z if stepwise else g
    loss_type = "ce"
    if loss_type == 'mse':
        mse_nodes = (target.nodes - pred_graph.nodes) ** 2
        mse_edges = (target.edges - pred_graph.edges) ** 2
        weight = 1000
        weighted_mse_nodes = np.where(z.nodes != 1, weight, 1) * mse_nodes
        count_diff = (
            (target.edges.argmax(-1) != 0).sum() - (pred_graph.edges.argmax(-1) != 0).sum()
        ) ** 2
        loss = weighted_mse_nodes.mean() + mse_edges.mean() + match_edges * count_diff
    elif loss_type == 'ce':
        loss = gd.cross_entropy(pred_graph, target).mean()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return loss


@typed
def compute_val_loss(
    *,
    target: GraphDistribution,
    transition_model: TransitionModel,
    get_probability: Callable[
        [GraphDistribution, Int[Array, "batch_size"]], GraphDistribution
    ],
    nodes_dist: Array,
    rng_key: Key,
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
    # edges_counts = jax.lax.select(
    #     bits_per_edge, target.edges_counts * 2, np.ones(len(target.edges_counts), int)
    # )
    tot_loss = (-log_probs + kl_prior + loss_all_t - reconstruction_logp)
    return {
        "log_pn": log_probs,
        "kl_prior": kl_prior,
        "diffusion_loss": loss_all_t,
        "rec_logp": reconstruction_logp,
        "nll": tot_loss,
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
    match_edges: bool = True,
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
            match_edges=match_edges,
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


DataLoader = Iterable[GraphDistribution]


from dataclasses import dataclass


@dataclass
class Trainer:
    model: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    params: FrozenDict
    num_epochs: int
    rngs: dict[str, Key]
    learning_rate: float
    save_path: str
    ds_name: str
    nodes_dist: Array
    nodes_prior: Array
    edges_prior: Array
    bits_per_edge: bool
    diffusion_steps: int
    noise_schedule_type: str
    log_every_steps: int
    max_num_nodes: int
    num_edge_features: int
    num_node_features: int
    match_edges: bool = True
    restart: bool = False
    plot_every_steps: int = 2

    def __post_init__(
        self,
    ):
        self.n = next(iter(self.train_loader)).nodes.shape[1]
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=self.params,
            tx=get_optimizer(self.learning_rate),
            key=self.rngs["dropout"],
            last_loss=1000000,
            lr=self.learning_rate,
        )

        self.transition_model = TransitionModel.create(
            schedule_type=self.noise_schedule_type,
            x_priors=self.nodes_prior,
            e_priors=self.edges_prior,
            diffusion_steps=self.diffusion_steps,
            temporal_embedding_dim=128,
            n=self.n,
        )
        self.plot_path = os.path.join(self.save_path, "plots")
        os.system(f"rm -rf {self.plot_path}")
        os.makedirs(self.plot_path, exist_ok=True)
        orbax_checkpointer = checkpoint.PyTreeCheckpointer()
        options = checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
        self.checkpoint_manager = checkpoint.CheckpointManager(
            self.save_path, orbax_checkpointer, options
        )
        # return best_val_loss

    @typed
    def __val_epoch(
        self,
        *,
        rng: Key,
        exhaustive: bool = False,
        n_val_steps: int = 30,
    ) -> tuple[dict[str, Float[Array, ""]], float]:
        run_losses: list[dict[str, Float[Array, "batch_size"]]] = []
        t0 = time()
        # n_val_steps = (
        #     30  # 10 if not exhaustive else len(self.val_loader)  # type: ignore
        # )
        for i in tqdm(range(n_val_steps)):
            graph_dist = next(self.val_loader)  # type: ignore
            losses = val_step(
                dense_data=graph_dist,  # , mask.astype(bool)),
                state=self.state,
                diffusion_steps=self.diffusion_steps,
                rng=rng,
                transition_model=self.transition_model,
                nodes_dist=self.nodes_dist,
            )
            run_losses.append(losses)
        t1 = time()
        total_time = t1 - t0
        avg_losses = {
            k: np.mean(np.array([r[k] for r in run_losses]))
            for k in run_losses[0].keys()
        }
        # avg_loss = np.mean(np.array(run_loss)).tolist()
        return avg_losses, total_time

    @typed
    def plot_preds(
        self,
        plot_path: str,
        load_from_disk: bool = True,
        epoch: int = -1,
    ):
        rng = jax.random.PRNGKey(0)
        if load_from_disk:
            self.__restore_checkpoint()
        model = GetProbabilityFromState(self.state, rng, self.transition_model)

        print(f"Plotting noised graphs to {plot_path}")
        T = len(self.transition_model.q_bars)
        g_batch = next(iter(self.val_loader))
        g = g_batch[np.array([0])].repeat(len(self.transition_model.q_bars))
        q_bars = self.transition_model.q_bars  # [timesteps]
        posterior_samples = (g @ q_bars).sample_one_hot(rng)
        timesteps = np.arange(len(self.transition_model.q_bars))
        model_probs = model(posterior_samples, timesteps)
        val_losses = compute_val_loss(
            target=posterior_samples,
            transition_model=self.transition_model,
            get_probability=model,
            nodes_dist=self.nodes_dist,
            rng_key=jax.random.PRNGKey(23),
        )["nll"]
        corr = np.corrcoef(val_losses, np.arange(len(self.transition_model.q_bars)))[
            0, 1
        ]
        if epoch > -1:
            wandb.log({"corr_t_vs_elbo": corr}, step=epoch)
        model_samples = model_probs.argmax()
        GraphDistribution.plot(
            [posterior_samples, model_samples],
            location=plot_path,
            share_position_among_graphs=True,
            title=f"Correlation: {corr:.3f}",
        )

    def __restore_checkpoint(self):
        if self.checkpoint_manager.latest_step() is not None:
            state_dict = self.checkpoint_manager.restore(
                self.checkpoint_manager.latest_step()
            )
            self.state = TrainState(
                tx=get_optimizer(lr=state_dict["lr"]),
                apply_fn=self.state.apply_fn,
                **{k: v for k, v in state_dict.items()},
            )
            print(
                f"[yellow]Restored from epoch {self.checkpoint_manager.latest_step()}[/yellow]"
            )
        else:
            print("[yellow]No checkpoint found, starting from scratch[/yellow]")

    @typed
    def train(
        self,
    ):
        if self.restart:
            self.__restore_checkpoint()
            print("Restarting training")

        val_losses = []
        train_losses = []
        patience = 10
        current_patience = patience
        stopping_criterion = -5
        rng, _ = jax.random.split(self.rngs["params"])
        for epoch_idx in range(1, self.num_epochs + 1):
            rng, rng_this_epoch = jax.random.split(rng)  # TODO use that
            print(
                f"[green bold]Epoch[/green bold]: {epoch_idx} \n\n[underline]Validating[/underline]"
            )
            val_loss, val_time = self.__val_epoch(
                rng=rng_this_epoch,
            )
            val_losses.append(val_loss)
            print(
                f"""[green underline]Validation[/green underline]
                current={prettify(val_loss)}
                best={prettify(min(val_losses, key=lambda x: x['nll']))}
                time={val_time:.4f}"""
            )
            avg_loss = val_losses[-1]["nll"]
            if self.state.last_loss < avg_loss:
                if current_patience <= 0:
                    new_lr = self.state.lr * 0.1
                    if new_lr >= 1e-05:
                        print(
                            f"[red] learning rate did not decrease. Reducing lr to {new_lr} [/red]"
                        )
                        self.state = self.state.replace(
                            lr=new_lr, tx=get_optimizer(new_lr), last_loss=avg_loss
                        )
                        current_patience = patience
                else:
                    current_patience -= 1
                    if current_patience <= stopping_criterion:
                        print(
                            f"[red] stopping criterion reached. Stopping training [/red]"
                        )
                        break

            train_loss, train_time = self.train_epoch(
                rng=rng,
                epoch=epoch_idx,
            )
            train_losses.append(train_loss)

            wandb.log({"train_loss": train_loss, "val_loss": val_loss}, epoch_idx)

            if avg_loss < self.state.last_loss:
                self.state = self.state.replace(last_loss=avg_loss)
                print(
                    f"[red underline]Train[/red underline]\nloss={train_loss:.5f} best={min(train_losses):.5f} time={train_time:.4f}"
                )
                print(f"[yellow] Saving checkpoint[/yellow]")
                self.checkpoint_manager.save(epoch_idx, self.state)
                print(f"[yellow] Saved! [/yellow]")

                if epoch_idx % self.plot_every_steps == 0:
                    self.plot_preds(
                        plot_path=os.path.join(
                            self.plot_path, f"{epoch_idx}_preds.png"
                        ),
                        load_from_disk=False,
                    )
                    self.sample(
                        restore_checkpoint=False,
                        save_to=os.path.join(
                            self.plot_path, f"{epoch_idx}_samples.png"
                        ),
                    )

        rng, _ = jax.random.split(self.rngs["params"])
        val_loss, val_time = self.__val_epoch(
            rng=rng,
            exhaustive=True,
        )
        print(
            f"Final loss: {val_loss:.4f} best={min(val_losses):.5f} time: {val_time:.4f}"
        )

    def sample(self, restore_checkpoint: bool = True, save_to: str | None = None):
        if restore_checkpoint:
            self.__restore_checkpoint()
        import random

        rng = jax.random.PRNGKey(random.randint(0, 1000000))
        model = GetProbabilityFromState(self.state, rng, self.transition_model)
        g_batch = next(iter(self.val_loader))[np.array([0])]
        # g = GraphDistribution.noise(
        #     rng,
        #     num_node_features=g_batch.nodes.shape[-1],
        #     num_edge_features=g_batch.edges.shape[-1],
        #     num_nodes=g_batch.nodes.shape[1],
        #     batch_size=1,
        # )
        limit_dist = g_batch.set(
            "edges", jax.nn.softmax(self.transition_model.limit_dist.edges + 0.1)
        ).set("nodes", jax.nn.softmax(self.transition_model.limit_dist.nodes + 0.1))
        g = limit_dist.sample_one_hot(rng)
        # g = g.set("edges_counts", ((g.edges.argmax(-1) != 0).sum() / 2).astype(int))
        gs = [g]
        # print(f"Plotting noised graphs to {plot_path}")
        # timesteps = np.array([len(self.transition_model.q_bars)] * len(g))
        for t in reversed(range(1, self.transition_model.diffusion_steps)):
            # g = g_batch[np.array([0])].repeat(len(self.transition_model.q_bars))
            # posterior_samples = (g @ q_bars).sample_one_hot(rng)
            timesteps = np.array([t] * len(g))
            model_probs = model(g, timesteps)
            # g = model_probs.argmax()
            g = df.posterior_distribution(
                g=g,
                g_t=model_probs,
                t=timesteps,
                transition_model=self.transition_model,
            )
            gs.append(g)
            # print(t)

        model_samples = GraphDistribution.concatenate(list(reversed(gs)))

        GraphDistribution.plot(
            [model_samples], share_position_among_graphs=True, location=save_to
        )

    @typed
    def train_epoch(
        self,
        *,
        epoch: int,
        rng: Key,
    ):
        run_losses = []
        t0 = time()
        n_train_steps = 40
        print(f"[pink] {self.state.lr=} [/pink]")
        for batch_index in tqdm(range(n_train_steps)):
            graph_dist = next(self.train_loader)
            state, loss = train_step(
                g=graph_dist,
                i=epoch,
                state=self.state,
                diffusion_steps=self.diffusion_steps,
                rng=rng,
                transition_model=self.transition_model,
                nodes_dist=self.nodes_dist,
                bits_per_edge=self.bits_per_edge,
                match_edges=self.match_edges,
            )
            self.state = state
            run_losses.append(loss)
        avg_loss = np.mean(np.array(run_losses)).tolist()
        t1 = time()
        tot_time = t1 - t0
        return avg_loss, tot_time


# def noise(transition_model, g, t, rng):
#     q = transition_model.q_bars[np.array(t)[None]]
#     probs = g @ q
#     noised = probs.sample_one_hot(rng)
#     return noised
#


@typed
def plot_noised_graphs(train_loader, transition_model: TransitionModel, save_path: str):
    print(f"Plotting noised graphs to {save_path}")
    T = len(transition_model.q_bars)
    g_batch = next(iter(train_loader))
    g = g_batch[np.array([0])].repeat(len(transition_model.q_bars))
    rng = jax.random.PRNGKey(0)
    q_bars = transition_model.q_bars  # [timesteps]
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
) -> dict[str, Float[Array, "batch_size"]]:
    dropout_test_key = jax.random.fold_in(key=rng, data=state.step)

    get_probability = GetProbabilityFromState(
        state=state, dropout_rng=dropout_test_key, transition_model=transition_model
    )
    return compute_val_loss(
        target=dense_data,
        transition_model=transition_model,
        rng_key=rng,
        get_probability=get_probability,
        nodes_dist=nodes_dist,
        bits_per_edge=bits_per_edge,
    )
