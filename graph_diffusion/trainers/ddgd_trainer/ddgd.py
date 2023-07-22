"""
Entrypoint of the discrete denoising diffusion model.
The function `train_model` is the main entrypoint.
"""
import sys
from memory_profiler import profile
import os
from . import torch_utils
from typing import Callable, cast
from jax import numpy as np, Array
from flax.training import train_state
import matplotlib.pyplot as plt
import flax.linen as nn
from jax.debug import print as jprint  # type: ignore
from jaxtyping import Float, Int, Bool
from time import time
import jax_dataclasses as jdc
from jaxtyping import jaxtyped
import jax
import optax
import wandb
from typing import Iterable, cast
import ipdb


from rich import print as print
from tqdm import tqdm
from flax.core.frozen_dict import FrozenDict
from mate.jax import SFloat, SInt, SBool, SUInt, Key  # , jit
from jax import jit
from orbax import checkpoint


# from ...shared.graph import SimpleGraphDist
from ...shared.graph import graph_distribution as gd
from . import diffusion_functions as df
from .types import (
    TransitionModel,
)
from beartype import beartype

enc = df.enc


GraphDistribution = gd.GraphDistribution


def typed(f):
    return jaxtyped(beartype(f))


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
        self, g: gd.OneHotGraph, t: Int[Array, "batch_size"]
    ) -> gd.DenseGraphDistribution:
        temporal_embeddings = self.transition_model.temporal_embeddings[t]
        # temporal_embeddings = np.zeros((1, 129))  # FIXME just a test
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
        self, g: gd.OneHotGraph, t: Int[Array, "batch_size"]
    ) -> gd.DenseGraphDistribution:
        temporal_embeddings = self.transition_model.temporal_embeddings[t]
        # temporal_embeddings = np.zeros((1, 129))  # FIXME just a test
        pred_graph = self.state.apply_fn(
            self.params,
            g,
            temporal_embeddings,
            deterministic=False,
            rngs={"dropout": self.dropout_rng},
            # deterministic=False,
        )
        return pred_graph


@jit
@typed
def compute_train_loss(
    g: gd.OneHotGraph,
    rng: Key,
    transition_model: TransitionModel,
    get_probability: Callable[
        [gd.OneHotGraph, Int[Array, "batch_size"]], gd.DenseGraphDistribution
    ],
    match_edges: SBool = True,
):
    rng_t = jax.random.fold_in(rng, enc("t"))
    rng_z = jax.random.fold_in(rng, enc("z"))
    ce = optax.softmax_cross_entropy
    t = jax.random.randint(rng_t, (g.nodes.shape[0],), 1, len(transition_model.q_bars))
    q_bars = transition_model.q_bars[t]
    z = gd.sample_one_hot(gd.matmul(g, q_bars), rng_z)
    pred_graph = gd.softmax(get_probability(z, t))
    target = g
    loss_type = "ce"
    # jax.debug.breakpoint()
    if loss_type == "mse":
        mse_nodes = (target.nodes - pred_graph.nodes) ** 2
        mse_edges = (target.edges - pred_graph.edges) ** 2
        weight = 1
        weighted_mse_nodes = np.where(z.nodes != 1, weight, 1) * mse_nodes
        count_diff = (
            (target.edges.argmax(-1) != 0).sum()
            - (pred_graph.edges.argmax(-1) != 0).sum()
        ) ** 2
        loss = np.array(
            [weighted_mse_nodes.mean(), mse_edges.mean(), match_edges * count_diff]
        ).mean()
    elif loss_type == "ce":
        loss = gd.cross_entropy(pred_graph, target).mean()
        # jax.debug.breakpoint()
    elif loss_type == "mse+ce":
        count_diff = (
            (target.edges.argmax(-1) != 0).sum()
            - (pred_graph.edges.argmax(-1) != 0).sum()
        ) ** 2

        loss = gd.cross_entropy(pred_graph, target).mean() + count_diff
    elif loss_type == "kl":
        loss = gd.softmax_kl_div(pred_graph, target).mean()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return loss


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


@jit
@jaxtyped
@beartype
def train_step(
    *,
    g: gd.OneHotGraph,
    state: TrainState,
    rng: Key,
    transition_model: TransitionModel,
    nodes_dist: Array,
):  # -> tuple[TrainState, Float[Array, "batch_size"]]:
    dropout_train_key = jax.random.fold_in(rng, enc("dropout"))
    train_rng = jax.random.fold_in(rng, enc("train"))

    # get_probability = GetProbabilityFromParams(
    #     state=state,
    #     params=state.params,
    #     transition_model=transition_model,
    #     dropout_rng=dropout_train_key,
    # )
    # # losses = df.compute_train_loss(
    # #     target=g,
    # #     transition_model=transition_model,
    # #     rng_key=rng,
    # #     get_probability=get_probability,
    # # ).mean()
    # losses = compute_train_loss(
    #     g=g,
    #     transition_model=transition_model,
    #     get_probability=get_probability,
    #     rng=rng,
    # )

    def loss_fn(params: FrozenDict):
        get_probability = GetProbabilityFromParams(
            state=state,
            params=params,
            transition_model=transition_model,
            dropout_rng=dropout_train_key,
        )
        losses = df.compute_train_loss(
            target=g,
            transition_model=transition_model,
            rng_key=train_rng,
            get_probability=get_probability,
        ).mean()
        # losses = compute_train_loss(
        #     g=g,
        #     transition_model=transition_model,
        #     get_probability=get_probability,
        #     rng=train_rng,
        # )
        return losses, None

    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = gradient_fn(state.params)
    # print_gradient_analysis(grads)
    # grads = jax.tree_map(
    #     lambda x: np.where(np.isnan(x), 0.0, x), grads
    # )  # remove the nans :(
    # state = state.replace()
    state = state.apply_gradients(grads=grads)
    return state, loss


@jit
@jaxtyped
@beartype
def val_step(
    *,
    data: gd.OneHotGraph,
    state: TrainState,
    rng: Key,
    transition_model: TransitionModel,
    nodes_dist: Array,
) -> dict[str, Float[Array, "batch_size"]]:
    dropout_test_key = jax.random.fold_in(rng, enc("dropout"))
    val_rng = jax.random.fold_in(rng, enc("val"))
    p = GetProbabilityFromState(
        state=state, dropout_rng=dropout_test_key, transition_model=transition_model
    )
    # get_probability = lambda x, t: gd.create_dense(
    #     data.nodes * 100.0,
    #     data.edges * 100.0,
    #     data.nodes_mask,
    #     data.edges_mask,
    # ) # this was a test, how does a cheating model perform?
    return df.compute_val_loss(
        target=data,
        transition_model=transition_model,
        rng_key=val_rng,
        p=p,
        nodes_dist=nodes_dist,
    )


from einop import einop


@jit
@typed
def _sample_step(
    g: gd.DenseGraphDistribution,
    transition_model: TransitionModel,
    model,
    t: Int[Array, ""],
    rng,
):
    timesteps = einop(t, " -> b", b=len(g))
    # g_t = gd.sample_one_hot(g, rng)
    # model_probs = gd.softmax(model(g_t, timesteps))
    # g = df.posterior_distribution(
    #     g=model_probs,
    #     g_t=g_t,
    #     t=timesteps,
    #     transition_model=transition_model,
    # )
    g = df.sample_p_zs_given_zt(
        p=model,
        t=timesteps,
        g_t=g,
        transition_model=transition_model,
        rng=rng,
    )
    return g


@typed
def _sample_steps(transition_model: TransitionModel, model, rng):
    g = gd.sample_one_hot(transition_model.limit_dist, rng)
    gs = [g]
    _, rng_this_epoch = jax.random.split(rng)
    timesteps = np.arange(1, transition_model.diffusion_steps)[::-1]
    for t in timesteps:
        rng = jax.random.fold_in(rng, enc(f"sample_{t}"))  # TODO use that
        # g = _sample_step(g, transition_model, model, t, rng)
        # g = df.sample_p_zs_given_zt(
        #     p=model,
        #     t=t,
        #     g_t=g,
        #     transition_model=transition_model,
        #     rng=rng,
        # )
        # g = _sample_step(g, transition_model, model, t, rng)

        timesteps = einop(t, " -> b", b=len(g))
        g = df.sample_p_zs_given_zt(
            p=model,
            t=timesteps,
            g_t=g,
            transition_model=transition_model,
            rng=rng,
        )
        gs.append(g)
    model_samples = gd.concatenate(list(reversed(gs)))
    return model_samples


@typed
def _sample(transition_model: TransitionModel, model, rng, n: SInt):
    _, rng_this_epoch = jax.random.split(rng)
    one_hot = gd.sample_one_hot(
        gd.repeat_dense(transition_model.limit_dist, n), rng_this_epoch
    )
    # g = gd.create_dense(
    #     nodes=one_hot.nodes,
    #     edges=one_hot.edges,
    #     nodes_mask=one_hot.nodes_mask,
    #     edges_mask=one_hot.edges_mask,
    # )
    g = one_hot
    # ipdb.set_trace()
    for t in tqdm(list(reversed(range(1, transition_model.diffusion_steps)))):
        rng = jax.random.fold_in(rng, enc(f"sample_{t}"))
        t = np.array(t).repeat(n)
        g = df.sample_p_zs_given_zt(
            p=model,
            t=t,
            g_t=g,
            transition_model=transition_model,
            rng=rng,
        )
        # g = _sample_step(g, transition_model, model, t[0], rng)
    return g


def to_one_hot(x):
    dense_data, mask = torch_utils.to_dense(x.x, x.edge_index, x.edge_attr, x.batch)
    nodes = np.asarray(dense_data.X.numpy())
    edges = np.asarray(dense_data.E.numpy())
    nodes_mask = np.asarray(mask.numpy())

    return gd.create_one_hot_minimal(
        nodes=nodes,
        edges=edges,
        nodes_mask=nodes_mask,
    )


DataLoader = Iterable[gd.OneHotGraph]


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
    do_restart: bool = False
    plot_every_steps: int = 1
    temporal_embedding_dim: int = 128
    n_val_steps: int = 60
    grad_clip: float = -1.0
    weight_decay: float = 1e-12
    min_learning_rate: float = 1e-6

    def __post_init__(
        self,
    ):
        self.n = 9  # FIXME: get it from the data
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=self.params,
            tx=self.__get_optimizer(),
            key=self.rngs["dropout"],
            last_loss=1000000,
            lr=self.learning_rate,
        )

        self.transition_model = TransitionModel.create(
            schedule_type=self.noise_schedule_type,
            x_priors=self.nodes_prior,
            e_priors=self.edges_prior,
            diffusion_steps=self.diffusion_steps,
            temporal_embedding_dim=self.temporal_embedding_dim,
            n=self.n,
        )
        self.plot_path = os.path.join(self.save_path, "plots")

        orbax_checkpointer = checkpoint.PyTreeCheckpointer()
        options = checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
        self.checkpoint_manager = checkpoint.CheckpointManager(
            self.save_path, orbax_checkpointer, options
        )
        self.a_val_batch = None

    @typed
    def __val_epoch(
        self,
        *,
        rng: Key,
    ) -> tuple[dict[str, Float[Array, ""]], float]:
        run_losses: list[dict[str, Float[Array, "batch_size"]]] = []
        t0 = time()
        for i, x in enumerate(tqdm(self.val_loader)):
            step_rng = jax.random.fold_in(rng, enc(f"val_step_{i}"))
            graph_dist = to_one_hot(x)  # type: ignore
            # if i == 0:
            #     print(f"val={graph_dist.nodes[0]}")
            self.a_val_batch = graph_dist
            losses = val_step(
                data=graph_dist,  # , mask.astype(bool)),
                state=self.state,
                rng=step_rng,
                transition_model=self.transition_model,
                nodes_dist=self.nodes_dist,
            )
            run_losses.append(losses)
        t1 = time()
        total_time = t1 - t0
        avg_losses = {
            k: np.concatenate(
                [(r[k] if len(r[k].shape) > 0 else r[k][None]) for r in run_losses]
            ).mean()
            for k in run_losses[0].keys()
            if k != "kl_prior"
        }
        # avg_loss = np.mean(np.array(run_loss)).tolist()
        # ipdb.set_trace()
        return avg_losses, total_time

    @typed
    def plot_preds(
        self,
        plot_path: str | None = None,
        load_from_disk: bool = True,
        epoch: int = -1,
    ):
        import random

        rng = jax.random.PRNGKey(random.randint(0, 100000))
        if load_from_disk:
            self.__restore_checkpoint()
        model = GetProbabilityFromState(self.state, rng, self.transition_model)
        print(f"Plotting noised graphs to {plot_path}")
        for i, x in enumerate(tqdm(self.val_loader)):
            g_batch = to_one_hot(x)
            break

        mean_loss, _ = self.__val_epoch(rng=jax.random.fold_in(rng, enc("val")))
        print(f"{mean_loss=}")
        g = gd.repeat(g_batch[np.array([0])], len(self.transition_model.q_bars))
        q_bars = self.transition_model.q_bars  # [timesteps]
        posterior_samples = gd.sample_one_hot(gd.matmul(g, q_bars), rng)
        timesteps = np.arange(len(self.transition_model.q_bars))
        model_probs = model(posterior_samples, timesteps)
        val_losses = df.compute_val_loss(
            target=posterior_samples,
            transition_model=self.transition_model,
            p=model,
            nodes_dist=self.nodes_dist,
            rng_key=jax.random.PRNGKey(random.randint(0, 100000)),
        )["nll"]
        corr = np.corrcoef(val_losses, np.arange(len(self.transition_model.q_bars)))[
            0, 1
        ]
        # ipdb.set_trace()
        # corr = 0.5
        if epoch > -1:
            wandb.log({"corr_t_vs_elbo": corr}, step=epoch)
        model_samples = gd.argmax(model_probs)
        gd.plot(
            [posterior_samples, model_samples],
            location=plot_path,
            shared_position="all",
            title=f"{round(mean_loss['nll'].tolist(), 2)} Correlation: {corr:.3f}",
        )

    def __restore_checkpoint(self):
        if self.checkpoint_manager.latest_step() is not None:
            state_dict = self.checkpoint_manager.restore(
                self.checkpoint_manager.latest_step()
            )
            self.learning_rate = state_dict["lr"]
            # ipdb.set_trace()
            self.state = TrainState(
                tx=self.__get_optimizer(),
                apply_fn=self.state.apply_fn,
                **{k: v for k, v in state_dict.items()},
            )
            print(
                f"[yellow]Restored from epoch {self.checkpoint_manager.latest_step()}[/yellow]"
            )
        else:
            print("[yellow]No checkpoint found, starting from scratch[/yellow]")

    def trivial_test(self):
        run_losses: list[dict[str, Float[Array, "batch_size"]]] = []
        t0 = time()
        rng = jax.random.PRNGKey(0)
        from flax.struct import dataclass

        @dataclass
        class Dummy:
            def __call__(self, x, t):
                return gd.create_dense(
                    nodes=x.nodes,
                    edges=x.edges,
                    nodes_mask=x.nodes_mask,
                    edges_mask=x.edges_mask,
                )

        identity = Dummy()
        for i in tqdm(range(20 * self.n_val_steps)):
            graph_dist = self.a_val_batch  # type: ignore
            losses = df.compute_val_loss(
                target=graph_dist,
                transition_model=self.transition_model,
                rng_key=rng,
                get_probability=identity,
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
        print(avg_losses)

    # @typed
    def restart(self):
        self.do_restart = True
        self.train()

    def __get_optimizer(self):
        components = []
        if self.grad_clip > 0:
            components.append(optax.clip_by_global_norm(10.0))
        # components.append(
        #     optax.adamw(
        #         learning_rate=self.learning_rate, weight_decay=self.weight_decay
        #     )
        # )
        components.extend(
            [
                optax.amsgrad(learning_rate=self.learning_rate),
                optax.add_decayed_weights(self.weight_decay),
            ]
        )

        return optax.chain(*components) if len(components) > 1 else components[0]

    @typed
    def train(
        self,
    ):
        os.system(f"rm -rf {self.plot_path}")
        os.makedirs(self.plot_path, exist_ok=True)
        last_epoch = 0
        if self.do_restart:
            self.__restore_checkpoint()
            last_epoch = self.checkpoint_manager.latest_step()
            print("Restarting training")

        assert last_epoch is not None
        assert last_epoch <= self.num_epochs, "Already trained for this many epochs."

        val_losses = []
        train_losses = []
        patience = 5
        current_patience = patience
        stopping_criterion = -5
        rng, _ = jax.random.split(self.rngs["params"])
        rng, rng_this_epoch = jax.random.split(rng)  # TODO use that

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
        eval_every = 4
        for epoch_idx in range(last_epoch, self.num_epochs + 1):
            val_rng_epoch = jax.random.fold_in(rng, enc(f"val_rng_{epoch_idx}"))
            train_rng_epoch = jax.random.fold_in(rng, enc(f"train_rng_{epoch_idx}"))
            print(f"[green bold]Epoch[/green bold]: {epoch_idx}")
            train_loss, train_time = self.__train_epoch(
                key=train_rng_epoch,
            )
            train_losses.append(train_loss)

            wandb.log({"train_loss": train_loss, "val_loss": val_loss}, epoch_idx)
            print(
                f"[red underline]Train[/red underline]\nloss={train_loss:.5f} best={min(train_losses):.5f} time={train_time:.4f}"
            )
            if epoch_idx % eval_every == 0:
                print("[green underline]Validating[/green underline]")
                val_loss, val_time = self.__val_epoch(
                    rng=val_rng_epoch,
                )
                val_losses.append(val_loss)
                print(
                    f"""
                    current={prettify(val_loss)}
                    best={prettify(min(val_losses, key=lambda x: x['nll']))}
                    time={val_time:.4f}"""
                )
                avg_loss = val_losses[-1]["nll"]
                if self.state.last_loss < avg_loss:
                    if current_patience <= 0:
                        new_lr = self.state.lr * 0.1
                        if self.learning_rate >= self.min_learning_rate:
                            self.learning_rate = new_lr
                            print(
                                f"[red] learning rate did not decrease. Reducing lr to {self.learning_rate} [/red]"
                            )
                            self.state = self.state.replace(
                                lr=new_lr, tx=self.__get_optimizer(), last_loss=avg_loss
                            )
                            current_patience = patience
                    else:
                        current_patience -= 1
                        if current_patience <= stopping_criterion:
                            print(
                                f"[red] stopping criterion reached. Stopping training [/red]"
                            )
                            break

                print(f"{current_patience=}")
                if avg_loss < self.state.last_loss:
                    current_patience = patience
                    self.state = self.state.replace(last_loss=avg_loss)

                    print(f"[yellow] Saving checkpoint[/yellow]")
                    self.checkpoint_manager.save(epoch_idx, self.state)
                    print(
                        f"[yellow] Saved to {os.path.join(str(self.checkpoint_manager.directory), str(self.checkpoint_manager.latest_step()))} [/yellow]"
                    )
                    if epoch_idx % self.plot_every_steps == 0:
                        # self.__plot_targets(
                        #     save_to=os.path.join(
                        #         self.plot_path, f"{epoch_idx}_targets.png"
                        #     )
                        # )
                        # self.plot_preds(
                        #     plot_path=os.path.join(
                        #         self.plot_path, f"{epoch_idx}_preds.png"
                        #     ),
                        #     load_from_disk=False,
                        # )
                        # self.sample(
                        #     restore_checkpoint=False,
                        #     save_to=os.path.join(
                        #         self.plot_path, f"{epoch_idx}_samples.png"
                        #     ),
                        # )
                        pass

        rng, _ = jax.random.split(self.rngs["params"])
        val_loss, val_time = self.__val_epoch(
            rng=rng,
            exhaustive=True,
        )
        print(
            f"Final loss: {val_loss:.4f} best={min(val_losses):.5f} time: {val_time:.4f}"
        )

    def __plot_targets(self, save_to: str | None = None):
        batch = self.a_val_batch
        import random

        rng = jax.random.PRNGKey(random.randint(0, 1000000))
        model = GetProbabilityFromState(self.state, rng, self.transition_model)
        t = np.array([400] * batch.nodes.shape[0])
        q_bars = self.transition_model.q_bars[t]
        noisy_batch = gd.sample_one_hot(gd.matmul(batch, q_bars), rng)
        preds = gd.softmax(model(batch, t))
        edges_flat = einop(
            batch.edges,
            "b n1 n2 ne -> (b n1 n2) ne",
        )
        pred_edges_flat = einop(
            preds.edges,
            "b n1 n2 ne -> (b n1 n2) ne",
        )
        noisy_edges_flat = einop(
            noisy_batch.edges,
            "b n1 n2 ne -> (b n1 n2) ne",
        )

        import matplotlib.pyplot as plt

        indices = np.where(
            np.argmax(edges_flat, axis=-1) != np.argmax(noisy_edges_flat, axis=-1)
        )[0][:10]
        edges_flat = edges_flat[indices]
        pred_edges_flat = pred_edges_flat[indices]
        noisy_edges_flat = noisy_edges_flat[indices]
        fig, axs = plt.subplots(10, 1, figsize=(15, 5))
        for i in range(10):
            x = np.arange(pred_edges_flat.shape[-1])
            axs[i].hlines(
                pred_edges_flat[i],
                x - 0.2,
                x + 0.2,
                label="prediction",
                color="blue",
                linewidth=2,
            )
            argmax = np.argmax(pred_edges_flat[i])
            axs[i].hlines(
                pred_edges_flat[i][argmax],
                x[argmax] - 0.1,
                x[argmax] + 0.1,
                color="orange",
                linewidth=2,
            )
            # plots a vertical line at the position of the target (where the target is 1)
            axs[i].axvline(np.argmax(edges_flat[i]), color="green", label="target")
            # does the same for the noisy target
            axs[i].axvline(
                np.argmax(noisy_edges_flat[i]), color="red", label="noisy target"
            )
            axs[i].set_xticks(np.arange(edges_flat.shape[-1]))
            axs[i].set_ylim(0, 1)
            axs[i].grid()

        plt.legend()

        if save_to is not None:
            plt.savefig(save_to, dpi=500)
        else:
            plt.show()

    def sample_steps(self, restore_checkpoint: bool = True, save_to: str | None = None):
        if restore_checkpoint:
            self.__restore_checkpoint()
        import random

        rng = jax.random.PRNGKey(random.randint(0, 1000000))
        model = GetProbabilityFromState(self.state, rng, self.transition_model)
        model_samples = _sample_steps(self.transition_model, model, rng)
        gd.plot([model_samples], shared_position="row", location=save_to)

    def sample(
        self, restore_checkpoint: bool = True, save_to: str | None = None, n: int = 10
    ):
        print(f"Saving samples to: {save_to}")
        if restore_checkpoint:
            self.__restore_checkpoint()
        import random

        rng_model = jax.random.PRNGKey(random.randint(0, 1000000))
        rng_sample = jax.random.fold_in(rng_model, enc("sample"))
        model = GetProbabilityFromState(self.state, rng_model, self.transition_model)
        model_samples = _sample(self.transition_model, model, rng_sample, n)

        data_sample = None
        for i, x in enumerate(tqdm(self.val_loader)):
            data_sample = to_one_hot(x)[:n]
            break
        assert data_sample is not None
        prior_sample = gd.sample_one_hot(
            gd.repeat_dense(self.transition_model.limit_dist, n), rng_sample
        )
        gd.plot(
            [data_sample, prior_sample, model_samples],  # prior_sample,
            shared_position=None,
            location=save_to,
        )

    @typed
    def __train_epoch(
        self,
        *,
        key: Key,
    ):
        run_losses = []
        t0 = time()
        print(f"[pink] LR={round(np.array(self.state.lr).tolist(), 7)} [/pink]")
        for i, x in enumerate(tqdm(self.train_loader)):
            step_key = jax.random.fold_in(key, enc(f"train_step_{i}"))
            graph_dist = to_one_hot(x)
            # if i == 0:
            #     print(f"train={graph_dist.nodes[0]}")
            state, loss = train_step(
                g=graph_dist,
                state=self.state,
                rng=step_key,
                transition_model=self.transition_model,
                nodes_dist=self.nodes_dist,
            )
            self.state = state
            run_losses.append(loss)
        avg_loss = np.mean(np.array(run_losses)).tolist()
        t1 = time()
        tot_time = t1 - t0
        return avg_loss, tot_time


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


def save_stuff():
    pass
