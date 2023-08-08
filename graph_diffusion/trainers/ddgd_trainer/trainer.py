import os
from . import torch_utils
import ipdb
from jax import numpy as np, Array
from flax.training import train_state
import matplotlib.pyplot as plt
import flax.linen as nn
from jaxtyping import Float, Int, Bool
from time import time
from jaxtyping import jaxtyped
from flax.struct import dataclass as flax_dataclass
from functools import partial
import jax
from jax import jit
import optax
import wandb
from typing import Iterable
from dataclasses import dataclass
import enum
from flax import jax_utils
from .metrics import SamplingMolecularMetrics
from rich import print as print
from tqdm import tqdm
from flax.core.frozen_dict import FrozenDict
from mate.jax import SFloat, SInt, SBool, Key
from orbax import checkpoint
from einops import rearrange, reduce, repeat
import hashlib
from beartype import beartype
from jax import pmap
from .extra_features.extra_features import compute as compute_extra_features
from ...shared.graph import graph_distribution as gd
from ...models.ddgd import DDGD, SimpleDDGD, StructureFirstDDGD

# from . import diffusion_functions as df
# from .transition_model import (
#     TransitionModel,
# )
from functools import partial


def enc(x: str):
    return int(hashlib.md5(x.encode()).hexdigest()[:8], base=16)


DataLoader = Iterable[gd.OneHotGraph]
GraphDistribution = gd.GraphDistribution


class TrainState(train_state.TrainState):
    key: jax.random.PRNGKey
    lr: SFloat
    last_loss: SFloat


@jit
def single_train_step(
    *,
    g: gd.OneHotGraph,
    state: TrainState,
    rng: Key,
):
    train_rng = jax.random.fold_in(rng, enc("train"))
    dropout_rng = jax.random.fold_in(rng, enc("dropout"))

    def loss_fn(params: FrozenDict):
        loss = state.apply_fn(
            params,
            target=g,
            rng_key=train_rng,
            rngs={"dropout": dropout_rng},
        ).mean()
        return loss, None

    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = gradient_fn(state.params)
    # state = state.apply_gradients(grads=grads)
    return grads, loss


@pmap
def parallel_train_step(
    sharded_nodes,
    sharded_edges,
    sharded_nodes_mask,
    sharded_edges_mask,
    state: TrainState,
    rng: Key,
):
    graph = gd.OneHotGraph.create(
        nodes=sharded_nodes,
        edges=sharded_edges,
        nodes_mask=sharded_nodes_mask,
        edges_mask=sharded_edges_mask,
    )
    return single_train_step(g=graph, state=state, rng=rng)


def to_one_hot(nodes, edges, _, nodes_counts):
    return gd.OneHotGraph.create_from_counts(
        nodes=np.asarray(nodes),
        edges=np.asarray(edges),
        nodes_counts=np.asarray(nodes_counts),
    )


@jit
def convert_to_shuffle_conding_metric(
    val: Float[Array, "v"], target: GraphDistribution
):
    tot_edges_mask = target.edges.argmax(-1) > 0 & target.edges_mask
    edges_count = tot_edges_mask.sum()
    # converts val in log2
    uba = val / (edges_count * np.log(2))
    return uba


def shard_graph(g: gd.GraphDistribution):
    num_devices = jax.device_count()
    data = (g.nodes, g.edges, g.nodes_mask, g.edges_mask)
    return jax.tree_map(lambda x: x.reshape((num_devices, -1) + x.shape[1:]), data)


@dataclass
class Trainer:
    class DiffusionType(enum.Enum):
        structure_first = "structure-first"
        simple = "simple"

    model_class: type[nn.Module]
    train_loader: DataLoader
    val_loader: DataLoader
    num_epochs: int
    rngs: dict[str, Key]
    learning_rate: float
    save_path: str
    nodes_dist: Float[Array, "k"]
    feature_nodes_prior: Float[Array, "num_node_features"]
    feature_edges_prior: Float[Array, "num_edge_features"]
    bits_per_edge: bool
    diffusion_steps: int
    noise_schedule_type: str
    log_every_steps: int
    max_num_nodes: int
    num_edge_features: int
    num_node_features: int
    dataset_infos: object | None = None
    structure_nodes_prior: Float[Array, "2"] | None = None
    structure_edges_prior: Float[Array, "2"] | None = None
    train_smiles: Iterable[str] | None = None
    diffusion_type: DiffusionType = DiffusionType.simple
    match_edges: bool = True
    do_restart: bool = False
    plot_every_steps: int = 1
    temporal_embedding_dim: int = 128
    n_val_steps: int = 60
    grad_clip: float = -1.0
    weight_decay: float = 1e-12
    min_learning_rate: float = 1e-6
    use_extra_features: bool = False
    shuffle_coding_metric: bool = False
    n_layers: int = 5
    patience: int = 10

    def __post_init__(
        self,
    ):
        self.plot_path = os.path.join(self.save_path, "plots")
        orbax_checkpointer = checkpoint.PyTreeCheckpointer()
        options = checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
        self.checkpoint_manager = checkpoint.CheckpointManager(
            self.save_path, orbax_checkpointer, options
        )
        self.a_val_batch = None
        if self.train_smiles is not None and self.dataset_infos is not None:
            self.sampling_metric = SamplingMolecularMetrics(
                self.dataset_infos,
                self.train_smiles,
            )
        else:
            self.sampling_metric = None
        raw_dummy = to_one_hot(*next(iter(self.val_loader)))
        print(f"{raw_dummy.nodes.shape=}")
        self.n = raw_dummy.nodes.shape[1]
        # dummy, global_features = self.ddgd.get_model_input(
        #     raw_dummy,
        #     t=np.array([10]),
        # )
        # params = self.model.init(
        #     self.rngs["params"],
        #     dummy,
        #     global_features,
        #     deterministic=True,
        # )
        if self.diffusion_type == self.__class__.DiffusionType.simple:
            self.ddgd = SimpleDDGD(
                model_class=self.model_class,
                nodes_dist=self.nodes_dist,
                nodes_prior=self.feature_nodes_prior,
                edges_prior=self.feature_edges_prior,
                diffusion_steps=self.diffusion_steps,
                temporal_embedding_dim=self.temporal_embedding_dim,
                n=self.n,
                noise_schedule_type=self.noise_schedule_type,
                use_extra_features=self.use_extra_features,
                n_layers=self.n_layers,
            )
        elif self.diffusion_type == self.__class__.DiffusionType.structure_first:
            assert self.structure_nodes_prior is not None
            assert self.structure_edges_prior is not None
            self.ddgd = StructureFirstDDGD(
                model_class=self.model_class,
                nodes_dist=self.nodes_dist,
                feature_nodes_prior=self.feature_nodes_prior,
                feature_edges_prior=self.feature_edges_prior,
                structure_nodes_prior=self.structure_nodes_prior,
                structure_edges_prior=self.structure_edges_prior,
                structure_diffusion_steps=self.diffusion_steps // 2,
                feature_diffusion_steps=self.diffusion_steps // 2,
                temporal_embedding_dim=self.temporal_embedding_dim,
                n=self.n,
                noise_schedule_type=self.noise_schedule_type,
                use_extra_features=self.use_extra_features,
                n_layers=self.n_layers,
            )
        else:
            raise ValueError(f"Unknown diffusion type {self.diffusion_type}")

        rng_train = jax.random.fold_in(self.rngs["params"], enc("train"))
        self.params = self.ddgd.init(
            self.rngs,
            raw_dummy,
            rng_key=rng_train,
        )
        self.state = TrainState.create(
            apply_fn=self.ddgd.apply,
            params=FrozenDict(self.params),
            tx=self.__get_optimizer(),
            key=self.rngs["dropout"],
            last_loss=1000000,
            lr=self.learning_rate,
        )

        @jit
        def val_step(data, val_rng, params):
            return self.ddgd.apply(
                params,
                data,
                rng_key=val_rng,
                method=self.ddgd.compute_val_loss,
            )

        @pmap
        def parallel_val_step(
            sharded_nodes,
            sharded_edges,
            sharded_nodes_mask,
            sharded_edges_mask,
            val_rng,
            params,
        ):
            graph = gd.OneHotGraph.create(
                nodes=sharded_nodes,
                edges=sharded_edges,
                nodes_mask=sharded_nodes_mask,
                edges_mask=sharded_edges_mask,
            )
            return val_step(graph, val_rng, params)

        @jit
        def __sample_step(rng, t, g):
            return self.ddgd.apply(
                self.state.params, rng, t, g, method=self.ddgd.sample_step
            )

        def __sample(rng, n_samples):
            return self.ddgd.apply(
                self.state.params,
                __sample_step,
                rng,
                n_samples,
                method=self.ddgd.sample,
            )

        # self.val_step = val_step
        self.parallel_val_step = parallel_val_step
        self.__sample = __sample
        self.n_devices = jax.local_device_count()

    def __val_epoch(
        self,
        *,
        rng: Key,
    ) -> tuple[dict[str, Float[Array, ""]], float]:
        run_losses: list[dict[str, Float[Array, "batch_size"]]] = []
        t0 = time()
        for i, x in enumerate(tqdm(self.val_loader)):
            step_rng = jax.random.fold_in(rng, enc(f"val_step_{i}"))
            one_hot_graph = to_one_hot(*x)
            (
                sharded_nodes,
                sharded_edges,
                sharded_nodes_mask,
                sharded_edges_mask,
            ) = shard_graph(one_hot_graph)
            losses = jax.tree_map(
                lambda x: x.reshape(-1),
                self.parallel_val_step(
                    sharded_nodes,
                    sharded_edges,
                    sharded_nodes_mask,
                    sharded_edges_mask,
                    params=jax_utils.replicate(self.state.params),
                    val_rng=jax.random.split(step_rng, self.n_devices),
                ),
            )
            if self.shuffle_coding_metric:
                losses = {
                    k: convert_to_shuffle_conding_metric(v, one_hot_graph)
                    for k, v in losses.items()
                }
            run_losses.append(losses)
        assert len(run_losses) > 0
        t1 = time()
        total_time = t1 - t0
        avg_losses = {
            k: np.concatenate(
                [(r[k] if len(r[k].shape) > 0 else r[k][None]) for r in run_losses]
            ).mean()
            for k in run_losses[0].keys()
            if k != "kl_prior"
        }

        return avg_losses, total_time

    def __train_epoch(
        self,
        *,
        key: Key,
    ):
        run_losses = []
        t0 = time()
        print(f"[pink] LR={round(np.array(self.state.lr).tolist(), 7)} [/pink]")
        for i, x in enumerate(tqdm(self.train_loader)):
            one_hot_graph = to_one_hot(*x)
            step_key = jax.random.fold_in(key, enc(f"train_step_{i}"))
            structure_key = jax.random.fold_in(key, enc(f"train_step_{i}"))
            do_backprop_over_structure = (
                (
                    jax.random.uniform(
                        structure_key,
                    )
                    > 0.5
                ).item()
                if self.diffusion_type == self.__class__.DiffusionType.structure_first
                else False
            )
            (
                sharded_nodes,
                sharded_edges,
                sharded_nodes_mask,
                sharded_edges_mask,
            ) = shard_graph(one_hot_graph)
            result = parallel_train_step(
                sharded_nodes,
                sharded_edges,
                sharded_nodes_mask,
                sharded_edges_mask,
                state=jax_utils.replicate(self.state),
                rng=jax.random.split(step_key, self.n_devices),
            )
            grads, loss = jax.tree_map(lambda x: x.mean(0), result)
            self.state = self.state.apply_gradients(grads=grads)
            run_losses.append(loss)
        avg_loss = np.mean(np.array(run_losses)).tolist()
        t1 = time()
        tot_time = t1 - t0
        return avg_loss, tot_time

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
        p = GetProbabilityFromState(
            self.state,
            rng,
            self.transition_model,
            extra_features=self.use_extra_features,
        )
        print(f"Plotting noised graphs to {plot_path}")
        g_batch = None
        for i, x in enumerate(tqdm(self.val_loader)):
            g_batch = to_one_hot(x)
            break
        assert not g_batch is None
        mean_loss, _ = self.__val_epoch(rng=jax.random.fold_in(rng, enc("val")))
        g = g_batch[np.array([0])].repeat(len(self.transition_model.q_bars))
        q_bars = self.transition_model.q_bars  # [timesteps]
        posterior_samples = gd.sample_one_hot(gd.matmul(g, q_bars), rng)
        timesteps = np.arange(len(self.transition_model.q_bars))
        model_probs = p(posterior_samples, timesteps)
        val_losses = df.compute_val_loss(
            target=posterior_samples,
            transition_model=self.transition_model,
            p=p,
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
        model_samples = model_probs.argmax()
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
        """
        Useful for debugging.
        """
        run_losses: list[dict[str, Float[Array, "batch_size"]]] = []
        t0 = time()
        rng = jax.random.PRNGKey(0)

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

    def restart(self):
        self.do_restart = True
        self.train()

    def __get_optimizer(self):
        components = []
        if self.grad_clip > 0:
            components.append(optax.clip_by_global_norm(10.0))
        components.extend(
            [
                optax.amsgrad(learning_rate=self.learning_rate),
                optax.add_decayed_weights(self.weight_decay),
            ]
        )

        return optax.chain(*components) if len(components) > 1 else components[0]

    # @typed
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
        current_patience = self.patience
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
                            current_patience = self.patience
                    else:
                        current_patience -= 1
                        if current_patience <= stopping_criterion:
                            print(
                                f"[red] stopping criterion reached. Stopping training [/red]"
                            )
                            break

                print(f"{current_patience=}")
                if avg_loss < self.state.last_loss:
                    current_patience = self.patience
                    self.state = self.state.replace(last_loss=avg_loss)

                    print(f"[yellow] Saving checkpoint[/yellow]")
                    self.checkpoint_manager.save(epoch_idx, self.state)
                    print(
                        f"[yellow] Saved to {os.path.join(str(self.checkpoint_manager.directory), str(self.checkpoint_manager.latest_step()))} [/yellow]"
                    )
                    if avg_loss < min(val_losses, key=lambda x: x["nll"])["nll"]:
                        # self.plot_preds(
                        #     plot_path=os.path.join(
                        #         self.plot_path, f"{epoch_idx}_preds.png"
                        #     ),
                        #     load_from_disk=False,
                        # )
                        self.sample(restore_checkpoint=False, save_to="wandb")

        rng, _ = jax.random.split(self.rngs["params"])
        val_loss, val_time = self.__val_epoch(
            rng=rng,
        )
        print(
            f"Final loss: {val_loss:.4f} best={min(val_losses):.5f} time: {val_time:.4f}"
        )

    def __plot_targets(self, save_to: str | None = None):
        """
        This function was useful for debugging.
        """
        batch = self.a_val_batch
        import random

        rng = jax.random.PRNGKey(random.randint(0, 1000000))
        model = GetProbabilityFromState(
            self.state,
            rng,
            self.transition_model,
            extra_features=self.use_extra_features,
        )
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
        model = GetProbabilityFromState(
            self.state,
            rng,
            self.transition_model,
            extra_features=self.use_extra_features,
        )
        model_samples = _sample_steps(self.transition_model, model, rng)
        gd.plot([model_samples], shared_position="row", location=save_to)

    def sample(
        self, restore_checkpoint: bool = True, save_to: str = "wandb", n: int = 500
    ):
        print(f"Saving samples to: {save_to}")
        if restore_checkpoint:
            self.__restore_checkpoint()
        import random

        rng_model = jax.random.PRNGKey(random.randint(0, 1000000))
        rng_sample = jax.random.fold_in(rng_model, enc("sample"))

        model_samples, prior_sample = self.__sample(rng_sample, n)
        data_sample = None
        rnd_i = random.randint(0, len(self.val_loader) - 1)
        for i, x in enumerate(self.val_loader):
            if i == rnd_i:
                data_sample = to_one_hot(*x)[:n]
                break
        assert data_sample is not None
        # prior_sample = gd.sample_one_hot(
        #     self.transition_model.limit_dist.repeat(n), rng_sample
        # )
        title = "A Title"
        if self.sampling_metric is not None:
            metrics = self.sampling_metric(model_samples)
            print(
                f"Relaxed validity: {metrics['relaxed_validity']:.3f} {metrics['uniqueness']:.3f}"
            )
            title = f"{metrics['relaxed_validity']:.3f} {metrics['uniqueness']:.3f}"
        gd.plot(
            [
                data_sample,
                model_samples,
                prior_sample,
            ],  # prior_sample,  # prior_sample,
            shared_position=None,
            location=save_to,
            title=title,
        )


# # @typed
# def plot_noised_graphs(train_loader, transition_model: TransitionModel, save_path: str):
#     print(f"Plotting noised graphs to {save_path}")
#     T = len(transition_model.q_bars)
#     g_batch = next(iter(train_loader))
#     g = g_batch[np.array([0])].repeat(len(transition_model.q_bars))
#     rng = jax.random.PRNGKey(0)
#     q_bars = transition_model.q_bars  # [timesteps]
#     probs = (g @ q_bars).sample_one_hot(rng)
#     probs.plot()
#


def prettify(val: dict[str, Float[Array, ""]]) -> dict[str, float]:
    return {k: float(f"{v.tolist():.4f}") for k, v in val.items()}


def save_stuff():
    pass
