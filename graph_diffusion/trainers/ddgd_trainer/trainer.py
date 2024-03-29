import os
from . import torch_utils
import ipdb
from jax import numpy as np, Array
from flax.training import train_state
import matplotlib.pyplot as plt
import flax.linen as nn
from jaxtyping import Float, Int, Bool, PRNGKeyArray
from time import time
import numpy as nnp
from flax.struct import dataclass as flax_dataclass
from functools import partial
import jax
from jax import jit
import optax

# import wandb
from collections.abc import Iterable
from dataclasses import dataclass
import enum
from flax import jax_utils
from .metrics import SamplingMolecularMetrics
from rich import print as print
from tqdm import tqdm
from flax.core.frozen_dict import FrozenDict
from mate.jax import SFloat, SInt, SBool, Key

# from orbax import checkpoint
from einops import rearrange, reduce, repeat
import hashlib
from beartype import beartype
from jax import pmap
import random as pyrandom
from .extra_features.extra_features import compute as compute_extra_features
from .checkpoint_manager import CheckpointManager
from ...shared.graph import graph_distribution as gd
from ...models.ddgd import DDGD, SimpleDDGD, StructureFirstDDGD

# from . import diffusion_functions as df
# from .transition_model import (
#     TransitionModel,
# )
from functools import partial


# creates a fake pmap, useful for debugggin. Instead of running for every device,
# it has a for loop that runs for every device
def fake_pmap(f):
    def faked(*args):
        return np.concatenate(
            [f(*(a[i] for a in args)) for i in args[0].shape[0]], axis=0
        )

    return faked


def enc(x: str):
    return int(hashlib.md5(x.encode()).hexdigest()[:8], base=16)


DataLoader = Iterable[gd.OneHotGraph]
GraphDistribution = gd.GraphDistribution

from typing import no_type_check


@no_type_check
class TrainState(train_state.TrainState):
    key: Key


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
    return grads, loss


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
    return single_train_step(
        g=graph,
        state=state,
        rng=rng,
    )


def to_one_hot(nodes, edges, _, nodes_counts, device=jax.devices("cpu")[0]):
    return gd.OneHotGraph.create_from_counts(
        nodes=np.array(nodes),
        edges=np.array(edges),
        nodes_counts=np.array(nodes_counts),
    )


def shard_graph(g: gd.GraphDistribution):
    num_devices = jax.device_count()

    # Function to trim data such that its length becomes divisible by num_devices
    def trim_to_divisible(x):
        remainder = len(x) % num_devices
        if remainder != 0:
            return x[:-remainder]
        return x

    data = (g.nodes, g.edges, g.nodes_mask, g.edges_mask)
    trimmed_data = tuple(trim_to_divisible(x) for x in data)
    return tuple(x.reshape((num_devices, -1) + x.shape[1:]) for x in trimmed_data)


@dataclass
class Trainer:
    class DiffusionType(enum.Enum):
        structure_first = "structure-first"
        structure_only = "structure-only"
        feature_only = "feature-only"
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
        # orbax_checkpointer = checkpoint.PyTreeCheckpointer()
        # options = checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
        # self.checkpoint_manager = checkpoint.CheckpointManager(
        #     self.save_path, orbax_checkpointer, options
        # )
        self.checkpoint_manager = CheckpointManager(self.save_path)

        self.a_val_batch = None
        if self.train_smiles is not None and self.dataset_infos is not None:
            self.sampling_metric = SamplingMolecularMetrics(
                self.dataset_infos,
                self.train_smiles,
            )
        else:
            self.sampling_metric = None
        raw_dummy = to_one_hot(*next(iter(self.val_loader)))
        self.n = raw_dummy.nodes.shape[1]
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
        elif self.diffusion_type in (
            self.__class__.DiffusionType.structure_first,
            self.__class__.DiffusionType.structure_only,
            self.__class__.DiffusionType.feature_only,
        ):
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
                use_structure=self.diffusion_type
                in (
                    self.__class__.DiffusionType.structure_only,
                    self.__class__.DiffusionType.structure_first,
                ),
                use_feature=self.diffusion_type
                in (
                    self.__class__.DiffusionType.feature_only,
                    self.__class__.DiffusionType.structure_first,
                ),
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
        )

        @jit
        def val_step(data, val_rng, params):
            return self.ddgd.apply(
                params,
                data,
                rng_key=val_rng,
                method=self.ddgd.compute_val_loss,
            )

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

        self.parallel_val_step = pmap(parallel_val_step)
        self.parallel_train_step = pmap(parallel_train_step)
        self.n_devices = jax.local_device_count()
        self.cpu_device = jax.devices("cpu")[0]
        self.best_validity = {}

    def __val_epoch(
        self,
        *,
        rng: Key,
    ):  # -> tuple[dict[str, Float[Array, ""]], float]:
        # run_losses: list[dict[str, Float[Array, "batch_size"]]] = []
        t0 = time()
        losses = None
        for i, x in enumerate(tqdm(self.val_loader)):
            step_rng = jax.random.fold_in(rng, enc(f"val_step_{i}"))
            one_hot_graph = to_one_hot(*x)
            (
                sharded_nodes,
                sharded_edges,
                sharded_nodes_mask,
                sharded_edges_mask,
            ) = shard_graph(one_hot_graph)
            new_bs = sharded_nodes.shape[0] * sharded_nodes.shape[1]
            one_hot_graph = one_hot_graph[
                :new_bs
            ]  # the new exact batch size depends on the number of devices
            new_loss = self.parallel_val_step(
                sharded_nodes,
                sharded_edges,
                sharded_nodes_mask,
                sharded_edges_mask,
                params=jax_utils.replicate(self.state.params),
                val_rng=jax.random.split(step_rng, self.n_devices),
            ).unshard()
            if losses is None:
                losses = new_loss
            else:
                losses += new_loss
            if self.shuffle_coding_metric:
                losses = losses.convert_to_shuffle_conding_metric(one_hot_graph)
        assert losses is not None
        t1 = time()
        total_time = t1 - t0
        avg_losses = losses.mean()
        return avg_losses, total_time

    def __sample_structure(
        self,
        *,
        restore_checkpoint: bool = True,
    ):
        rng = jax.random.fold_in(self.rngs["params"], enc("sample_structure"))
        n_samples = 9
        if restore_checkpoint:
            self.__restore_checkpoint()
        result = self.ddgd.sample_structure(
            self.state.params,
            rng,
            n_samples,
        )
        return result

    def __sample_feature(
        self,
        *,
        restore_checkpoint: bool = True,
        g: gd.GraphDistribution | None = None,
    ):
        rng = jax.random.fold_in(self.rngs["params"], enc("sample_structure"))
        n_samples = 5
        if restore_checkpoint:
            self.__restore_checkpoint()

        if g is None:
            # extracts a batch from the val set
            g = to_one_hot(*next(iter(self.val_loader)))[: np.array(n_samples)]
        structure = g.structure_one_hot()
        result = self.ddgd.sample_feature(
            params=self.state.params,
            sampled_structure=structure,
            rng=rng,
            n_samples=n_samples,
        )
        return g, result

    def __sample_structure_and_plot(
        self,
        *,
        restore_checkpoint: bool = True,
        location: str | None = None,
    ):
        result = self.__sample_structure(restore_checkpoint=restore_checkpoint)
        gd.plot(result, location=location, shared_position_option="col")

    def __sample_feature_and_plot(
        self,
        *,
        restore_checkpoint: bool = True,
        location: str | None = None,
    ):
        result = self.__sample_feature(restore_checkpoint=restore_checkpoint)
        gd.plot(
            result,
            location=location,
            shared_position_option="col",
        )

    def __sample(
        self,
        *,
        restore_checkpoint: bool = True,
    ):  # -> tuple[gd.GraphDistribution, gd.GraphDistribution]:
        rng = jax.random.fold_in(self.rngs["params"], enc("sample"))
        n_samples = 100
        if restore_checkpoint:
            self.__restore_checkpoint()

        result = self.ddgd.sample(
            self.state.params,
            rng,
            n_samples,
        )
        return result

    def __sample_and_plot(
        self,
        *,
        restore_checkpoint: bool = True,
        location: str | None = None,
        compute_metrics: bool = True,
    ):
        sample, _ = self.__sample(restore_checkpoint=restore_checkpoint)
        if compute_metrics:
            self.__print_metrics(sample, title="Sample")
        gd.plot(
            sample,
            location=location,
        )

    def __train_epoch(
        self,
        *,
        key: Key,
        epoch: int,
    ) -> tuple[float, float]:
        run_losses = []
        t0 = time()
        # print(f"[pink] LR={round(np.array(self.state.lr).tolist(), 7)} [/pink]")
        use_structure_this_epoch = epoch < 30
        use_stucture = jax_utils.replicate(use_structure_this_epoch)
        use_feature = np.logical_not(use_stucture)
        for i, x in enumerate(tqdm(self.train_loader)):
            one_hot_graph = to_one_hot(*x)
            step_key = jax.random.fold_in(key, enc(f"train_step_{i}"))
            (
                sharded_nodes,
                sharded_edges,
                sharded_nodes_mask,
                sharded_edges_mask,
            ) = shard_graph(one_hot_graph)

            result = self.parallel_train_step(
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
        return avg_loss, tot_time  # type: ignore

    # def predict_structure(
    #     self,
    #     title: str = "",
    #     restore_checkpoint: bool = True,
    #     location: str | None = None,
    # ):
    #     if restore_checkpoint:
    #         self.__restore_checkpoint()
    #     data = to_one_hot(*next(iter(self.val_loader)))
    #     data = data[:5]
    #     t_0 = np.ones(data.batch_size, int)
    #     rng_0 = jax.random.PRNGKey(pyrandom.randint(0, 100000))
    #     g_t_0, g_pred_0 = self.ddgd.apply(
    #         self.params,
    #         data,
    #         t_0,
    #         rng_0,
    #         method=self.ddgd.predict_structure,
    #     )
    #     t_mid = np.ones(data.batch_size, int) * 20
    #     rng_mid = jax.random.PRNGKey(pyrandom.randint(0, 100000))
    #     g_t_mid, g_pred_mid = self.ddgd.apply(
    #         self.params,
    #         data,
    #         t_mid,
    #         rng_mid,
    #         method=self.ddgd.predict_structure,
    #     )
    #     gd.plot(
    #         [data, g_pred_0, g_t_0, g_pred_mid, g_t_mid],
    #         title=title,
    #         shared_position_option="col",
    #         location=location,
    #     )

    # def predict_feature(
    #     self,
    #     title: str = "",
    #     restore_checkpoint: bool = True,
    #     location: str | None = None,
    # ):
    #     if restore_checkpoint:
    #         self.__restore_checkpoint()
    #     data = to_one_hot(*next(iter(self.val_loader)))
    #     data = data[:9]
    #     t_0 = np.ones(data.batch_size, int)
    #     rng_0 = jax.random.PRNGKey(pyrandom.randint(0, 100000))
    #     g_t_0, g_pred_0 = self.ddgd.apply(
    #         self.params,
    #         data,
    #         t_0,
    #         rng_0,
    #         method=self.ddgd.predict_feature,
    #     )
    #     t_mid = np.ones(data.batch_size, int) * 20
    #     rng_mid = jax.random.PRNGKey(pyrandom.randint(0, 100000))
    #     g_t_mid, g_pred_mid = self.ddgd.apply(
    #         self.params,
    #         data,
    #         t_mid,
    #         rng_mid,
    #         method=self.ddgd.predict_feature,
    #     )
    #     gd.plot(
    #         [data, g_pred_0, g_t_0, g_pred_mid, g_t_mid],
    #         title=title,
    #         shared_position_option="col",
    #         location=location,
    #     )

    def predict(
        self,
        title: str = "",
        restore_checkpoint: bool = True,
        location: str | None = None,
    ):
        if self.diffusion_type in (
            self.__class__.DiffusionType.structure_first,
            self.__class__.DiffusionType.structure_only,
        ):
            self.predict_structure(
                title=title, restore_checkpoint=restore_checkpoint, location=location
            )
        elif self.diffusion_type in (
            self.__class__.DiffusionType.structure_first,
            self.__class__.DiffusionType.structure_only,
        ):
            self.predict_feature(
                title=title, restore_checkpoint=restore_checkpoint, location=location
            )

    def plot_preds(
        self,
        plot_path: str | None = None,
        load_from_disk: bool = True,
        epoch: int = -1,
    ):
        from scipy.stats import spearmanr

        rng = jax.random.PRNGKey(pyrandom.randint(0, 100000))
        n_samples = 4
        g_batch = to_one_hot(*next(iter(self.val_loader)))[:n_samples]
        correlations = []
        for s in range(n_samples):
            rng_s = jax.random.fold_in(rng, s)
            g_t, preds, losses = self.ddgd.predict_for_all_timesteps(
                self.state.params,
                g_batch[np.array([s])],
                rng_s,
            )
            preds = gd.sample_one_hot(
                gd.softmax(preds), jax.random.fold_in(rng_s, enc("sample"))
            )
            losses = losses["nll"]

            correlation, p_value = spearmanr(np.arange(len(losses)), losses)
            correlations.append(correlation)
            current_plot_path = plot_path + f"_{s}" if plot_path else None
            print(f"Plotting preds to: {current_plot_path}")
            gd.plot(
                [g_t, preds],
                location=current_plot_path,
                shared_position_option="all",
                # title=f"{round(mean_loss['nll'].tolist(), 2)} Correlation: {corr:.3f}",
            )
        correlation = np.mean(np.array(correlations))
        print(f"{correlation=}")
        # if load_from_disk:
        #     self.__restore_checkpoint()
        # p = GetProbabilityFromState(
        #     self.state,
        #     rng,
        #     self.transition_model,
        #     extra_features=self.use_extra_features,
        # )
        # print(f"Plotting noised graphs to {plot_path}")
        # g_batch = None
        # for i, x in enumerate(tqdm(self.val_loader)):
        #     g_batch = to_one_hot(x)
        #     break
        # assert not g_batch is None
        # mean_loss, _ = self.__val_epoch(rng=jax.random.fold_in(rng, enc("val")))
        # g = g_batch[np.array([0])].repeat(len(self.transition_model.q_bars))
        # q_bars = self.transition_model.q_bars  # [timesteps]
        # posterior_samples = gd.sample_one_hot(gd.matmul(g, q_bars), rng)
        # timesteps = np.arange(len(self.transition_model.q_bars))
        # model_probs = p(posterior_samples, timesteps)
        # val_losses = df.compute_val_loss(
        #     target=posterior_samples,
        #     transition_model=self.transition_model,
        #     p=p,
        #     nodes_dist=self.nodes_dist,
        #     rng_key=jax.random.PRNGKey(random.randint(0, 100000)),
        # )["nll"]
        # corr = np.corrcoef(val_losses, np.arange(len(self.transition_model.q_bars)))[
        #     0, 1
        # ]
        # # FIXME
        # # if epoch > -1:
        # #     wandb.log({"corr_t_vs_elbo": corr}, step=epoch)
        # model_samples = model_probs.argmax()

    def __restore_checkpoint(self):
        # FIXME
        # if self.checkpoint_manager.latest_step() is not None:
        #     state_dict = self.checkpoint_manager.restore(
        #         self.checkpoint_manager.latest_step()
        #     )
        #     self.learning_rate = state_dict["lr"]
        #     state_dict = jax.tree_map(np.asarray, state_dict)
        #     state_dict["step"] = state_dict["step"].item()
        #     self.state = TrainState(
        #         tx=self.__get_optimizer(),
        #         apply_fn=self.state.apply_fn,
        #         **{
        #             k: v if not isinstance(v, dict) else FrozenDict(v)
        #             for k, v in state_dict.items()
        #             if not k in ("last_loss", "lr")
        #         },
        #     )
        #     print(
        #         f"[yellow]Restored from epoch {self.checkpoint_manager.latest_step()}[/yellow]"
        #     )
        # else:
        #     print("[yellow]No checkpoint found, starting from scratch[/yellow]")
        pass

    def trivial_test(self):
        """
        Useful for debugging.
        """
        run_losses: list[dict[str, Float[Array, "batch_size"]]] = []
        t0 = time()
        rng = jax.random.PRNGKey(0)

        @dataclass
        class Dummy:
            def __call__(self, x, t):  # dfsewfrd
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
        avg_losses = {
            k: np.mean(np.array([r[k] for r in run_losses]))
            for k in run_losses[0].keys()
        }
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
        self.plot_preds(
            plot_path=os.path.join(self.plot_path, "initial_preds_all_t"),
            load_from_disk=False,
        )
        self.sample_and_plot(
            restore_checkpoint=False,
            location=self.plot_path,
        )
        rng, _ = jax.random.split(self.rngs["params"])
        rng, rng_this_epoch = jax.random.split(rng)  # TODO use that
        val_loss, val_time = self.__val_epoch(
            rng=rng_this_epoch,
        )
        min_val_loss = val_loss
        min_train_loss = np.inf
        print(val_loss.to_rich_table(f"Val Loss (prior training)", epoch=0))
        eval_every = 1

        for epoch_idx in range(last_epoch, self.num_epochs + 1):
            val_rng_epoch = jax.random.fold_in(rng, enc(f"val_rng_{epoch_idx}"))
            train_rng_epoch = jax.random.fold_in(rng, enc(f"train_rng_{epoch_idx}"))
            print(f"[green bold]Epoch[/green bold]: {epoch_idx}")
            train_loss, train_time = self.__train_epoch(
                key=train_rng_epoch,
                epoch=epoch_idx,
            )
            if train_loss < min_train_loss:
                min_train_loss = train_loss
            print(
                f"[red underline]Train[/red underline]\nloss={train_loss:.5f} best={min_train_loss:.5f} time={train_time:.4f}"
            )
            if epoch_idx % eval_every == 0:
                print("[green underline]Validating[/green underline]")
                val_loss, val_time = self.__val_epoch(
                    rng=val_rng_epoch,
                )
                print(val_loss.to_rich_table(f"Cur Val Loss", epoch=0))
                print(min_val_loss.to_rich_table(f"Best Val Loss", epoch=0))
                print(f"[green]time took[/green]={val_time:.4f}")
                if val_loss < min_val_loss:
                    print(
                        f"[green] New best val loss: {val_loss.nll:.4f} < {min_val_loss.nll:.4f} [/green]"
                    )
                    print(f"[yellow] Saving checkpoint[/yellow]")
                    self.checkpoint_manager.save(epoch_idx, self.state)
                    # self.predict(
                    #     restore_checkpoint=False,
                    #     location="wandb",
                    #     title=f"val nll: {val_loss.nll:.4f}",
                    # )
                    self.plot_preds(
                        plot_path=os.path.join(
                            self.plot_path, f"preds_all_t_{epoch_idx}"
                        ),
                    )
                    self.sample_and_plot(
                        restore_checkpoint=False,
                        location=self.plot_path,
                    )
                    print(
                        f"[yellow] Saved to {os.path.join(str(self.checkpoint_manager.directory), str(self.checkpoint_manager.latest_step()))} [/yellow]"
                    )
                    min_val_loss = val_loss

        rng = jax.random.fold_in(self.rngs["params"], enc("final_val"))
        val_loss, val_time = self.__val_epoch(
            rng=rng,
        )
        print(
            f"Final loss: {val_loss:.4f} best={min(val_loss):.5f} time: {val_time:.4f}"
        )

    def sample_and_plot(
        self, restore_checkpoint: bool = True, location: str | None = None
    ):
        if self.diffusion_type == self.__class__.DiffusionType.structure_only:
            self.__sample_structure_and_plot(
                restore_checkpoint=restore_checkpoint,
                location=location,
            )
        elif self.diffusion_type == self.__class__.DiffusionType.structure_first:
            self.__sample_and_plot(
                restore_checkpoint=restore_checkpoint,
                location=location,
            )
        elif self.diffusion_type == self.__class__.DiffusionType.feature_only:
            self.__sample_feature_and_plot(
                restore_checkpoint=restore_checkpoint,
                location=location,
            )
        elif self.diffusion_type == self.__class__.DiffusionType.simple:
            self.__sample_and_plot(
                restore_checkpoint=restore_checkpoint,
                location=location,
            )

    def compute_metrics(self):
        assert self.sampling_metric is not None
        model_samples, _ = self.__sample(restore_checkpoint=True)
        self.__print_metrics(model_samples, "Sampled")

    def __print_metrics(self, sample: gd.GraphDistribution, title: str):
        # metrics = self.sampling_metric(sample)
        # wandb.log(metrics, step=self.state.step)
        # print(f"{title}: {metrics['relaxed_validity']:.3f} {metrics['uniqueness']:.3f}")
        assert self.sampling_metric is not None
        metrics = self.sampling_metric(sample)
        # print(metrics, f"step={self.state.step}")
        # print(
        #     f"Validity: {metrics['validity']}\n Relaxed validity: {metrics['relaxed_validity']:.3f}\nUniqueness: {metrics['uniqueness']:.3f}"
        # )
        if ("validity" not in self.best_validity) or (
            metrics["validity"] > self.best_validity["validity"]
        ):
            self.best_validity = metrics
        print("[red]Current[/red] [green]Validity[/green]")
        print(
            {
                k: float(f"{v:.4f}")
                for k, v in metrics.items()
                if k in ("validity", "relaxed_validity", "uniqueness")
            }
        )
        print("[red]Best[/red] [green]Validity[/green]")
        print(
            {
                k: float(f"{v:.4f}")
                for k, v in self.best_validity.items()
                if k in ("validity", "relaxed_validity", "uniqueness")
            }
        )


def prettify(val: dict[str, Float[Array, ""]]) -> dict[str, float]:
    return {k: float(f"{v.tolist():.4f}") for k, v in val.items()}


def save_stuff():
    pass
