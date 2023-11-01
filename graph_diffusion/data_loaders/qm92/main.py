# import ipdb;ipdb.set_trace()
import os
from jax import jit
import random
from tqdm import tqdm
import pickle
import ipdb
from collections.abc import Iterable
from dataclasses import dataclass
from jaxtyping import Float, Array
from rich import print

from jax import numpy as np
from mate.jax import SFloat
from .qm9_dataset import QM9DataModule, QM9infos, get_train_smiles
from ...shared.graph import graph_distribution as gd, graph
from flax.struct import dataclass as flax_dataclass

GraphDistribution = gd.GraphDistribution


@dataclass(frozen=True)
class QM9Dataset:
    train_loader: Iterable
    test_loader: Iterable
    max_node_feature: int
    max_edge_feature: int
    n: int
    nodes_dist: Float[Array, "k"]
    feature_node_prior: Float[Array, "m"]
    feature_edge_prior: Float[Array, "l"]
    structure_node_prior: Float[Array, "k"]
    structure_edge_prior: Float[Array, "k"]
    infos: QM9infos
    train_smiles: Iterable[str]
    mean_edge_count: SFloat = 0.0
    mean_node_count: SFloat = 0.0
    var_edge_count: SFloat = 0.0
    var_node_count: SFloat = 0.0


class Bunch:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getitem__(self, key):
        return self.__dict__[key]

    @classmethod
    def from_dict(cls, dictionary):
        bunch = cls()
        for key, value in dictionary.items():
            if isinstance(value, dict):
                bunch.__dict__[key] = cls.from_dict(value)
            else:
                bunch.__dict__[key] = value
        return bunch


print("Converting to dense numpy")


from . import utils


# def to_dense_numpy(dataloader):
#     xs = []
#     es = []
#     nodes_counts = []
#     for data in tqdm(dataloader):
#         dense_data, node_mask = utils.to_dense(
#             data.x, data.edge_index, data.edge_attr, data.batch
#         )
#         dense_data = dense_data.mask(node_mask)
#         X, E = dense_data.X, dense_data.E
#         if X.shape[1] == 9:  # for some reason in one case there are only 8 nodes
#             xs.append(X)
#             es.append(E)
#             nodes_counts.append(node_mask.sum(-1))
#     xs = np.concatenate(xs, axis=0)
#     es = np.concatenate(es, axis=0)
#     # es[np.where(es.sum(-1) == 0)] = np.eye(es.shape[-1])[0]
#     es = np.where(es.sum(-1) == 0, np.eye(es.shape[-1])[0], es)
#     nodes_counts = np.concatenate(nodes_counts, axis=0)
#     return xs, es, nodes_counts

from jaxtyping import Int


@flax_dataclass
class DataItem:
    x: Array
    edge_index: Array
    edge_attr: Array
    batch: Array


def data_loader_to_data_items(dataloader):
    print("Converting to JAX data items")
    res = [
        DataItem(
            x=np.asarray(data.x),
            edge_index=np.asarray(data.edge_index),
            edge_attr=np.asarray(data.edge_attr),
            batch=np.asarray(data.batch),
        )
        for data in dataloader
    ]
    print("Done converting to JAX data items")
    return res


def to_dense_numpy(data_items: Iterable[DataItem]):
    xs = []
    es = []
    nodes_counts = []
    for data in tqdm(data_items):
        # dense_data, node_mask = utils.to_dense(
        #     data.x, data.edge_index, data.edge_attr, data.batch
        # )
        # dense_data = dense_data.mask(node_mask)
        # removes self loops
        # X, E = dense_data.X, dense_data.E
        X, E, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        # FIXME: remove self loops!
        if X.shape[1] == 9:  # for some reason in one case there are only 8 nodes
            xs.append(X)
            es.append(E)
            nodes_counts.append(node_mask.sum(-1))
    xs = np.concatenate(xs, axis=0)
    es = np.concatenate(es, axis=0)
    # es[np.where(es.sum(-1) == 0)] = np.eye(es.shape[-1])[0]
    es = np.where(es.sum(-1) == 0, np.eye(es.shape[-1])[0], es)
    nodes_counts = np.concatenate(nodes_counts, axis=0)
    return xs, es, nodes_counts


def create_graph(nodes, edges, edges_counts, nodes_counts, onehot: bool, train=False):
    return (
        graph.Graph.create(
            nodes=nodes.argmax(-1).astype(int),
            edges=edges.argmax(-1).astype(int),
            edges_counts=edges_counts,
            nodes_counts=nodes_counts,
            node_vocab_size=nodes.shape[-1],
            edge_vocab_size=edges.shape[-1],
        )
        if not onehot
        else gd.OneHotGraph.create_from_counts(
            nodes=jnp.asarray(nodes),
            edges=jnp.asarray(edges),
            nodes_counts=jnp.asarray(nodes_counts),
        )
    )


def get_dense_data(save_dir: str, batch_size: int, cache_dir: str, onehot: bool = True):
    cfg = Bunch.from_dict(
        {
            "dataset": {
                "datadir": os.path.join(save_dir, "data"),
                "remove_h": True,
            },
            "general": {
                "guidance_target": None,
                "name": "qm9",
            },
            "train": {
                "batch_size": batch_size,
                "num_workers": 0,
            },
        }
    )
    intermediate_cache = os.path.join(cache_dir, "intermediate_cache.pt")
    raw_cache = os.path.join(cache_dir, "raw_cache.pt")
    if not os.path.exists(intermediate_cache):
        print(f"Getting datamodule")
        datamodule = QM9DataModule(cfg)
        print(f"Got datamodule")
        dataset_infos = QM9infos(datamodule=datamodule, cfg=cfg)
        print(f"Got dataset infos, preparing data")
        datamodule.prepare_data()
        print(f"Prepared data")
        print("\nGetting train dataloader")
        train_dataloader = datamodule.train_dataloader()
        print(f"Got train dataloader\nGetting test dataloader")
        test_dataloader = datamodule.test_dataloader()
        jax_train_data = data_loader_to_data_items(train_dataloader)
        jax_test_data = data_loader_to_data_items(test_dataloader)
        train_smiles = get_train_smiles(
            cfg=cfg,
            train_dataloader=datamodule.train_dataloader(),
            dataset_infos=dataset_infos,
            evaluate_dataset=False,
        )
        with open(intermediate_cache, "wb") as f:
            pickle.dump(
                {
                    "train": jax_train_data,
                    "test": jax_test_data,
                    "infos": dataset_infos,
                    "train_smiles": train_smiles,
                },
                f,
            )
    else:
        print(f"Loading intermediate cache from {intermediate_cache}")
        with open(intermediate_cache, "rb") as f:
            intermediate_cache_dict = pickle.load(f)
        jax_train_data = intermediate_cache_dict["train"]
        jax_test_data = intermediate_cache_dict["test"]
        dataset_infos = intermediate_cache_dict["infos"]
        train_smiles = intermediate_cache_dict["train_smiles"]
    if not os.path.exists(raw_cache):
        assert False, f"Raw cache {raw_cache} does not exist"
        print(f"Got test dataloader\nConverting to dense numpy")
        train_nodes, train_edges, train_nodes_counts = to_dense_numpy(jax_train_data)
        print(f"Converted train dataloader\nConverting to dense numpy")
        test_nodes, test_edges, test_nodes_counts = to_dense_numpy(jax_test_data)
    else:
        print(f"Loading raw cache from {raw_cache}")
        with open(raw_cache, "rb") as f:
            raw_cache_dict = pickle.load(f)
        train_nodes = raw_cache_dict["train_nodes"]
        train_edges = raw_cache_dict["train_edges"]
        train_nodes_counts = raw_cache_dict["train_nodes_counts"]
        test_nodes = raw_cache_dict["test_nodes"]
        test_edges = raw_cache_dict["test_edges"]
        test_nodes_counts = raw_cache_dict["test_nodes_counts"]
    priors = compute_priors(train_nodes, train_edges, train_nodes_counts)
    print(f"Computed priors\nGetting train smiles")
    return Bunch.from_dict(
        {
            "train_nodes": train_nodes,
            "train_edges": train_edges,
            "train_nodes_counts": train_nodes_counts,
            "test_nodes": test_nodes,
            "test_edges": test_edges,
            "test_nodes_counts": test_nodes_counts,
            "feature_node_prior": priors.feature_node_prior,
            "feature_edge_prior": priors.feature_edge_prior,
            "structure_node_prior": priors.structure_node_prior,
            "structure_edge_prior": priors.structure_edge_prior,
            "nodes_dist": dataset_infos.n_nodes,
            "infos": dataset_infos,
            "train_smiles": train_smiles,
        }
    )


def compute_priors(train_nodes, train_edges, train_nodes_counts):
    feature_node_prior = np.zeros(train_nodes.shape[-1])
    feature_edge_prior = np.zeros(train_edges.shape[-1])
    structure_node_prior = np.zeros(2)  # any feature is present or not
    structure_edge_prior = np.zeros(2)  # any feature is present or not
    for i in range(train_nodes.shape[0]):
        current_nodes = train_nodes[i, : train_nodes_counts[i]]
        current_edges = train_edges[i, : train_nodes_counts[i], : train_nodes_counts[i]]
        nodes_feature_sum = current_nodes.sum(0)
        edges_feature_sum = current_edges.sum(0).sum(0)
        feature_node_prior += nodes_feature_sum
        feature_edge_prior += edges_feature_sum
        structure_node_prior += np.stack(
            [nodes_feature_sum[0], nodes_feature_sum[1:].sum()]
        )
        structure_edge_prior += np.stack(
            [edges_feature_sum[0], edges_feature_sum[1:].sum()]
        )

    structure_node_prior = structure_node_prior / structure_node_prior.sum()
    structure_edge_prior = structure_edge_prior / structure_edge_prior.sum()
    feature_node_prior = feature_node_prior / feature_node_prior.sum()
    feature_edge_prior = feature_edge_prior / feature_edge_prior.sum()
    return Bunch.from_dict(
        dict(
            feature_node_prior=feature_node_prior,
            feature_edge_prior=feature_edge_prior,
            structure_node_prior=structure_node_prior,
            structure_edge_prior=structure_edge_prior,
        )
    )


class CustomDataLoader:
    def __init__(self, *datasets, batch_size=32, shuffle=True):
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(datasets[0])))
        if shuffle:
            random.shuffle(self.indices)
        self.indices = np.array(self.indices)

    def __iter__(self):
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i : i + self.batch_size]
            batch = tuple([dataset[batch_indices] for dataset in self.datasets])
            yield batch

    def __len__(self):
        return len(self.indices) // self.batch_size


def get_dataloaders(bunch: Bunch, batch_size: int, onehot: bool) -> QM9Dataset:
    # train_loader = (
    #     Dataset.zip(
    #         (
    #             # Dataset.from_tensor_slices(indices_train),
    #             Dataset.from_tensor_slices(bunch.train_nodes),
    #             Dataset.from_tensor_slices(bunch.train_edges),
    #             Dataset.from_tensor_slices(np.ones_like(bunch.train_nodes_counts)),
    #             Dataset.from_tensor_slices(bunch.train_nodes_counts),
    #         )
    #     )
    #     .shuffle(1000)
    #     .batch(batch_size)
    # )
    # test_loader = (
    #     Dataset.zip(
    #         (
    #             # Dataset.from_tensor_slices(indices_train),
    #             Dataset.from_tensor_slices(bunch.test_nodes),
    #             Dataset.from_tensor_slices(bunch.test_edges),
    #             Dataset.from_tensor_slices(np.ones_like(bunch.test_nodes_counts)),
    #             Dataset.from_tensor_slices(bunch.test_nodes_counts),
    #         )
    #     )
    #     # .drop_remainder(True)
    #     .shuffle(1000).batch(batch_size)
    # )
    train_loader = CustomDataLoader(
        bunch.train_nodes,
        bunch.train_edges,
        np.ones_like(bunch.train_nodes_counts),
        bunch.train_nodes_counts,
        batch_size=batch_size,
    )
    test_loader = CustomDataLoader(
        bunch.test_nodes,
        bunch.test_edges,
        np.ones_like(bunch.test_nodes_counts),
        bunch.test_nodes_counts,
        batch_size=batch_size,
    )
    return QM9Dataset(
        train_loader=train_loader,
        test_loader=test_loader,
        feature_node_prior=np.array(bunch.feature_node_prior),
        feature_edge_prior=np.array(bunch.feature_edge_prior),
        structure_node_prior=np.array(bunch.structure_node_prior),
        structure_edge_prior=np.array(bunch.structure_edge_prior),
        nodes_dist=np.array(bunch.nodes_dist),
        max_node_feature=bunch.train_nodes.shape[-1],
        max_edge_feature=bunch.train_edges.shape[-1],
        infos=bunch.infos,
        train_smiles=bunch.train_smiles,
        n=bunch.train_nodes.shape[1],
    )


def load_data(save_dir: str, batch_size: int, onehot: bool = True):
    # gets the directory containing this file
    # save_dir = os.path.dirname(os.path.realpath(__file__))
    cache = os.path.join(save_dir, "qm9.pt")
    print(os.listdir(save_dir))
    print(f"Cache path: [red]{cache}[/red]")
    if not os.path.exists(cache):
        print("Creating cache")
        dense_data = get_dense_data(save_dir, batch_size, save_dir, onehot=False)
        pickle.dump(dense_data, open(cache, "wb"))
    else:
        print("Loading cache")
        dense_data = pickle.load(open(cache, "rb"))
        print(f"Done.")
    dataset = get_dataloaders(dense_data, batch_size, onehot)
    return dataset


if __name__ == "__main__":
    load_data("", 32)
