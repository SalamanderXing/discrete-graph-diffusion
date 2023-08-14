# import ipdb;ipdb.set_trace()
import os
import numpy as np
from tqdm import tqdm
import pickle
import ipdb
from typing import Iterable
from dataclasses import dataclass
from jaxtyping import Float, Array
from jax import numpy as jnp
import tensorflow as tf
from tensorflow.data import Dataset
from mate.jax import SFloat
from .qm9_dataset import QM9DataModule, QM9infos, get_train_smiles
from ...shared.graph import graph_distribution as gd, graph

tf.config.experimental.set_visible_devices([], "GPU")
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

    @classmethod
    def from_dict(cls, dictionary):
        bunch = cls()
        for key, value in dictionary.items():
            if isinstance(value, dict):
                bunch.__dict__[key] = cls.from_dict(value)
            else:
                bunch.__dict__[key] = value
        return bunch


def to_dense_numpy(dataloader):
    from . import utils

    xs = []
    es = []
    nodes_counts = []
    for data in tqdm(dataloader):
        dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X.numpy(), dense_data.E.numpy()
        if X.shape[1] == 9:  # for some reason in one case there are only 8 nodes
            xs.append(X)
            es.append(E)
            nodes_counts.append(node_mask.sum(-1).numpy())
    xs = np.concatenate(xs, axis=0)
    es = np.concatenate(es, axis=0)
    es[np.where(es.sum(-1) == 0)] = np.eye(es.shape[-1])[0]
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


def get_dense_data(save_dir: str, batch_size: int, onehot: bool = True):
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
    datamodule = QM9DataModule(cfg)
    dataset_infos = QM9infos(datamodule=datamodule, cfg=cfg)
    datamodule.prepare_data()
    train_dataloader = datamodule.train_dataloader()
    test_dataloader = datamodule.test_dataloader()
    train_nodes, train_edges, train_nodes_counts = to_dense_numpy(train_dataloader)
    test_nodes, test_edges, test_nodes_counts = to_dense_numpy(test_dataloader)
    priors = compute_priors(train_nodes, train_edges, train_nodes_counts)
    train_smiles = get_train_smiles(
        cfg=cfg,
        train_dataloader=datamodule.train_dataloader(),
        dataset_infos=dataset_infos,
        evaluate_dataset=False,
    )
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
            "nodes_dist": dataset_infos.n_nodes.numpy(),
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


def get_dataloaders(bunch: Bunch, batch_size: int, onehot: bool) -> QM9Dataset:
    train_loader = (
        Dataset.zip(
            (
                # Dataset.from_tensor_slices(indices_train),
                Dataset.from_tensor_slices(bunch.train_nodes),
                Dataset.from_tensor_slices(bunch.train_edges),
                Dataset.from_tensor_slices(np.ones_like(bunch.train_nodes_counts)),
                Dataset.from_tensor_slices(bunch.train_nodes_counts),
            )
        )
        .shuffle(1000)
        .batch(batch_size)
    )
    test_loader = (
        Dataset.zip(
            (
                # Dataset.from_tensor_slices(indices_train),
                Dataset.from_tensor_slices(bunch.test_nodes),
                Dataset.from_tensor_slices(bunch.test_edges),
                Dataset.from_tensor_slices(np.ones_like(bunch.test_nodes_counts)),
                Dataset.from_tensor_slices(bunch.test_nodes_counts),
            )
        )
        # .drop_remainder(True)
        .shuffle(1000).batch(batch_size)
    )
    return QM9Dataset(
        train_loader=train_loader,
        test_loader=test_loader,
        feature_node_prior=jnp.array(bunch.feature_node_prior),
        feature_edge_prior=jnp.array(bunch.feature_edge_prior),
        structure_node_prior=jnp.array(bunch.structure_node_prior),
        structure_edge_prior=jnp.array(bunch.structure_edge_prior),
        nodes_dist=jnp.array(bunch.nodes_dist),
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
    if not os.path.exists(cache):
        print("Creating cache")
        dense_data = get_dense_data(save_dir, batch_size, onehot=False)
        pickle.dump(dense_data, open(cache, "wb"))
    else:
        print("Loading cache")
        dense_data = pickle.load(open(cache, "rb"))
    dataset = get_dataloaders(dense_data, batch_size, onehot)
    return dataset


if __name__ == "__main__":
    load_data("", 32)
