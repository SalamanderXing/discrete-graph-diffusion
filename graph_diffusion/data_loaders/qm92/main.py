from .qm9_dataset import QM9DataModule, QM9infos, get_train_smiles
from . import utils
import os
import numpy as np
from tqdm import tqdm
import pickle
import ipdb
from typing import Iterable
from dataclasses import dataclass
from jaxtyping import Float, Array
from jax import numpy as jnp
from ...shared.graph import Graph, SimpleGraphDist


@dataclass(frozen=True)
class QM9Dataset:
    train_loader: Iterable
    test_loader: Iterable
    max_node_feature: int
    max_edge_feature: int
    n: int
    nodes_dist: Float[Array, "k"]
    node_prior: Float[Array, "m"]
    edge_prior: Float[Array, "l"]


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
    xs = []
    es = []
    nodes_counts = []
    for data in tqdm(dataloader):
        dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        xs.append(X.numpy())
        es.append(E.numpy())
        nodes_counts.append(node_mask.sum(-1).numpy())
    xs = np.concatenate(xs, axis=0)
    es = np.concatenate(es, axis=0)
    nodes_counts = np.concatenate(nodes_counts, axis=0)

    zero_node = np.zeros_like(xs)[..., 0][..., None]
    zero_edge = np.zeros_like(es)[..., 0][..., None]
    xs_ = np.concatenate([zero_node, xs], axis=-1)
    es_ = np.concatenate([zero_edge, es], axis=-1)
    xs_[np.where(xs.sum(-1) == 0)] = np.eye(xs_.shape[-1])[0]
    es_[np.where(es.sum(-1) == 0)] = np.eye(es_.shape[-1])[0]
    assert np.all(xs_.sum(-1) == 1)
    assert np.all(es_.sum(-1) == 1)
    return xs_, es_, nodes_counts


def create_graph(nodes, edges, edges_counts, nodes_counts, train=False):
    nodes = np.asarray(nodes)
    edges = np.asarray(edges)
    nodes_counts = np.asarray(nodes_counts)
    edges_counts = np.asarray(edges_counts)
    assert nodes_counts.shape == edges_counts.shape and len(nodes_counts.shape) == 1
    if len(nodes.shape) > 3:
        nodes = nodes.squeeze(-1)
    if train:
        batch_size, n, _ = nodes.shape

        # if np.random.rand() < 0.5:
        #     # takes a subgraph
        #     max_n = np.max(nodes_counts)
        #     new_nodes_counts = np.random.randint(1, max_n, size=batch_size)
        #     # ipdb.set_trace()
        #     for i in range(batch_size):
        #         nodes[i, new_nodes_counts[i] :] = np.eye(nodes.shape[-1])[0]
        #         edges[i, new_nodes_counts[i] :, new_nodes_counts[i] :] = np.eye(
        #             edges.shape[-1]
        #         )[0]
        # pass

    nodes = jnp.asarray(nodes)

    edges = jnp.asarray(edges)
    edges_counts = jnp.asarray(edges_counts)
    # assert len(nodes_counts.shape) == 1
    nodes_counts = jnp.asarray(nodes_counts)
    # assert len(nodes_counts.shape) == 1
    return (
        Graph.create(
            nodes=nodes,
            edges=edges,
            edges_counts=edges_counts,
            nodes_counts=nodes_counts,
        )
        if len(edges.shape) < 4
        else SimpleGraphDist.create(
            nodes=nodes,
            edges=edges,
            edges_counts=edges_counts,
            nodes_counts=nodes_counts,
            # node_masks=jnp.ones_like(nodes_counts),
        )
    )


def get_dense_data(save_dir: str, batch_size: int):
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
    train_smiles = get_train_smiles(
        cfg=cfg,
        train_dataloader=datamodule.train_dataloader(),
        dataset_infos=dataset_infos,
        evaluate_dataset=False,
    )
    train_dataloader = datamodule.train_dataloader()
    test_dataloader = datamodule.test_dataloader()
    train_nodes, train_edges, train_nodes_counts = to_dense_numpy(train_dataloader)
    test_nodes, test_edges, test_nodes_counts = to_dense_numpy(test_dataloader)
    node_prior, edge_prior = compute_priors(
        train_nodes, train_edges, train_nodes_counts
    )
    return Bunch.from_dict(
        {
            "train_nodes": train_nodes,
            "train_edges": train_edges,
            "train_nodes_counts": train_nodes_counts,
            "test_nodes": test_nodes,
            "test_edges": test_edges,
            "test_nodes_counts": test_nodes_counts,
            "node_prior": node_prior,
            "edge_prior": edge_prior,
            "nodes_dist": dataset_infos.n_nodes.numpy(),
        }
    )


def compute_priors(train_nodes, train_edges, train_nodes_counts):
    node_prior = np.zeros(train_nodes.shape[-1])
    edge_prior = np.zeros(train_edges.shape[-1])
    for i in range(train_nodes.shape[0]):
        node_prior += train_nodes[i, : train_nodes_counts[i]].sum(0)
        edge_prior += (
            train_edges[i, : train_nodes_counts[i], : train_nodes_counts[i]]
            .sum(0)
            .sum(0)
        )
    node_prior = node_prior / node_prior.sum()
    edge_prior = edge_prior / edge_prior.sum()
    return node_prior, edge_prior


def get_dataloaders(bunch: Bunch, batch_size: int) -> QM9Dataset:
    import tensorflow as tf

    tf.config.experimental.set_visible_devices([], "GPU")
    from tensorflow.data import Dataset

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
        .repeat()
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
        .shuffle(1000)
        .repeat()
        .batch(batch_size)
    )
    train_loader = map(
        lambda x: create_graph(*x, train=True),
        train_loader,
    )
    test_loader = map(
        lambda x: create_graph(*x),
        test_loader,
    )
    return QM9Dataset(
        train_loader=train_loader,
        test_loader=test_loader,
        node_prior=jnp.array(bunch.node_prior),
        edge_prior=jnp.array(bunch.edge_prior),
        nodes_dist=jnp.array(bunch.nodes_dist),
        max_node_feature=bunch.train_nodes.shape[-1],
        max_edge_feature=bunch.train_edges.shape[-1],
        n=bunch.train_nodes.shape[1],
    )


def load_data(save_dir: str, batch_size: int):
    # gets the directory containing this file
    save_dir = os.path.dirname(os.path.realpath(__file__))
    cache = os.path.join(save_dir, "qm9.pt")
    if not os.path.exists(cache):
        print("Creating cache")
        dense_data = get_dense_data(save_dir, batch_size)
        pickle.dump(dense_data, open(cache, "wb"))
    else:
        print("Loading cache")
        dense_data = pickle.load(open(cache, "rb"))
    dataset = get_dataloaders(dense_data, batch_size)
    return dataset


if __name__ == "__main__":
    load_data("", 32)
