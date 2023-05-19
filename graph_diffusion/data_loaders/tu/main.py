import pickle
import os
import h5py
from jaxtyping import Float, Array
from rich import print
import ipdb
from dataclasses import dataclass
import numpy as np
from scipy.signal import savgol_filter
from jax import numpy as jnp
from ...shared.graph import Graph, SimpleGraphDist

# @dataclass(frozen=True)
# class DatasetInfo:
#     num_node_features: int
#     num_edge_features: int
#     max_num_nodes: int
#     nodes_dist: np.ndarray
#     edges_prior: np.ndarray
#     nodes_prior: np.ndarray


def __to_dense(x, edge_index, edge_attr, batch):
    x, node_mask = to_dense_batch(x=x, batch=batch)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    # TODO: carefully check if setting node_mask as a bool breaks the continuous case
    max_num_nodes = x.size(1)
    e = to_dense_adj(
        edge_index=edge_index,
        batch=batch,
        edge_attr=edge_attr,
        max_num_nodes=max_num_nodes,
    )
    e = __encode_no_edge(e)
    return x, e, node_mask


def __encode_no_edge(E):
    # assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = (
        torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    )
    E[diag] = 0
    return E


def compute_distribution(
    node_masks: Float[Array, "bs m"], margin: int
) -> Float[Array, "size"]:
    per_node_n = jnp.sum(node_masks, axis=1).astype(int)
    max_n = jnp.max(per_node_n) + margin
    dist = jnp.zeros(max_n, dtype=int)
    for n in per_node_n:
        dist = dist.at[n].add(1)
    norm_dist = dist / jnp.sum(dist)
    # performs a smoothing of the distribution
    norm_dist_smooth = jnp.clip(
        jnp.array(savgol_filter(np.array(norm_dist), 5, 3)), 0, 1
    )
    # applies a moving average to the distribution
    norm_dist_smooth_2 = jnp.convolve(norm_dist_smooth, jnp.ones(3) / 3, mode="same")
    norm_dist_smooth_3 = jnp.convolve(norm_dist_smooth_2, jnp.ones(5) / 5, mode="same")
    norm_dist_smooth_3 = norm_dist_smooth_3 / jnp.sum(norm_dist_smooth_3)
    return norm_dist_smooth_3


def save_cache(cache_location, test_dataset, train_dataset, freqs):
    with h5py.File(cache_location, "w") as hf:
        hf.create_dataset("testset", data=test_dataset)
        hf.create_dataset("trainset", data=train_dataset)
        hf.create_dataset("freqs", data=freqs)


def load_cache(
    cache_location: str,
):  # -> tuple[jdl.ArrayDataset, jdl.ArrayDataset, Array]:
    with h5py.File(cache_location, "r") as hf:
        test_dataset = hf["testset"][:]  # type: ignore
        train_dataset = hf["trainset"][:]  # type: ignore
        freqs = hf["freqs"][:]  # type: ignore
    return test_dataset, train_dataset, freqs


# def compute_priors(nodes:Float[], edges):


def split_dataset(
    *,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    nodes: np.ndarray,
    edges: np.ndarray,
    nodes_counts: np.ndarray,
    batch_size: int,
    edges_counts: np.ndarray,
    node_masks: np.ndarray,
):
    edges_counts = np.array(edges_counts)
    train_nodes = nodes[train_indices]
    train_edges = edges[train_indices]
    train_masks = node_masks[train_indices]
    train_nodes_counts = nodes_counts[train_indices]
    train_edges_counts = edges_counts[train_indices]
    test_nodes = nodes[test_indices]
    test_edges = edges[test_indices]
    test_nodes_counts = nodes_counts[test_indices]
    test_edges_counts = edges_counts[test_indices]
    max_n = train_nodes.shape[1]
    num_edge_features = train_edges.shape[-1]
    max_n_atom = train_nodes.shape[-1]
    train_nodes = np.array(train_nodes).astype(float)[..., None]
    train_edges = np.array(train_edges).astype(float)
    # train_node_masks = np.array(train_node_masks)
    test_nodes = np.array(test_nodes).astype(float)[..., None]
    test_edges = np.array(test_edges).astype(float)
    # test_node_masks = np.array(test_node_masks)
    # freqs = (
    #     compute_distribution(train_node_masks, margin=10) if do_dist else np.array(-1)
    # )
    import tensorflow as tf

    tf.config.experimental.set_visible_devices([], "GPU")
    from tensorflow.data import Dataset
    from typing import Iterable

    @dataclass(frozen=True)
    class TUDataset:
        train_loader: Iterable
        test_loader: Iterable
        max_node_feature: int
        max_edge_feature: int
        n: int
        nodes_dist: Float[Array, "k"]
        node_prior: Float[Array, "m"]
        edge_prior: Float[Array, "l"]

    train_dataset = (
        Dataset.zip(
            (
                Dataset.from_tensor_slices(train_nodes),
                Dataset.from_tensor_slices(train_edges),
                # Dataset.from_tensor_slices(train_node_masks),
                Dataset.from_tensor_slices(train_edges_counts),
                Dataset.from_tensor_slices(train_nodes_counts),
            )
        )
        .repeat()
        .shuffle(1000)
    )
    test_dataset = Dataset.zip(
        (
            Dataset.from_tensor_slices(test_nodes),
            Dataset.from_tensor_slices(test_edges),
            # Dataset.from_tensor_slices(test_node_masks),
            Dataset.from_tensor_slices(test_edges_counts),
            Dataset.from_tensor_slices(test_nodes_counts),
        )
    ).repeat()

    def create_graph(nodes, edges, edges_counts, nodes_counts):
        nodes = jnp.asarray(nodes)
        if len(nodes.shape) > 3:
            nodes = nodes.squeeze(-1)
        edges = jnp.asarray(edges)
        edges_counts = jnp.asarray(edges_counts)
        nodes_counts = jnp.asarray(nodes_counts)
        return (
            Graph.create(
                nodes=nodes,
                edges=edges,
                edges_counts=edges_counts,
                nodes_counts=nodes_counts,
            )
            if len(edges.shape) < 4
            else SimpleGraphDist(
                nodes=nodes,
                edges=edges,
                edges_counts=edges_counts,
                nodes_counts=nodes_counts,
                node_masks=jnp.ones_like(nodes_counts),
            )
        )

    # return train_dataset, test_dataset
    train_loader = train_dataset.batch(batch_size)
    # .map(
    #     lambda nodes, edges, edges_counts, nodes_counts: tf.py_function(
    #         func=create_graph,
    #         inp=[nodes, edges, edges_counts, nodes_counts],
    #         Tout=Graph,  # tf.float32,  # replace this with the actual data type of your Graph
    #     )
    # )

    test_loader = test_dataset.batch(batch_size)
    # .map(
    #     lambda nodes, edges, edges_counts, nodes_counts: tf.py_function(
    #         func=create_graph,
    #         inp=[nodes, edges, edges_counts, nodes_counts],
    #         Tout=tf.float32,  # replace this with the actual data type of your Graph
    #     )
    # )

    # dataset_info = DatasetInfo(
    #     num_node_features=max_n_atom,
    #     num_edge_features=num_edge_features,
    #     max_num_nodes=max_n,
    #     nodes_dist=np.array(freqs),
    #     nodes_prior=train_nodes.reshape(-1, train_nodes.shape[-1]).mean(axis=0),
    #     edges_prior=train_edges.reshape(-1, train_edges.shape[-1]).mean(axis=0),
    # )
    train_loader = map(
        lambda x: create_graph(*x),
        train_loader,
    )
    test_loader = map(
        lambda x: create_graph(*x),
        test_loader,
    )
    nodes_dist = compute_distribution(train_masks, margin=5)
    nodes_prior = jnp.array(train_nodes.mean(axis=(0, 1)).squeeze())
    edges_prior = jnp.array(train_edges.mean(axis=(0, 1, 2)).squeeze())
    return TUDataset(
        train_loader=train_loader,
        test_loader=train_loader,
        max_node_feature=int(np.max(train_nodes)),
        max_edge_feature=int(np.max(train_edges)),
        n=int(max_n),
        nodes_dist=nodes_dist,
        node_prior=nodes_prior,
        edge_prior=edges_prior,
    )


def load_data(
    *,
    save_path: str,
    train_size: float = 0.8,
    seed,
    batch_size: int,
    name: str = "MUTAG",
    verbose: bool = True,
    cache: bool = False,
    one_hot: bool = False,
):
    cache_location = os.path.join(save_path, "processed.pickle")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("Cache not found, creating new one...")
    from torch_geometric.datasets import TUDataset, QM9
    from torch_geometric.utils import (
        to_dense_batch,
        to_dense_adj,
        remove_self_loops,
    )

    print("Processing dataset...")
    dataset = TUDataset(
        root=save_path,
        name=name,  # "PTC_MR",  # "MUTAG"
        use_node_attr=True,
        use_edge_attr=True,
    )
    items = len(dataset)
    # Get the maximum number of nodes (atoms) in the dataset
    max_n = (
        max([data.num_nodes for data in dataset]) + 1
    )  # one for the absence of a node
    print(f"[red]Max number of nodes:[/red] {max_n}")
    # Get unique atom types
    max_n_atom = dataset[0].x.shape[1]

    num_edge_features = dataset.num_edge_features + 1  # one for the absence of an edge

    nodes = np.zeros((items, max_n, max_n_atom))
    edges = np.zeros((items, max_n, max_n, num_edge_features))
    node_masks = np.zeros((items, max_n))
    num_nodes_list = []
    edges_counts = []
    for idx, data in enumerate(dataset):
        tot_edges = set()
        num_nodes = data.num_nodes
        # Fill in the node features as one-hot encoded atomic numbers
        atom_one_hot = data.x.numpy()
        nodes[idx, :num_nodes, :] = atom_one_hot

        num_nodes_list.append(num_nodes)
        # Fill in the edge features
        edge_indices = data.edge_index.numpy()
        for j, (src, dst) in enumerate(edge_indices.T):
            edges[idx, src, dst, :] = [0] + data.edge_attr[j].tolist()
            edges[idx, dst, src, :] = [0] + data.edge_attr[j].tolist()
            tot_edges.add((src.item(), dst.item()))
            tot_edges.add((dst.item(), src.item()))

        # Fill in the node_masks
        node_masks[idx, :num_nodes] = 1
        edges_counts.append(len(tot_edges))
    # print(edges_counts)

    edges[np.where(edges.sum(axis=-1) == 0)] = np.eye(num_edge_features)[0]
    nodes[np.where(nodes.sum(axis=-1) == 0)] = np.eye(max_n_atom)[0]

    if not one_hot:
        nodes = np.argmax(nodes, axis=-1)
        edges = np.argmax(edges, axis=-1)

    num_nodes_list = np.array(num_nodes_list)

    nodes = np.array(nodes)
    edges = np.array(edges)
    node_masks = np.array(node_masks)

    print("Processed dataset.")
    items = len(nodes)
    num_edge_features = edges.shape[-1]
    import torch

    shuffling_indices = torch.randperm(items)
    train_size = int(train_size * items)
    train_indices = shuffling_indices[:train_size]
    test_indices = shuffling_indices[train_size:]

    result = split_dataset(
        train_indices=train_indices.numpy(),
        test_indices=test_indices.numpy(),
        nodes=nodes,
        edges=edges,
        batch_size=batch_size,
        edges_counts=np.asarray(edges_counts),
        nodes_counts=num_nodes_list,
        node_masks=node_masks,
    )
    return result
    # return cache


def load_data_no_attributes(
    *,
    save_path: str,
    train_size: float = 0.8,
    seed,
    batch_size: int,
    name: str = "PTC_MR",
    verbose: bool = True,
    cache: bool = False,
    one_hot: bool = False,
):
    print("Processing dataset...")
    dataset = TUDataset(
        root=save_path,
        name=name,  # "PTC_MR",  # "MUTAG"
        use_node_attr=False,
        use_edge_attr=False,
    )
    len_dataset = len(dataset)
    # Get the maximum number of nodes (atoms) in the dataset
    max_n = max([data.num_nodes for data in dataset])
    print(f"[red underline]Max number of nodes:[/red underline] {max_n}")
    # Get unique atom types
    # max_n_atom = dataset[0].x.shape[1]

    size = 5
    nodes = np.zeros((len_dataset, max_n, size))
    edges = np.zeros((len_dataset, max_n, max_n, 2))
    node_masks = np.zeros((len_dataset, max_n))
    num_nodes_list = []
    for idx, data in enumerate(dataset):
        num_nodes = data.num_nodes
        num_nodes_list.append(num_nodes)
        # Fill in the node features as one-hot encoded atomic numbers
        # atom_one_hot = data.x.numpy()
        nodes[idx, :num_nodes, 0] = 1

        # Fill in the edge features
        edge_indices = data.edge_index.numpy()

        # Fill in the node_masks
        node_masks[idx, :num_nodes] = 1

        for j, (src, dst) in enumerate(edge_indices.T):
            edges[idx, src, dst, 0] = 1
            edges[idx, dst, src, 0] = 1

    edges[np.where(edges.sum(axis=-1) == 0)] = np.eye(2)[-1]
    nodes[np.where(nodes.sum(axis=-1) == 0)] = np.eye(size)[-1]
    # make the diagonal 0
    edges[:, np.arange(max_n), np.arange(max_n), :] = [0.0, 1.0]
    # make sure the matrix is symmetric
    edges = np.maximum(edges, edges.transpose(0, 2, 1, 3))
    # turns num_nodes_list into a distribution
    num_nodes_list = np.array(num_nodes_list)

    assert (edges.sum(axis=-1) == 1).all(), ipdb.set_trace()
    assert (nodes.sum(axis=-1) == 1).all()
    # saves the data in h5 format
    nodes = jnp.array(nodes)
    edges = jnp.array(edges)
    node_masks = jnp.array(node_masks)
    print("Processed and saved dataset.")

    shuffling_indices = random.permutation(x=len_dataset, key=seed)
    nodes = nodes[shuffling_indices]
    edges = edges[shuffling_indices]
    node_masks = node_masks[shuffling_indices]
    train_size = int(train_size * len_dataset)
    train_indices = shuffling_indices[:train_size]
    test_indices = shuffling_indices[train_size:]

    return split_dataset(
        train_indices,
        test_indices,
        nodes,
        edges,
        node_masks,
        num_nodes_list,
        batch_size,
    )
