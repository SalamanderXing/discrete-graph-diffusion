from torch_geometric.datasets import TUDataset, QM9
import jax_dataloader as jdl
from torch_geometric.utils import to_dense_batch, to_dense_adj, remove_self_loops
import jax
from jaxtyping import Array, Float
import pickle
import torch
import os
import h5py
from rich import print
import ipdb
from dataclasses import dataclass
from jax import numpy as jnp
import numpy as np
from mate.jax import Key
from jax import random
from scipy.signal import savgol_filter


@dataclass(frozen=True)
class DatasetInfo:
    num_node_features: int
    num_edge_features: int
    max_num_nodes: int
    nodes_dist: Float[Array, "size"]
    edges_prior: Float[Array, "ee"]
    nodes_prior: Float[Array, "en"]


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
    # import matplotlib.pyplot as plt
    #
    # plt.plot(norm_dist)
    # plt.plot(norm_dist_smooth)
    # plt.plot(norm_dist_smooth_2)
    # plt.plot(norm_dist_smooth_3)
    # plt.show()
    return norm_dist_smooth_3


def save_cache(cache_location, test_dataset, train_dataset, freqs):
    with h5py.File(cache_location, "w") as hf:
        hf.create_dataset("testset", data=test_dataset)
        hf.create_dataset("trainset", data=train_dataset)
        hf.create_dataset("freqs", data=freqs)


def load_cache(cache_location: str) -> tuple[jdl.ArrayDataset, jdl.ArrayDataset, Array]:
    with h5py.File(cache_location, "r") as hf:
        test_dataset = hf["testset"][:]  # type: ignore
        train_dataset = hf["trainset"][:]  # type: ignore
        freqs = hf["freqs"][:]  # type: ignore
    return test_dataset, train_dataset, freqs


# def compute_priors(nodes:Float[], edges):


def create_dataloders(
    train_nodes,
    train_edges,
    train_node_masks,
    test_nodes,
    test_edges,
    test_node_masks,
    batch_size,
):
    max_n = train_nodes.shape[1]
    num_edge_features = train_edges.shape[-1]
    max_n_atom = train_nodes.shape[-1]
    train_nodes = jnp.array(train_nodes)
    train_edges = jnp.array(train_edges)
    train_node_masks = jnp.array(train_node_masks)

    test_nodes = jnp.array(test_nodes)
    test_edes = jnp.array(test_edges)
    test_node_masks = jnp.array(test_node_masks)
    freqs = compute_distribution(train_node_masks, margin=10)
    train_dataset = jdl.ArrayDataset(
        train_nodes,
        train_edges,
        train_node_masks,
    )
    test_dataset = jdl.ArrayDataset(
        test_nodes,
        test_edges,
        test_node_masks,
    )
    # return train_dataset, test_dataset
    train_loader = jdl.DataLoader(
        train_dataset,
        backend="jax",
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = jdl.DataLoader(
        test_dataset, backend="jax", batch_size=batch_size, shuffle=True
    )
    dataset_info = DatasetInfo(
        num_node_features=max_n_atom,
        num_edge_features=num_edge_features,
        max_num_nodes=max_n,
        nodes_dist=jnp.array(freqs),
        nodes_prior=train_nodes.reshape(-1, train_nodes.shape[-1]).mean(axis=0),
        edges_prior=train_edges.reshape(-1, train_edges.shape[-1]).mean(axis=0),
    )
    return train_loader, test_loader, dataset_info


def split_dataset(
    train_indices: Array,
    test_indices: Array,
    nodes: Array,
    edges: Array,
    node_masks: Array,
    batch_size: int,
):
    train_nodes = nodes[train_indices]
    train_edges = edges[train_indices]
    train_node_masks = node_masks[train_indices]
    test_nodes = nodes[test_indices]
    test_edges = edges[test_indices]
    test_node_masks = node_masks[test_indices]
    return create_dataloders(
        train_nodes,
        train_edges,
        train_node_masks,
        test_nodes,
        test_edges,
        test_node_masks,
        batch_size,
    )


def load_data(
    *,
    save_path: str,
    train_size: float = 0.8,
    seed: Key,
    batch_size: int,
    name: str = "PTC_MR",
    verbose: bool = True,
    cache: bool = True,
):
    cache_location = os.path.join(save_path, "processed.h5")
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
    for idx, data in enumerate(dataset):
        num_nodes = data.num_nodes
        # Fill in the node features as one-hot encoded atomic numbers
        atom_one_hot = data.x.numpy()
        nodes[idx, :num_nodes, :] = atom_one_hot

        num_nodes_list.append(num_nodes)
        # Fill in the edge features
        edge_indices = data.edge_index.numpy()
        for j, (src, dst) in enumerate(edge_indices.T):
            edges[idx, src, dst, :] = data.edge_attr[j].tolist() + [0]
            edges[idx, dst, src, :] = data.edge_attr[j].tolist() + [
                0
            ]  # Assuming undirected graph

        # Fill in the node_masks
        node_masks[idx, :num_nodes] = 1

    edges[np.where(edges.sum(axis=-1) == 0)] = np.eye(num_edge_features)[-1]
    nodes[np.where(nodes.sum(axis=-1) == 0)] = np.eye(max_n_atom)[-1]

    num_nodes_list = np.array(num_nodes_list)

    nodes = jnp.array(nodes)
    edges = jnp.array(edges)
    node_masks = jnp.array(node_masks)

    print("Processed dataset.")
    items = len(nodes)
    num_edge_features = edges.shape[-1]

    shuffling_indices = random.permutation(x=items, key=seed)
    train_size = int(train_size * items)
    train_indices = shuffling_indices[:train_size]
    test_indices = shuffling_indices[train_size:]

    return split_dataset(
        train_indices,
        test_indices,
        nodes,
        edges,
        node_masks,
        batch_size,
    )


def load_data_no_attributes(
    *,
    save_path: str,
    train_size: float = 0.8,
    seed: Key,
    batch_size: int,
    name: str = "PTC_MR",
    verbose: bool = True,
    cache: bool = False,
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
        batch_size,
    )
