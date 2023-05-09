from torch_geometric.datasets import TUDataset, QM9
import jax_dataloader as jdl
from torch_geometric.utils import to_dense_batch, to_dense_adj, remove_self_loops
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


@dataclass(frozen=True)
class DatasetInfo:
    num_node_features: int
    num_edge_features: int
    max_num_nodes: int


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
    cache_location = os.path.join(save_path, "processed.pt")
    if not (cache and os.path.exists(cache_location)):
        print("Processing dataset...")
        dataset = TUDataset(
            root=save_path,
            name=name,  # "PTC_MR",  # "MUTAG"
            use_node_attr=True,
            use_edge_attr=True,
        )
        items = len(dataset)
        # Get the maximum number of nodes (atoms) in the dataset
        max_n = max([data.num_nodes for data in dataset])
        print(f"[red]Max number of nodes:[/red] {max_n}")
        # Get unique atom types
        max_n_atom = dataset[0].x.shape[1]

        num_edge_features = dataset.num_edge_features

        nodes = np.zeros((items, max_n, max_n_atom))
        edges = np.zeros((items, max_n, max_n, num_edge_features))
        node_masks = np.zeros((items, max_n))
        for idx, data in enumerate(dataset):
            num_nodes = data.num_nodes
            # Fill in the node features as one-hot encoded atomic numbers
            atom_one_hot = data.x.numpy()
            nodes[idx, :num_nodes, :] = atom_one_hot

            # Fill in the edge features
            edge_indices = data.edge_index.numpy()
            for j, (src, dst) in enumerate(edge_indices.T):
                edges[idx, src, dst, :] = data.edge_attr[j].numpy()
                edges[idx, dst, src, :] = data.edge_attr[
                    j
                ].numpy()  # Assuming undirected graph

            # Fill in the node_masks
            node_masks[idx, :num_nodes] = 1
        nodes = jnp.array(nodes)
        edges = jnp.array(edges)
        node_masks = jnp.array(node_masks)
        with h5py.File(cache_location, "w") as hf:
            hf.create_dataset("nodes", data=nodes)
            hf.create_dataset("edges", data=edges)
            hf.create_dataset("node_masks", data=node_masks)

        print("Processed dataset.")
    else:
        print("Loading cached dataset...")
        nodes, edges, node_masks = pickle.load(open(cache_location, "rb"))
        items = len(nodes)
        max_n = nodes.shape[1]
        max_n_atom = nodes.shape[2]
        num_edge_features = edges.shape[-1]
        print("Loaded cached dataset.")

    shuffling_indices = random.permutation(x=items, key=seed)
    nodes = nodes[shuffling_indices]
    edges = edges[shuffling_indices]
    node_masks = node_masks[shuffling_indices]
    train_size = int(train_size * items)
    train_dataset = jdl.ArrayDataset(
        nodes[:train_size],
        edges[:train_size],
        node_masks[:train_size],
    )
    test_dataset = jdl.ArrayDataset(
        nodes[train_size:],
        edges[train_size:],
        node_masks[train_size:],
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
    )
    return train_loader, test_loader, dataset_info


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
    cache_location = os.path.join(save_path, "processed_no_attributes.h5")
    if not (cache and os.path.exists(cache_location)):
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

        num_edge_features = max(dataset.num_edge_features, 2)
        size = 4
        nodes = np.zeros((len_dataset, max_n, size))
        edges = np.zeros((len_dataset, max_n, max_n, size))
        node_masks = np.zeros((len_dataset, max_n))
        for idx, data in enumerate(dataset):
            num_nodes = data.num_nodes
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

        # pickle.dump(
        #     (nodes, edges, node_masks),
        #     open(cache_location, "wb"),
        # )
        # saves the data in h5 format
        nodes = jnp.array(nodes)
        edges = jnp.array(edges)
        node_masks = jnp.array(node_masks)
        with h5py.File(cache_location, "w") as hf:
            hf.create_dataset("nodes", data=nodes)
            hf.create_dataset("edges", data=edges)
            hf.create_dataset("node_masks", data=node_masks)

        print("Processed and saved dataset.")
    else:
        print("Loading cached dataset...")
        # loads the data in h5 format
        with h5py.File(cache_location, "r") as hf:
            nodes = np.asarray(hf["nodes"][:])  # type: ignore
            edges = np.asarray(hf["edges"][:])  # type: ignore
            node_masks = np.asarray(hf["node_masks"][:])  # type: ignore
        len_dataset = len(nodes)
        print("Loaded cached dataset.")
    shuffling_indices = random.permutation(x=len_dataset, key=seed)
    nodes = nodes[shuffling_indices]
    edges = edges[shuffling_indices]
    node_masks = node_masks[shuffling_indices]
    train_size = int(train_size * len_dataset)
    print("Creating datasets")
    train_dataset = jdl.ArrayDataset(
        jnp.array(nodes[:train_size]),
        jnp.array(edges[:train_size]),
        jnp.array(node_masks[:train_size]),
    )
    test_dataset = jdl.ArrayDataset(
        jnp.array(nodes[train_size:]),
        jnp.array(edges[train_size:]),
        jnp.array(node_masks[train_size:]),
    )
    print("Creating data loaders")
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
        num_node_features=nodes.shape[-1],
        num_edge_features=edges.shape[-1],
        max_num_nodes=nodes.shape[1],
    )
    print("Done")
    return train_loader, test_loader, dataset_info
