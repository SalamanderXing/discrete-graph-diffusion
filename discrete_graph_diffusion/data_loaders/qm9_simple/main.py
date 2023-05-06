from torch_geometric.datasets import TUDataset, QM9
from torch_geometric.utils import to_dense_batch, to_dense_adj, remove_self_loops
import jax_dataloader as jdl
import torch
import ipdb
from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
import jax_dataloader as jdl
from jax import random

def to_dense(x, edge_index, edge_attr, batch):
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
    e = encode_no_edge(e)
    return x, e, node_mask


@dataclass(frozen=True)
class DatasetInfo:
    num_node_features: int
    num_edge_features: int
    max_num_nodes: int


def encode_no_edge(E):
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


@dataclass(frozen=True)
class Dataset:
    x: torch.Tensor
    e: torch.Tensor
    node_mask: torch.Tensor


def load_data(
    *, save_path: str, seed: int = 0, train_size: float = 0.8, batch_size: int
):
    # dataset = TUDataset(
    #     root=save_path, name="PTC_MR", use_node_attr=True, use_edge_attr=True
    # )
    dataset = QM9(root=save_path)

    items = len(dataset)
    # Get the maximum number of nodes (atoms) in the dataset
    max_n = max([data.num_nodes for data in dataset])

    # Get unique atom types
    max_n_atom = max([data.z.max().item() for data in dataset])

    num_edge_features = dataset.num_edge_features

    nodes = np.zeros((items, max_n, max_n_atom))
    edges = np.zeros((items, max_n, max_n, num_edge_features))
    node_masks = np.zeros((items, max_n))
    id = np.eye(max_n_atom)
    for idx, data in enumerate(dataset):
        num_nodes = data.num_nodes
        # Fill in the node features as one-hot encoded atomic numbers
        atom_one_hot = id[data.z.numpy() - 1]
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

    shuffling_indices = random.permutation(x=items, key=seed)
    nodes = nodes[shuffling_indices]
    edges = edges[shuffling_indices]
    node_masks = node_masks[shuffling_indices]
    train_size = int(train_size * items)
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
