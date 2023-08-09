import pickle
import os
import h5py
from tqdm import tqdm
from jaxtyping import Float, Array
from rich import print
import ipdb
from dataclasses import dataclass
import numpy as np
from scipy.signal import filtfilt, savgol_filter
from jax import numpy as jnp
from jaxtyping import Int, Float
from tensorflow.data import Dataset
from torch_geometric.datasets import TUDataset as PyTUDataset
from typing import Iterable
import tensorflow as tf

tf.config.experimental.set_visible_devices([], "GPU")


@dataclass(frozen=True)
class TUDataset:
    train_loader: Iterable
    test_loader: Iterable
    max_node_feature: int
    max_edge_feature: int
    n: int
    nodes_dist: Float[Array, "k"]
    feature_node_prior: Float[Array, "m"]
    feature_edge_prior: Float[Array, "l"]

    @classmethod
    def create(
        cls,
        *,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        nodes: np.ndarray | Array,
        edges: np.ndarray | Array,
        nodes_counts: np.ndarray,
        train_batch_size: int,
        test_batch_size: int,
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
        train_nodes = np.array(train_nodes).astype(float)[..., None]
        train_edges = np.array(train_edges).astype(float)
        test_nodes = np.array(test_nodes).astype(float)[..., None]
        test_edges = np.array(test_edges).astype(float)
        train_nodes = train_nodes.squeeze(-1)
        test_nodes = test_nodes.squeeze(-1)

        # if filter_graphs_by_max_node_count is not None:
        #     print(
        #         f"Filtering graphs by max node count {filter_graphs_by_max_node_count}"
        #     )
        #     len_before_filtering = len(train_nodes)
        #     train_mask = train_nodes_counts <= filter_graphs_by_max_node_count
        #     test_mask = test_nodes_counts <= filter_graphs_by_max_node_count
        #     train_nodes = train_nodes[train_mask, :filter_graphs_by_max_node_count]
        #     train_edges = train_edges[
        #         train_mask,
        #         :filter_graphs_by_max_node_count,
        #         :filter_graphs_by_max_node_count,
        #     ]
        #     train_nodes_counts = train_nodes_counts[train_mask]
        #     train_edges_counts = train_edges_counts[train_mask]
        #     test_nodes = test_nodes[test_mask, :filter_graphs_by_max_node_count]
        #     test_edges = test_edges[
        #         test_mask,
        #         :filter_graphs_by_max_node_count,
        #         :filter_graphs_by_max_node_count,
        #     ]
        #     test_nodes_counts = test_nodes_counts[test_mask]
        #     test_edges_counts = test_edges_counts[test_mask]
        #     train_masks = train_masks[train_mask, :filter_graphs_by_max_node_count]
        #     len_after_filtering = len(train_nodes)
        #     # prints the relative number of graphs that were filtered out
        #     print(
        #         f"Filtered out {(1 - len_after_filtering/len_before_filtering)} of graphs due to max node count"
        #     )

        nodes_dist = compute_distribution(jnp.array(train_masks), margin=5)
        nodes_prior = (
            jnp.array(train_nodes.mean(axis=(0, 1)).squeeze())
            if not train_nodes.shape[-1] == 1
            else jnp.ones(1)
        )
        edges_prior = jnp.array(train_edges.mean(axis=(0, 1, 2)).squeeze())
        train_loader = (
            Dataset.zip(
                tuple(
                    map(
                        Dataset.from_tensor_slices,
                        (
                            train_nodes,
                            train_edges,
                            train_edges_counts,
                            train_nodes_counts,
                        ),
                    )
                )
            )
            .shuffle(1000)
            .batch(train_batch_size)
        )
        test_loader = Dataset.zip(
            tuple(
                map(
                    Dataset.from_tensor_slices,
                    (
                        test_nodes,
                        test_edges,
                        test_edges_counts,
                        test_nodes_counts,
                    ),
                )
            )
        ).batch(test_batch_size)
        return TUDataset(
            train_loader=train_loader,
            test_loader=test_loader,
            max_node_feature=int(np.max(train_nodes)),
            max_edge_feature=int(np.max(train_edges)),
            n=int(max_n),
            nodes_dist=nodes_dist,
            feature_node_prior=nodes_prior,
            feature_edge_prior=edges_prior,
        )


#
# def __to_dense(x, edge_index, edge_attr, batch):
#     x, node_mask = to_dense_batch(x=x, batch=batch)
#     edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
#     # TODO: carefully check if setting node_mask as a bool breaks the continuous case
#     max_num_nodes = x.size(1)
#     e = to_dense_adj(
#         edge_index=edge_index,
#         batch=batch,
#         edge_attr=edge_attr,
#         max_num_nodes=max_num_nodes,
#     )
#     e = __encode_no_edge(e)
#     return x, e, node_mask
#
#
# def __encode_no_edge(E):
#     # assert len(E.shape) == 4
#     if E.shape[-1] == 0:
#         return E
#     no_edge = torch.sum(E, dim=3) == 0
#     first_elt = E[:, :, :, 0]
#     first_elt[no_edge] = 1
#     E[:, :, :, 0] = first_elt
#     diag = (
#         torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
#     )
#     E[diag] = 0
#     return E
#
# def save_cache(cache_location, test_dataset, train_dataset, freqs):
#     with h5py.File(cache_location, "w") as hf:
#         hf.create_dataset("testset", data=test_dataset)
#         hf.create_dataset("trainset", data=train_dataset)
#         hf.create_dataset("freqs", data=freqs)
#
#
# def load_cache(
#     cache_location: str,
# ):  # -> tuple[jdl.ArrayDataset, jdl.ArrayDataset, Array]:
#     with h5py.File(cache_location, "r") as hf:
#         test_dataset = hf["testset"][:]  # type: ignore
#         train_dataset = hf["trainset"][:]  # type: ignore
#         freqs = hf["freqs"][:]  # type: ignore
#     return test_dataset, train_dataset, freqs


def compute_distribution(
    node_masks: Float[Array, "bs m"], margin: int
) -> Float[Array, "size"]:
    per_node_n = np.sum(node_masks, axis=1).astype(int)
    max_n = np.max(per_node_n) + margin
    dist = np.zeros(max_n, dtype=int)
    for n in per_node_n:
        dist[n] += 1
    norm_dist = dist / np.sum(dist)
    # performs a smoothing of the distribution
    norm_dist_smooth = np.clip(np.array(savgol_filter(np.array(norm_dist), 5, 3)), 0, 1)
    # applies a moving average to the distribution
    norm_dist_smooth_2 = np.convolve(norm_dist_smooth, np.ones(3) / 3, mode="same")
    norm_dist_smooth_3 = np.convolve(norm_dist_smooth_2, np.ones(5) / 5, mode="same")
    norm_dist_smooth_3 = norm_dist_smooth_3 / np.sum(norm_dist_smooth_3)
    return jnp.array(norm_dist_smooth_3)


def load_data(
    *,
    save_path: str,
    train_size: float = 0.8,
    train_batch_size: int,
    test_batch_size: int,
    name: str = "PTC_MR",
    filter_graphs_by_max_node_count: int | None = None,
    one_hot: bool = False,
):
    f_name = os.path.join(save_path, f"{name}_{filter_graphs_by_max_node_count}.h5")
    use_attrs = False if "ZINC" in name else True
    if not os.path.exists(f_name):
        print("Processing dataset...")
        dataset = PyTUDataset(
            root=save_path,
            name=name,  # "PTC_MR",  # "MUTAG"
            use_node_attr=use_attrs,
            use_edge_attr=use_attrs,
        )
        items = len(dataset)
        # Get the maximum number of nodes (atoms) in the dataset
        max_n = max(
            [data.num_nodes for data in dataset]  # type: ignore
        )  # one for the absence of a node
        if filter_graphs_by_max_node_count is not None:
            max_n = min(max_n, filter_graphs_by_max_node_count)

        print(f"[red]Max number of nodes:[/red] {max_n}")
        # Get unique atom types
        num_atom_features = dataset[0].x.shape[1] if use_attrs else 1

        num_edge_features = (
            (dataset.num_edge_features + 1) if use_attrs else 2
        )  # one for the absence of an edge

        nodes = np.zeros((items, max_n, num_atom_features))
        edges = np.zeros((items, max_n, max_n, num_edge_features))
        print(f"[orange]Nodes shape:[/orange] {nodes.shape}")
        print(f"[orange]Edges shape:[/orange] {edges.shape}")

        nodes_mask = np.zeros((items, max_n))
        num_nodes_list = np.zeros(items, int)
        edges_counts = np.zeros(items, int)
        for idx, data in enumerate(tqdm(dataset)):  # type: ignore
            if data.num_nodes > max_n:
                continue
            tot_edges = set()
            num_nodes = data.num_nodes
            # Fill in the node features as one-hot encoded atomic numbers
            atom_one_hot = data.x.numpy()[:, :num_atom_features]
            nodes[idx, :num_nodes, :] = atom_one_hot if use_attrs else 1
            num_nodes_list[idx] = num_nodes
            # Fill in the edge features
            edge_indices = data.edge_index.numpy().T
            edge_indices = np.unique(np.sort(edge_indices, axis=-1), axis=0)
            for j, (src, dst) in enumerate(edge_indices):
                assert src < num_nodes and dst < num_nodes, ipdb.set_trace()
                # if src == dst:
                #     continue
                if use_attrs:
                    edges[idx, src, dst, :] = [0] + data.edge_attr[j].tolist()
                    edges[idx, dst, src, :] = [0] + data.edge_attr[j].tolist()
                else:
                    edges[idx, src, dst, 1] = 1
                    edges[idx, dst, src, 1] = 1
                tot_edges.add((src.item(), dst.item()))
                tot_edges.add((dst.item(), src.item()))

            # Fill in the node_masks
            nodes_mask[idx, :num_nodes] = 1
            edges_counts[idx] = len(tot_edges)

        filter_mask = edges_counts > 0
        nodes = nodes[filter_mask]
        edges = edges[filter_mask]
        edges_counts = edges_counts[filter_mask]
        nodes_mask = nodes_mask[filter_mask]
        num_nodes_list = num_nodes_list[filter_mask]
        print(
            f"[red]Filtered {1 - nodes.shape[0]/len(dataset):.4f}% graphs due to max node count[/red]"
        )

        edges[np.where(edges.sum(axis=-1) == 0)] = np.eye(num_edge_features)[0]
        nodes[np.where(nodes.sum(axis=-1) == 0)] = np.eye(num_atom_features)[0]
        if not one_hot:
            nodes = np.argmax(nodes, axis=-1)
            edges = np.argmax(edges, axis=-1)

        num_nodes_list = np.array(num_nodes_list)
        print("Dataset turned into dense numpy arrays")
        num_edge_features = edges.shape[-1]

        if name.lower() in ("mutag", "ptc_mr"):
            print(f"Loading split indices from files...")
            indices_dir = os.path.join(os.path.dirname(__file__), name.lower())
            train_indices = np.array(
                [
                    int(el)
                    for el in open(os.path.join(indices_dir, "train_idx.txt"))
                    .read()
                    .split("\n")
                    if el != ""
                ]
            )
            train_indices = train_indices[filter_mask[train_indices]]
            test_indices = np.array(
                [
                    int(el)
                    for el in open(os.path.join(indices_dir, "test_idx.txt"))
                    .read()
                    .split("\n")
                    if el != ""
                ]
            )
            test_indices = test_indices[filter_mask[test_indices]]
        else:
            shuffling_indices = np.random.permutation(len(nodes))
            train_size = int(train_size * len(nodes))
            train_indices = shuffling_indices[:train_size]
            test_indices = shuffling_indices[train_size:]

        # if name == "ZINC_full":
        #     import jax
        #
        #     # take only the structure
        #     #nodes = nodes[:, :, 0][..., None]
        #     # nodes = jax.nn.one_hot(nodes[:, :, 0], 2)
        #     # edges = np.asarray(jax.nn.one_hot(edges[:, :, :, 0], 2))
        #     yes = np.array([1, 0])
        #     no = np.array([0, 1])
        #     nodes = np.where(
        #         nodes_mask[..., None], no[None, None], yes[None, None]
        #     ).astype(float)
        #     new_edges = np.where(
        #         (self.edges_mask & (self.edges[..., 0] < 0.5))[..., None],
        #         no[None, None, None],
        #         yes[None, None, None],
        #     ).astype(float)
        #
        # ipdb.set_trace()

        cache = dict(
            train_indices=train_indices,
            test_indices=test_indices,
            nodes=nodes,
            edges=edges,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            edges_counts=np.asarray(edges_counts),
            nodes_counts=num_nodes_list,
            node_masks=nodes_mask,
        )
        # with open(f_name, "wb") as f:
        #     pickle.dump(cache, f)
        # saves it as a h5 file
        # with h5py.File(f_name, "w") as f:
        #     for k, v in cache.items():
        #         f.create_dataset(k, data=v)

        with h5py.File(f_name, "w") as f:
            for k, v in cache.items():
                if (
                    not isinstance(v, int) and not isinstance(v, float) and v.ndim > 0
                ):  # Not a scalar dataset
                    f.create_dataset(
                        k, data=v, compression="gzip", compression_opts=9, chunks=True
                    )
                else:
                    f.create_dataset(k, data=v)
        print(f"Saved dataset to {f_name}")
    else:
        print(f"Loading dataset from {f_name}")
        # with open(f_name, "rb") as f:
        #     cache = pickle.load(f)

        with h5py.File(f_name, "r") as f:
            cache = dict()
            for k in f.keys():
                cache[k] = np.asarray(f[k])
        cache = cache | dict(
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
        )
    return TUDataset.create(**cache)  # type: ignore


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
