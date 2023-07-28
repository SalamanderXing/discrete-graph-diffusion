from . import qm9_dataset
import ipdb
import jax.numpy as np
from . import utils
from dataclasses import dataclass
from ...shared.graph import graph_distribution as gd, graph
from mate.jax import SFloat, SInt
from typing import Iterable
from jaxtyping import Float, Array


class Bunch(dict):
    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = Bunch(value)
        self.__dict__ = self

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    def __setattr__(self, key, value):
        self[key] = value


def create_graph(
    x,
):  # nodes, edges, edges_counts, nodes_counts, onehot: bool, train=False):
    dense_data, mask = utils.to_dense(x.x, x.edge_index, x.edge_attr, x.batch)
    nodes = np.asarray(dense_data.X.numpy())
    edges = np.asarray(dense_data.E.numpy())
    nodes_mask = np.asarray(mask.numpy())

    return gd.create_one_hot_minimal(
        nodes=nodes,
        edges=edges,
        nodes_mask=nodes_mask,
    )


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
    infos: qm9_dataset.QM9infos
    mean_edge_count: SFloat = 0.0
    mean_node_count: SFloat = 0.0
    var_edge_count: SFloat = 0.0
    var_node_count: SFloat = 0.0
    train_smiles: Iterable[str] = None


def load_data(save_dir: str, batch_size: int):
    cfg = Bunch(
        **{
            "dataset": {
                "datadir": save_dir,
                "remove_h": True,
            },
            "general": {"name": "QM9"},
            "train": {
                "batch_size": batch_size,
                "num_workers": 0,
            },
        }
    )
    datamodule = qm9_dataset.QM9DataModule(cfg)
    dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
    datamodule.prepare_data()
    # train_smiles = qm9_dataset.get_train_smiles(
    #     cfg=cfg,
    #     train_dataloader=datamodule.train_dataloader(),
    #     dataset_infos=dataset_infos,
    #     evaluate_dataset=False,
    # )
    nodes_dist = np.array(dataset_infos.n_nodes.numpy())
    datamodule.__dict__["nodes_dist"] = nodes_dist
    train_smiles = qm9_dataset.get_train_smiles(
        cfg=cfg,
        train_dataloader=datamodule.train_dataloader(),
        dataset_infos=dataset_infos,
        evaluate_dataset=False,
    )

    return QM9Dataset(
        train_loader=datamodule.dataloaders["train"],
        test_loader=datamodule.dataloaders["val"],
        max_node_feature=-1,
        max_edge_feature=-1,
        n=dataset_infos.max_n_nodes,
        nodes_dist=nodes_dist,
        node_prior=np.array(dataset_infos.node_types.numpy()),
        edge_prior=np.array(dataset_infos.edge_types.numpy()),
        mean_edge_count=0.0,
        mean_node_count=0.0,
        var_edge_count=0.0,
        var_node_count=0.0,
        train_smiles=train_smiles,
        infos=dataset_infos,
    )
