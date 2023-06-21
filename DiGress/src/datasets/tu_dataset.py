import os
import os.path as osp
import ipdb
import torch as t
import pathlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import subgraph

import src.utils as utils
from src.datasets.abstract_dataset import MolecularDataModule, AbstractDatasetInfos
from src.analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges
from src.analysis.rdkit_functions import compute_molecular_metrics


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


class RemoveYTransform:
    def __call__(self, data):
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data


class SelectMuTransform:
    def __call__(self, data):
        data.y = data.y[..., :1]
        return data


class SelectHOMOTransform:
    def __call__(self, data):
        data.y = data.y[..., 1:]
        return data


class TUDataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir
        super().__init__(cfg)
        self.remove_h = cfg.dataset.remove_h
        # target = getattr(self.cfg.general, "guidance_target", None)
        # regressor = getattr(self, "regressor", None)
        # if regressor and target == "mu":
        #     transform = SelectMuTransform()
        # elif regressor and target == "homo":
        #     transform = SelectHOMOTransform()
        # elif regressor and target == "both":
        #     transform = None
        # else:
        #     transform = RemoveYTransform()
        #
        # base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        # root_path = os.path.join(base_path, self.datadir)
        # datasets = {
        #     "train": QM9Dataset(
        #         stage="train",
        #         root=root_path,
        #         remove_h=self.cfg.dataset.remove_h,
        #         target_prop=target,
        #         transform=RemoveYTransform(),
        #     ),
        #     "val": QM9Dataset(
        #         stage="val",
        #         root=root_path,
        #         remove_h=self.cfg.dataset.remove_h,
        #         target_prop=target,
        #         transform=RemoveYTransform(),
        #     ),
        #     "test": QM9Dataset(
        #         stage="test",
        #         root=root_path,
        #         remove_h=self.cfg.dataset.remove_h,
        #         target_prop=target,
        #         transform=transform,
        #     ),
        # }
        # super().prepare_data(datasets)
        raw_dataset = TUDataset(
            root=self.datadir, name="PTC_MR", use_node_attr=True, use_edge_attr=True
        )
        n_nodes = {}
        node_features = []
        edge_features = []
        node_counts = []
        edge_counts = []
        for el in raw_dataset:
            n_nodes[el.num_nodes] = n_nodes.get(el.num_nodes, 0) + 1
            node_counts.append(el.num_nodes)
            edge_counts.append(el.num_edges)
            node_features.extend(el.x)
            edge_features.extend(el.edge_attr)
        n_nodes_tensor = t.tensor(list(n_nodes.items()))
        n_nodes_dist = t.zeros(n_nodes_tensor[:, 0].max() + 1)
        n_nodes_dist[n_nodes_tensor[:, 0]] = n_nodes_tensor[:, 1].float()
        self.mean_n_nodes = np.mean(node_counts)
        self.mean_n_edges = np.mean(edge_counts)
        self.n_nodes_dist = n_nodes_dist / n_nodes_dist.sum()
        assert self.n_nodes_dist.sum() == 1 and self.n_nodes_dist.min() >= 0
        self.node_features_dist = t.stack(node_features).mean(dim=0)
        self.edge_features_dist = t.stack(edge_features).mean(dim=0)
        assert (
            torch.allclose(self.node_features_dist.sum(), torch.tensor(1.0))
            and self.node_features_dist.min() >= 0
        )
        assert (
            torch.allclose(self.edge_features_dist.sum(), torch.tensor(1.0))
            and self.edge_features_dist.min() >= 0
        )

        train_size = int(len(raw_dataset) * 0.8)
        train_data = raw_dataset[:train_size]
        val_data = raw_dataset[train_size:]
        test_data = val_data
        super().prepare_data({"train": train_data, "val": val_data, "test": test_data})

    def prepare_data(self):
        pass


class TUInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False):
        self.remove_h = cfg.dataset.remove_h
        self.need_to_strip = (
            False  # to indicate whether we need to ignore one output from the model
        )
        self.name = "TU"
        self.n_nodes = datamodule.n_nodes_dist
        self.node_types = datamodule.node_features_dist
        self.edge_types = datamodule.edge_features_dist
        self.atom_decoder = [str(i) for i in range(len(self.n_nodes))]
        self.atom_encoder = {str(i): i for i in range(len(self.n_nodes))}
        self.valencies = [i + 1 for i in reversed(range(len(self.n_nodes)))]
        self.max_weight = max(self.valencies)
        self.atom_weights = {i: i + 1 for i in range(len(self.n_nodes))}
        self.mean_n_nodes = datamodule.mean_n_nodes
        self.mean_n_edges = datamodule.mean_n_edges
        self.use_bpe = True
        self.input_dims = {
            "X": len(self.node_types),
            "E": len(self.edge_types),
        }
        super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
        # super().compute_input_output_dims(datamodule, )
        # if self.remove_h:
        #     self.atom_encoder = {"C": 0, "N": 1, "O": 2, "F": 3}
        #     self.atom_decoder = ["C", "N", "O", "F"]
        #     self.num_atom_types = 4
        #     self.valencies = [4, 3, 2, 1]
        #     self.atom_weights = {0: 12, 1: 14, 2: 16, 3: 19}
        #     self.max_n_nodes = 9
        #     self.max_weight = 150
        #
        #     self.n_nodes = torch.tensor(
        #         [
        #             0,
        #             2.2930e-05,
        #             3.8217e-05,
        #             6.8791e-05,
        #             2.3695e-04,
        #             9.7072e-04,
        #             0.0046472,
        #             0.023985,
        #             0.13666,
        #             0.83337,
        #         ]
        #     )
        #     self.node_types = torch.tensor([0.7230, 0.1151, 0.1593, 0.0026])
        #     self.edge_types = torch.tensor([0.7261, 0.2384, 0.0274, 0.0081, 0.0])
        #     super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
        #     self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
        #     self.valency_distribution[0:6] = torch.tensor(
        #         [2.6071e-06, 0.163, 0.352, 0.320, 0.16313, 0.00073]
        #     )
        # else:
        #     self.atom_encoder = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
        #     self.atom_decoder = ["H", "C", "N", "O", "F"]
        #     self.valencies = [1, 4, 3, 2, 1]
        #     self.num_atom_types = 5
        #     self.max_n_nodes = 29
        #     self.max_weight = 390
        #     self.atom_weights = {0: 1, 1: 12, 2: 14, 3: 16, 4: 19}
        #     self.n_nodes = torch.tensor(
        #         [
        #             0,
        #             0,
        #             0,
        #             1.5287e-05,
        #             3.0574e-05,
        #             3.8217e-05,
        #             9.1721e-05,
        #             1.5287e-04,
        #             4.9682e-04,
        #             1.3147e-03,
        #             3.6918e-03,
        #             8.0486e-03,
        #             1.6732e-02,
        #             3.0780e-02,
        #             5.1654e-02,
        #             7.8085e-02,
        #             1.0566e-01,
        #             1.2970e-01,
        #             1.3332e-01,
        #             1.3870e-01,
        #             9.4802e-02,
        #             1.0063e-01,
        #             3.3845e-02,
        #             4.8628e-02,
        #             5.4421e-03,
        #             1.4698e-02,
        #             4.5096e-04,
        #             2.7211e-03,
        #             0.0000e00,
        #             2.6752e-04,
        #         ]
        #     )
        #
        #     self.node_types = torch.tensor([0.5122, 0.3526, 0.0562, 0.0777, 0.0013])
        #     self.edge_types = torch.tensor(
        #         [0.88162, 0.11062, 5.9875e-03, 1.7758e-03, 0]
        #     )
        #
        #     super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
        #     self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
        #     self.valency_distribution[0:6] = torch.tensor(
        #         [0, 0.5136, 0.0840, 0.0554, 0.3456, 0.0012]
        #     )
        #
        # if recompute_statistics:
        #     np.set_printoptions(suppress=True, precision=5)
        #     self.n_nodes = datamodule.node_counts()
        #     print("Distribution of number of nodes", self.n_nodes)
        #     np.savetxt("n_counts.txt", self.n_nodes.numpy())
        #     self.node_types = datamodule.node_types()  # There are no node types
        #     print("Distribution of node types", self.node_types)
        #     np.savetxt("atom_types.txt", self.node_types.numpy())
        #
        #     self.edge_types = datamodule.edge_counts()
        #     print("Distribution of edge types", self.edge_types)
        #     np.savetxt("edge_types.txt", self.edge_types.numpy())
        #
        #     valencies = datamodule.valency_count(self.max_n_nodes)
        #     print("Distribution of the valencies", valencies)
        #     np.savetxt("valencies.txt", valencies.numpy())
        #     self.valency_distribution = valencies
        #     assert False
        #


def get_train_smiles(cfg, train_dataloader, dataset_infos, evaluate_dataset=False):
    if evaluate_dataset:
        assert (
            dataset_infos is not None
        ), "If wanting to evaluate dataset, need to pass dataset_infos"
    datadir = cfg.dataset.datadir
    remove_h = cfg.dataset.remove_h
    atom_decoder = dataset_infos.atom_decoder
    root_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
    smiles_file_name = "train_smiles_no_h.npy" if remove_h else "train_smiles_h.npy"
    smiles_path = os.path.join(root_dir, datadir, smiles_file_name)
    if os.path.exists(smiles_path):
        print("Dataset smiles were found.")
        train_smiles = np.load(smiles_path)
    else:
        print("Computing dataset smiles...")
        train_smiles = compute_qm9_smiles(atom_decoder, train_dataloader, remove_h)
        np.save(smiles_path, np.array(train_smiles))

    if evaluate_dataset:
        train_dataloader = train_dataloader
        all_molecules = []
        for i, data in enumerate(train_dataloader):
            dense_data, node_mask = utils.to_dense(
                data.x, data.edge_index, data.edge_attr, data.batch
            )
            dense_data = dense_data.mask(node_mask, collapse=True)
            X, E = dense_data.X, dense_data.E

            for k in range(X.size(0)):
                n = int(torch.sum((X != -1)[k, :]))
                atom_types = X[k, :n].cpu()
                edge_types = E[k, :n, :n].cpu()
                all_molecules.append([atom_types, edge_types])

        print(
            "Evaluating the dataset -- number of molecules to evaluate",
            len(all_molecules),
        )
        metrics = compute_molecular_metrics(
            molecule_list=all_molecules,
            train_smiles=train_smiles,
            dataset_info=dataset_infos,
        )
        print(metrics[0])

    return train_smiles


def compute_qm9_smiles(atom_decoder, train_dataloader, remove_h):
    """

    :param dataset_name: qm9 or qm9_second_half
    :return:
    """
    print(f"\tConverting QM9 dataset to SMILES for remove_h={remove_h}...")

    mols_smiles = []
    len_train = len(train_dataloader)
    invalid = 0
    disconnected = 0
    for i, data in enumerate(train_dataloader):
        dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        dense_data = dense_data.mask(node_mask, collapse=True)
        X, E = dense_data.X, dense_data.E

        n_nodes = [int(torch.sum((X != -1)[j, :])) for j in range(X.size(0))]

        molecule_list = []
        for k in range(X.size(0)):
            n = n_nodes[k]
            atom_types = X[k, :n].cpu()
            edge_types = E[k, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        for l, molecule in enumerate(molecule_list):
            mol = build_molecule_with_partial_charges(
                molecule[0], molecule[1], atom_decoder
            )
            smile = mol2smiles(mol)
            if smile is not None:
                mols_smiles.append(smile)
                mol_frags = Chem.rdmolops.GetMolFrags(
                    mol, asMols=True, sanitizeFrags=True
                )
                if len(mol_frags) > 1:
                    print("Disconnected molecule", mol, mol_frags)
                    disconnected += 1
            else:
                print("Invalid molecule obtained.")
                invalid += 1

        if i % 1000 == 0:
            print(
                "\tConverting QM9 dataset to SMILES {0:.2%}".format(
                    float(i) / len_train
                )
            )
    print("Number of invalid molecules", invalid)
    print("Number of disconnected molecules", disconnected)
    return mols_smiles
