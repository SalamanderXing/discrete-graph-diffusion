"""
Contains all dataclasses used to store the configuration of the training.
"""

from dataclasses import dataclass
from jax import Array
from typing import Any
from dacite import from_dict


@dataclass(frozen=True)
class DatasetInfos:
    remove_h: bool
    need_to_strip: bool
    atom_encoder: dict[str, int]
    valencies: list[int]
    atom_weights: dict[int, int]
    max_n_nodes: int
    max_weight: int
    n_nodes: Array
    node_types: Array
    edge_types: Array
    num_classes: int
    nodes_dist: Any
    valency_distribution: Array


@dataclass(frozen=True)
class TrainingConfig:
    diffusion_steps: int
    diffusion_noise_schedule: str
    learning_rate: float
    lambda_train: tuple[float, float]
    transition: str
    number_chain_steps: int
    log_every_steps: int
    max_num_nodes: int
    num_edge_features: int
    num_node_features: int
    dataset: DatasetInfos | None = None

    @classmethod
    def from_dict(cls, config_dict):
        return from_dict(cls, config_dict)
