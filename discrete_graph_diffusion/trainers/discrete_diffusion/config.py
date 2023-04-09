from dataclasses import dataclass
from jax import Array

from dacite import from_dict

@dataclass(frozen=True)
class Dimensions:
    X: int
    E: int
    y: int


@dataclass(frozen=True)
class DatasetInfos:
    in_dims: Dimensions
    out_dims: Dimensions
    node_types: Array
    edge_types: Array
class DatasetInfos:
    remove_h: bool
    need_to_strip:bool
    atom_encoder:dict[str, int]
    valencies:list[int]
    atom_weights:dict[str, float]
    max_n_nodes:int
    max_weight:int
    n_nodes:Array
    node_types:Array
    edge_types:Array
    num_classes:int
    nodes_dist', 
    valency_distribution'

@dataclass(frozen=True)
class TrainingConfig:
    diffusion_steps: int
    diffusion_noise_schedule: str
    learning_rate: float
    lambda_train: tuple[float, float]
    transition: str
    number_chain_steps: int
    log_every_steps: int
    dataset: DatasetInfos

    @classmethod
    def from_dict(cls, config_dict):
        return from_dict(cls, config_dict)
