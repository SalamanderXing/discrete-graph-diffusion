from dataclasses import dataclass

from jax import Array


@dataclass(frozen=True)
class Dimensions:
    X: int
    E: int
    y: int


@dataclass(frozen=True)
class DatasetInfos:
    in_dims: Dimensions
    out_dims: Dimensions
    nodes_dist: str
    node_types: Array
    edge_types: Array


@dataclass(frozen=True)
class TrainConfig:
    diffusion_steps: int
    diffusion_noise_schedule: str
    learning_rat: float
    lambda_train: float
    transition: str
    learning_rate: float
    number_chain_steps: int
    log_every_steps: int


@dataclass(frozen=True)
class GeneralConfig:
    dataset: DatasetInfos
    train: TrainConfig
