from dacite import from_dict
from dataclasses import dataclass


@dataclass(frozen=True)
class Dimensions:
    X: int
    E: int
    y: int


@dataclass(frozen=True)
class HiddenDimensions:
    dx: int
    de: int
    dy: int
    n_head: int
    dim_ffX: int
    dim_ffE: int
    dim_ffy: int


@dataclass(frozen=True)
class GraphTransformerConfig:
    n_layers: int
    input_dims: Dimensions
    hidden_mlp_dims: Dimensions
    hidden_dims: HiddenDimensions
    output_dims: Dimensions

    # has a custom init method that creates a config object from a dictionary
    @classmethod
    def from_dict(cls, config_dict):
        return from_dict(cls, config_dict)
