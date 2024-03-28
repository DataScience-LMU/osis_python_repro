from dataclasses import dataclass
from enum import Enum

from src.utils import read_yml


class ModelType(Enum):
    LINEAR: str = 'linear'
    POLYNOMIAL: str = 'polynomial'


@dataclass
class RunConfiguration:
    seed: int
    num_samples: int
    model_type: ModelType
    polynomial_degree: int


def initialize_config(file_path: str) -> RunConfiguration:
    cfg = read_yml(file_path)
    cfg['model_type'] = ModelType(cfg['model_type'])
    return RunConfiguration(**cfg)
