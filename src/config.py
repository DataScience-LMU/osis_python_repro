"""This module contains the configuration classes."""

from dataclasses import dataclass
from enum import Enum

from src.utils import read_yml


class ModelType(Enum):
    """Enumeration class representing different types of models."""

    LINEAR: str = 'linear'
    POLYNOMIAL: str = 'polynomial'


@dataclass
class RunConfiguration:
    """
    Represent the configuration for a run.

    :param seed: The seed value for random number generation.
    :param num_samples: The number of samples to generate.
    :param model_type: The type of model to use.
    :param polynomial_degree: The degree of the polynomial model.
    """

    seed: int
    num_samples: int
    model_type: ModelType
    polynomial_degree: int


def initialize_config(file_path: str):
    """
    Initialize the configuration for running the application.

    :param file_path: The path to the configuration file.
    :return: The initialized configuration object.
    """
    cfg = read_yml(file_path)
    cfg['model_type'] = ModelType(cfg['model_type'])
    return RunConfiguration(**cfg)
