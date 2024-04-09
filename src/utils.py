"""Utility functions for the project."""

import numpy as np
import yaml


def initialize_rng(seed: int) -> np.random.Generator:
    """
    Initialize a random number generator.

    :param seed: The seed value for random number generation.
    :type seed: int
    :return: The random number generator.
    :rtype: np.random.Generator
    """
    bit_generator = np.random.PCG64(seed)
    return np.random.Generator(bit_generator)


def read_yml(path: str) -> dict:
    """
    Read a YAML file.

    :param path: The path to the YAML file.
    :type path: str
    :return: The data from the YAML file.
    :rtype: dict
    """
    with open(path, 'r') as file:
        return yaml.load(file, Loader=yaml.SafeLoader)
