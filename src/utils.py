import numpy as np
import yaml


def initialize_rng(seed: int) -> np.random.Generator:
    bit_generator = np.random.PCG64(seed)
    return np.random.Generator(bit_generator)


def read_yml(path: str) -> dict:
    with open(path, 'r') as file:
        return yaml.load(file, Loader=yaml.SafeLoader)
