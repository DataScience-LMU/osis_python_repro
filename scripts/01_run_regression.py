import fire
import matplotlib.pyplot as plt
import numpy as np

from src.config import initialize_config
from src.data import PolynomialDataGenerator
from src.model import create_model
from src.utils import initialize_rng


def run_classifier(cfg_path: str = 'config.yml') -> None:
    # Create configuration object
    cfg = initialize_config(cfg_path)

    # Initialize RNG object
    rng = initialize_rng(cfg.seed)

    # Create data
    x, y = PolynomialDataGenerator(rng=rng).generate(num_samples=cfg.num_samples)

    # Create & fit classifier
    model = create_model(cfg.model_type, degree=cfg.polynomial_degree)
    model.fit(x, y)

    # Plot approximated function
    func_in = np.linspace(-5, 5, 100)[:, np.newaxis]
    func_out = model.transform(func_in)
    plt.scatter(x, y)
    plt.plot(func_in, func_out, c='red')
    plt.show()


if __name__ == '__main__':
    fire.Fire(run_classifier)
