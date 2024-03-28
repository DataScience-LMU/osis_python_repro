from typing import Optional

import numpy as np
from nptyping import (
    Float,
    NDArray,
    Shape,
)


class PolynomialDataGenerator:
    def __init__(
        self,
        degree: int = 3,
        coeff_range: tuple[float, float] = (-0.5, 0.5),
        x_range: tuple[float, float] = (-5, 5),
        noise_mean: float = 0,
        noise_std: float = 5,
        rng: Optional[np.random.Generator] = None,
    ):
        self.degree = degree
        self.coeff_range = coeff_range
        self.x_range = x_range

        self.noise_mean = noise_mean
        self.noise_std = noise_std

        self.rng = np.random.default_rng() if rng is None else rng
        self.coefficients = self.rng.uniform(*coeff_range, degree + 1)

    def generate(
        self, num_samples: int
    ) -> tuple[NDArray[Shape['*, *'], Float], NDArray[Shape['*, *'], Float]]:
        x = self.rng.uniform(*self.x_range, (num_samples, 1))

        y = np.polyval(self.coefficients, x)
        y += self.rng.normal(self.noise_mean, self.noise_std, x.shape)

        return x, y

    def __call__(
        self, num_samples: int
    ) -> tuple[NDArray[Shape['*, *'], Float], NDArray[Shape['*, *'], Float]]:
        return self.generate(num_samples)
