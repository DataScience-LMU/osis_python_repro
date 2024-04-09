"""Data generation utilities."""

from typing import Optional

import numpy as np
from nptyping import (
    Float,
    NDArray,
    Shape,
)


class PolynomialDataGenerator:
    """
    A class that generates polynomial data with noise.

    :param degree: The degree of the polynomial.
    :type degree: int
    :param coeff_range: The range of coefficients for the polynomial.
    :type coeff_range: tuple[float, float]
    :param x_range: The range of x values.
    :type x_range: tuple[float, float]
    :param noise_mean: The mean of the noise.
    :type noise_mean: float
    :param noise_std: The standard deviation of the noise.
    :type noise_std: float
    :param rng: The random number generator.
    :type rng: Optional[np.random.Generator]

    :ivar degree: The degree of the polynomial.
    :vartype degree: int
    :ivar coeff_range: The range of coefficients for the polynomial.
    :vartype coeff_range: tuple[float, float]
    :ivar x_range: The range of x values.
    :vartype x_range: tuple[float, float]
    :ivar noise_mean: The mean of the noise.
    :vartype noise_mean: float
    :ivar noise_std: The standard deviation of the noise.
    :vartype noise_std: float
    :ivar rng: The random number generator.
    :vartype rng: np.random.Generator
    :ivar coefficients: The coefficients of the polynomial.
    :vartype coefficients: ndarray
    """

    def __init__(
        self,
        degree: int = 3,
        coeff_range: tuple[float, float] = (-0.5, 0.5),
        x_range: tuple[float, float] = (-5, 5),
        noise_mean: float = 0,
        noise_std: float = 5,
        rng: Optional[np.random.Generator] = None,
    ):
        """Initialize the PolynomialDataGenerator class."""
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
        """
        Generate polynomial data with noise.

        :param num_samples: The number of samples to generate.
        :type num_samples: int
        :return: A tuple containing the generated x and y values.
        :rtype: tuple[ndarray, ndarray]
        """
        x = self.rng.uniform(*self.x_range, (num_samples, 1))

        y = np.polyval(self.coefficients, x)
        y += self.rng.normal(self.noise_mean, self.noise_std, x.shape)

        return x, y

    def __call__(
        self, num_samples: int
    ) -> tuple[NDArray[Shape['*, *'], Float], NDArray[Shape['*, *'], Float]]:
        """
        Generate polynomial data with noise.

        :param num_samples: The number of samples to generate.
        :type num_samples: int
        :return: A tuple containing the generated x and y values.
        :rtype: tuple[ndarray, ndarray]
        """
        return self.generate(num_samples)
