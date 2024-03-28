from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from nptyping import (
    Float,
    NDArray,
    Shape,
)

from src.config import ModelType


class RegressionModel(ABC):
    def __init__(self, *args, **kwargs):
        self.weight: Optional[NDArray[Shape['*, *']]] = None

    def fit(
        self, x: NDArray[Shape['*, *'], Float], y: NDArray[Shape['*, *'], Float]
    ) -> RegressionModel:
        new_x = self._modify_input(x)
        self._compute_weight(new_x, y)
        return self

    def transform(
        self, x: NDArray[Shape['*, *'], Float]
    ) -> NDArray[Shape['*, *'], Float]:
        new_x = self._modify_input(x)
        return self._compute_output(new_x)

    def fit_transform(
        self, x: NDArray[Shape['*, *'], Float], y: NDArray[Shape['*, *'], Float]
    ) -> NDArray[Shape['*, *'], Float]:
        self.fit(x, y)
        return self.transform(x)

    @abstractmethod
    def _modify_input(
        self, x: NDArray[Shape['*, *'], Float]
    ) -> NDArray[Shape['*, *'], Float]:
        pass

    @abstractmethod
    def _compute_weight(
        self, x: NDArray[Shape['*, *'], Float], y: NDArray[Shape['*, *'], Float]
    ) -> None:
        pass

    @abstractmethod
    def _compute_output(
        self, x: NDArray[Shape['*, *'], Float]
    ) -> NDArray[Shape['*, *'], Float]:
        pass


class LinearRegressionModel(RegressionModel):
    def _modify_input(
        self, x: NDArray[Shape['*, *'], Float]
    ) -> NDArray[Shape['*, *'], Float]:
        bias = np.ones((x.shape[0], 1), dtype=float)
        return np.concatenate([bias, x], axis=1)

    def _compute_weight(
        self, x: NDArray[Shape['*, *'], Float], y: NDArray[Shape['*, *'], Float]
    ) -> None:
        self.weight = np.linalg.multi_dot([np.linalg.inv(x.T.dot(x)), x.T, y])

    def _compute_output(
        self, x: NDArray[Shape['*, *'], Float]
    ) -> NDArray[Shape['*, *'], Float]:
        assert self.weight is not None, 'Model is not fitted. Call fit() first.'
        return np.dot(x, self.weight)


class PolynomialRegressionModel(LinearRegressionModel):
    def __init__(self, degree: int = 3, *args, **kwargs):
        super().__init__()
        assert degree > 0, 'Degree must be greater than 0.'
        self.degree = degree

    def _modify_input(
        self, x: NDArray[Shape['*, *'], Float]
    ) -> NDArray[Shape['*, *'], Float]:
        transformed_features = []

        for degree in range(1, self.degree + 1):
            transformed_features.append(x**degree)

        x_transformed = np.concatenate(transformed_features, axis=1)
        return super()._modify_input(x_transformed)


def create_model(model_type: ModelType, *args, **kwargs) -> RegressionModel:
    match model_type:
        case ModelType.LINEAR:
            cls = LinearRegressionModel
        case ModelType.POLYNOMIAL:
            cls = PolynomialRegressionModel
        case _:
            raise ValueError(f'Unknown model type: {model_type}')
    return cls(*args, **kwargs)
