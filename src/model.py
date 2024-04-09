"""Module to define regression models."""

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
    """Abstract base class for regression models."""

    def __init__(self, *args, **kwargs):
        """
        Initialize the RegressionModel class.

        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        """
        self.weight: Optional[NDArray[Shape['*, *']]] = None

    def fit(
        self, x: NDArray[Shape['*, *'], Float], y: NDArray[Shape['*, *'], Float]
    ) -> RegressionModel:
        """
        Fit the regression model to the training data.

        Parameters:
        :param x: Input features as a 2D array-like object.
        :param y: Target values as a 2D array-like object.

        Returns:
        :return: The fitted regression model.
        """
        new_x = self._modify_input(x)
        self._compute_weight(new_x, y)
        return self

    def transform(
        self, x: NDArray[Shape['*, *'], Float]
    ) -> NDArray[Shape['*, *'], Float]:
        """
        Transform the input features using the fitted regression model.

        Parameters:
        :param x: Input features as a 2D array-like object.

        Returns:
        :return transformed_x: Transformed input features as a 2D array-like object.
        """
        new_x = self._modify_input(x)
        return self._compute_output(new_x)

    def fit_transform(
        self, x: NDArray[Shape['*, *'], Float], y: NDArray[Shape['*, *'], Float]
    ) -> NDArray[Shape['*, *'], Float]:
        """
        Fit the regression model to the training data and transform the input features.

        Parameters:
        :param x: Input features as a 2D array-like object.
        :param y: Target values as a 2D array-like object.

        Returns:
        :return transformed_x: Transformed input features as a 2D array-like object.
        """
        self.fit(x, y)
        return self.transform(x)

    @abstractmethod
    def _modify_input(
        self, x: NDArray[Shape['*, *'], Float]
    ) -> NDArray[Shape['*, *'], Float]:
        """
        Abstract method to modify the input features.

        Parameters:
        :param x: Input features as a 2D array-like object.

        Returns:
        :return modified_x: Modified input features as a 2D array-like object.
        """
        pass

    @abstractmethod
    def _compute_weight(
        self, x: NDArray[Shape['*, *'], Float], y: NDArray[Shape['*, *'], Float]
    ) -> None:
        """
        Abstract method to compute the weight of the regression model.

        Parameters:
        :param x: Input features as a 2D array-like object.
        :param y: Target values as a 2D array-like object.
        """
        pass

    @abstractmethod
    def _compute_output(
        self, x: NDArray[Shape['*, *'], Float]
    ) -> NDArray[Shape['*, *'], Float]:
        """
        Abstract method to compute the output of the regression model.

        Parameters:
        :param x: Input features as a 2D array-like object.

        Returns:
        :return output: Output of the regression model as a 2D array-like object.
        """
        pass


class LinearRegressionModel(RegressionModel):
    """
    Linear regression model for predicting continuous values.

    This class inherits from the `RegressionModel` base class.

    Methods:
        _modify_input: Modifies the input data by adding a bias column.
        _compute_weight: Computes the weight vector using the normal equation.
        _compute_output: Computes the predicted output values.

    Attributes:
        weight: The weight vector learned during training.

    """

    def _modify_input(
        self, x: NDArray[Shape['*, *'], Float]
    ) -> NDArray[Shape['*, *'], Float]:
        """
        Modifiy the input data by adding a bias column.

        Parameters:
        :param x: The input data.

        Returns:
        :return modified_x: The modified input data with a bias column added.

        """
        bias = np.ones((x.shape[0], 1), dtype=float)
        return np.concatenate([bias, x], axis=1)

    def _compute_weight(
        self, x: NDArray[Shape['*, *'], Float], y: NDArray[Shape['*, *'], Float]
    ) -> None:
        """
        Compute the weight vector using the normal equation.

        Parameters:
        :param x: The input data.
        :param y: The target values.

        Returns:
        :return: None. Updates the `weight` attribute of the model.

        """
        self.weight = np.linalg.multi_dot([np.linalg.inv(x.T.dot(x)), x.T, y])

    def _compute_output(
        self, x: NDArray[Shape['*, *'], Float]
    ) -> NDArray[Shape['*, *'], Float]:
        """
        Compute the predicted output values.

        Parameters:
        :param x: The input data.

        Returns:
        :return: The predicted output values.

        Raises:
        :raises AssertionError: If the model is not fitted. Call fit() first.

        """
        assert self.weight is not None, 'Model is not fitted. Call fit() first.'
        return np.dot(x, self.weight)


class PolynomialRegressionModel(LinearRegressionModel):
    """
    A polynomial regression model that extends the LinearRegressionModel class.

    Parameters:
    :param degree (int): The degree of the polynomial regression model. Must be greater
        than 0.

    Methods:
    :return _modify_input(x: NDArray[Shape['*, *'], Float]) ->
        NDArray[Shape['*, *'], Float]:
        Modifies the input features by transforming them into polynomial features.

    """

    def __init__(self, degree: int = 3, *args, **kwargs):
        """Initialize the PolynomialRegressionModel class."""
        super().__init__()
        assert degree > 0, 'Degree must be greater than 0.'
        self.degree = degree

    def _modify_input(
        self, x: NDArray[Shape['*, *'], Float]
    ) -> NDArray[Shape['*, *'], Float]:
        """
        Modifiy the input features by transforming them into polynomial features.

        Parameters:
        :param x (NDArray[Shape['*, *'], Float]): The input features.

        Returns:
        :return NDArray[Shape['*, *'], Float]: The transformed input features.

        """
        transformed_features = []

        for degree in range(1, self.degree + 1):
            transformed_features.append(x**degree)

        x_transformed = np.concatenate(transformed_features, axis=1)
        return super()._modify_input(x_transformed)


def create_model(model_type: ModelType, *args, **kwargs) -> RegressionModel:
    """
    Create a regression model based on the given model type.

    Parameters:
    :param model_type: The type of regression model to create.
    :type model_type: ModelType
    :param args: Variable length argument list.
    :param kwargs: Arbitrary keyword arguments.

    Returns:
    :return: An instance of the created regression model.
    :rtype: RegressionModel

    Raises:
    :raises ValueError: If the given model type is unknown.
    """
    match model_type:
        case ModelType.LINEAR:
            cls = LinearRegressionModel
        case ModelType.POLYNOMIAL:
            cls = PolynomialRegressionModel
        case _:
            raise ValueError(f'Unknown model type: {model_type}')
    return cls(*args, **kwargs)
