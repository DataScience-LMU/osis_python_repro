"""Test the RegressionModel classes."""

import unittest

import numpy as np

from src.model import LinearRegressionModel, PolynomialRegressionModel


class TestLinearRegressionModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.x = np.random.rand(100, 1)
        self.y = 2 * self.x + np.random.randn(100, 1) * 0.005

    def tearDown(self):
        pass

    def test_regression_fitting(self):
        model = LinearRegressionModel()
        model.fit(self.x, self.y)
        y_pred = model.transform(self.x)
        self.assertEqual(y_pred.shape, self.y.shape)
        self.assertAlmostEqual(y_pred.mean(), self.y.mean(), places=1)
        self.assertAlmostEqual(model.weight[1].item(), 2, places=2)
        self.assertAlmostEqual(model.weight[0].item(), 0, places=2)


class TestPolynomialRegressionModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.x = np.random.rand(1000, 1)
        self.y = (
            2 * self.x
            + 5 * self.x**2
            + 3 * self.x**3
            + 4 * self.x**4
            + np.random.randn(1000, 1) * 0.005
        )

    def tearDown(self):
        pass

    def test_polynomial_regression_fitting(self):
        model = PolynomialRegressionModel(degree=4)
        model.fit(self.x, self.y)
        y_pred = model.transform(self.x)
        self.assertEqual(y_pred.shape, self.y.shape)
        self.assertAlmostEqual(y_pred.mean(), self.y.mean(), places=1)
        self.assertAlmostEqual(model.weight[0].item(), 0, places=0)
        self.assertAlmostEqual(model.weight[1].item(), 2, places=0)
        self.assertAlmostEqual(model.weight[2].item(), 5, places=0)
        self.assertAlmostEqual(model.weight[3].item(), 3, places=0)
        self.assertAlmostEqual(model.weight[4].item(), 4, places=0)


if __name__ == '__main__':
    unittest.main()
