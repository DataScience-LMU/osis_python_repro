import unittest

import numpy as np

from src.model import LinearRegressionModel


class TestLinearRegressionModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.x = np.random.rand(100, 1)
        self.y = 2 * self.x + np.random.randn(100, 1) * 0.005

    def tearDown(self):
        pass

    def test_my_method(self):
        model = LinearRegressionModel()
        model.fit(self.x, self.y)
        y_pred = model.transform(self.x)
        self.assertEqual(y_pred.shape, self.y.shape)
        self.assertAlmostEqual(y_pred.mean(), self.y.mean(), places=1)
        self.assertAlmostEqual(model.weight[1].item(), 2, places=2)
        self.assertAlmostEqual(model.weight[0].item(), 0, places=2)


if __name__ == '__main__':
    unittest.main()
