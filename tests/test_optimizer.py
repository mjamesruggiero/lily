import unittest
from context import lily
from lily import optimizer
from numpy import mat, zeros

import logging
logging.basicConfig(level=logging.INFO, format="%(lineno)d\t%(message)s")


class TestOptimizer(unittest.TestCase):

    def setUp(self):
        self.tolerance = 0.001
        self.C = 0.6
        self.iterations = 40
        self.data_matrix = []
        self.label_matrix = []
        self.test_data = [[-0.017612, 14.053064, 0],
                          [-1.395634, 4.662541, 1],
                          [-0.752157, 6.538620, 0],
                          [-1.322371, 7.152853, 0],
                          [0.423363, 11.054677, 0],
                          [0.406704, 7.067335, 1],
                          [0.667394, 12.741452, 0],
                          [-2.460150, 6.866805, 1],
                          [0.569411, 9.548755, 0],
                          [-0.026632, 10.427743, 0],
                          [0.850433, 6.920334, 1],
                          [1.347183, 13.175500, 0],
                          [1.176813, 3.167020, 1],
                          [-1.781871, 9.097953, 0],
                          [-0.566606, 5.749003, 1]]
        for line in self.test_data:
            self.data_matrix.append([float(line[0]), float(line[1])])
            self.label_matrix.append(float(line[2]))
        kernel = ('lin', 1.3)
        self.os = optimizer.Optimizer(mat(self.data_matrix),
                                      mat(self.label_matrix).transpose(),
                                      self.C,
                                      self.tolerance,
                                      kernel)

    def test_optimizer(self):
        """
        optimizer - optimizer stores things for convenience
        """
        self.assertTrue(self.os.C == self.C)
        self.assertTrue(self.os.tolerance == self.tolerance)

        delta_between_label_matrices = self.os.label_matrix -\
            mat(self.label_matrix).transpose()
        self.assertFalse(delta_between_label_matrices.any())
        self.assertTrue(self.os.m == 15)

        delta_between_alphas = self.os.alphas - mat(zeros((15, 1)))
        self.assertFalse(delta_between_alphas.any())

        delta_between_caches = self.os.e_cache - mat(zeros((15, 2)))
        self.assertFalse(delta_between_caches.any())
        self.assertEqual(self.os.b, 0)

if __name__ == '__main__':
    unittest.main()
