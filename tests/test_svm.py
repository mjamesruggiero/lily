import unittest
from context import lily
from lily import svm, optimizer
from numpy import mat

import logging
logging.basicConfig(level=logging.WARNING, format="%(lineno)d\t%(message)s")


class TestSvm(unittest.TestCase):

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
        self.os = optimizer.Optimizer(mat(self.data_matrix),
                                      mat(self.label_matrix).transpose(),
                                      self.C,
                                      self.tolerance)

    def test_calculate_ek(self):
        """svm - calculate_ek calculates E value for a given alpha"""
        for k in range(len(self.test_data)):
            ek = svm.calculate_ek(self.os, k)

        self.assertEqual(ek, -1.0)

    def test_clip_alpha_greater_than_h(self):
        """svm - clip_alpha returns H when alpha greater than H"""
        alpha = 8
        H = 6
        L = 5
        self.assertEqual(svm.clip_alpha(alpha, H, L), 6)

    def test_clip_alpha_less_than_l(self):
        """svm - clip_alpha returns L when alpha less than L"""
        alpha = 8
        H = 6
        L = 7
        self.assertEqual(svm.clip_alpha(alpha, H, L), 7)

    def test_select_j_rand_doesnt_select_i(self):
        """svm - select_j_rand does not select i"""
        i = 4
        m = 76
        self.assertNotEqual(svm.select_j_rand(i, m), i)

    def test_needs_optimization_returns_false_for_low_ei(self):
        """
        svm - needs_optimization returns false for small nonneg ei
        """
        self.assertFalse(svm.needs_optimization(self.os, 5, 0.1))

    def test_needs_optimization_returns_true_for_neg_ei(self):
        """
        svm - needs_optimization returns true for small neg ei
        """
        self.assertTrue(svm.needs_optimization(self.os, 5, -5.1))

if __name__ == '__main__':
    unittest.main()
