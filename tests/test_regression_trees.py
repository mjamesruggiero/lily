import unittest
from context import lily
from lily import regression_trees
import numpy as np

import logging
logging.basicConfig(level=logging.INFO, format="%(lineno)d\t%(message)s")


class TestRegressionTrees(unittest.TestCase):
    def setUp(self):
        data = self._load_test_data('data/ex00.txt')
        self.tree = regression_trees.create_tree(np.mat(data))

    def _load_test_data(self, filepath):
        data_matrix = []
        fr = open(filepath)
        for line in fr.readlines():
            current_line = line.strip().split('\t')
            line_values = map(float, current_line)
            data_matrix.append(line_values)
        return data_matrix

    def test_is_tree_true_case(self):
        """regression_trees - is_tree returns true for dict"""
        self.assertTrue(regression_trees.is_tree({"foo": "bar"}))

    def test_is_tree_false_case(self):
        """regression_trees - is_tree returns false for non-dict"""
        self.assertFalse(regression_trees.is_tree(set((1, 2))))

    def test_integration(self):
        """regression_trees - capture tree result; mostly for refactoring"""
        expected = {
            'left': 1.018096767241379,
            'right': -0.044650285714285733,
            'spInd': 0,
            'spVal': np.matrix([[0.48813]])
        }
        self.assertEqual(expected, self.tree)

if __name__ == '__main__':
    unittest.main()
