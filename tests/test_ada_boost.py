import unittest
from context import lily
from lily import ada_boost
import numpy as np

import logging
logging.basicConfig(level=logging.INFO, format="%(lineno)d\t%(message)s")


class TestAdaBoost(unittest.TestCase):

    def setUp(self):
        self.data_matrix = np.matrix([
            [1.0, 2.2],
            [2.0, 1.1],
            [2.0, 1.0],
            [1.0, 2.0]])
        self.labels = [1.0, -1.0, -1.0, 1.0]
        self.larger_matrix = np.matrix([[1., 2.1],
                                        [2., 1.1],
                                        [1.3, 1.],
                                        [1., 1.],
                                        [2., 1.]])
        self.larger_class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]

    def test_ada_boost_stump_classify_partitions_lt(self):
        """ada_boost - stump_classify partitions on less-than"""
        i = 1
        range_min = self.data_matrix[:, i].min()
        threshold = (range_min * 2)
        inequal = 'lt'
        returned = ada_boost.stump_classify(self.data_matrix,
                                            i,
                                            threshold,
                                            inequal)
        expected = np.mat([1.0, -1.0, -1.0, -1.0])

        delta_between_elements = returned - expected.T
        self.assertFalse(delta_between_elements.any())

    def test_ada_boost_stump_classify_partitions_gt(self):
        """ada_boost - stump_classify partitions on greater-than"""
        i = 1
        range_min = self.data_matrix[:, i].min()
        threshold = (range_min * 2)
        inequal = 'gt'
        returned = ada_boost.stump_classify(self.data_matrix,
                                            i,
                                            threshold,
                                            inequal)
        expected = np.mat([-1.0, 1.0, 1.0, 1.0])

        delta_between_elements = returned - expected.T
        self.assertFalse(delta_between_elements.any())

    def test_aggregated_error_rate(self):
        """ada_boost - aggregated error rate returns... error rate"""
        estimates = np.mat([0.8, 0.4, 0.8, 0.4])
        m = np.shape(self.data_matrix)[0]
        returned = ada_boost.aggregated_error_rate(estimates, self.labels, m)
        self.assertEqual(returned, 2.0)

    def test_build_stump(self):
        """ada_boost - build stump finds the best decision stump"""
        D = np.mat(np.ones((5, 1)) / 5)
        best, min_err, best_estimate =\
            ada_boost.build_stump(self.larger_matrix,
                                  self.larger_class_labels,
                                  D)
        expected = {'threshold': 1.3, 'dim': 0, 'inequal': 'lt'}
        self.assertEqual(best, expected)

    def test_train_dataset(self):
        """ada_boost - train_dataset trains the dataset
        and returns estimates"""
        classifiers, estimates =\
            ada_boost.train_dataset(self.larger_matrix,
                                    self.larger_class_labels,
                                    9)
        expected = [
            {'alpha': 0.6931471805599453,
             'dim': 0,
             'inequal': 'lt',
             'threshold': 1.3},
            {'alpha': 0.9729550745276565,
             'dim': 1,
             'inequal': 'lt',
             'threshold': 1.0},
            {'alpha': 0.8958797346140273,
             'dim': 0,
             'inequal': 'lt',
             'threshold': 0.90000000000000002}
        ]
        self.assertEqual(classifiers, expected)

if __name__ == '__main__':
    unittest.main()
