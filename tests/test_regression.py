import unittest
from context import lily
from lily import regression, utils
from numpy import array

import logging
logging.basicConfig(level=logging.INFO, format="%(lineno)d\t%(message)s")


class TestRegression(unittest.TestCase):

    def setUp(self):
        self.test_file = "data/test_set.txt"
        self.data_array, self.label_matrix = utils.load_dataset(self.test_file)

    def test_gradient_ascent(self):
        """
        regression - gradient_ascent returns accurate weights
        """
        weights = regression.gradient_ascent(self.data_array,
                                             self.label_matrix)
        weights = [round(e, 2) for e in list(array(weights).reshape(-1,))]

        expected = [4.12, 0.48, -0.62]
        self.assertListEqual(expected, weights)

    def test_stochastic_gradient_ascent(self):
        """
        regression - stochastic_gradient_ascent returns accurate weights
        """
        weights = regression.stochastic_gradient_ascent(array(self.data_array),
                                                        self.label_matrix)
        weights = [round(e, 2) for e in list(array(weights).reshape(-1,))]

        expected = [1.02, 0.86, -0.37]
        self.assertListEqual(expected, weights)

    def test_modified_stochastic_gradient_ascent(self):
        """
        regression - modified_stochastic_gradient_ascent
        returns accurate weights
        """
        weights = regression.modified_stochastic_gradient_ascent(array(self.data_array),
                  self.label_matrix)
        weights = [round(e, 2) for e in list(array(weights).reshape(-1,))]

        # hard to test since the values are random
        # but they distribute this way (roughly)
        self.assertTrue(weights[0] > 10.0)
        self.assertTrue(weights[1] > 0.5)
        self.assertTrue(weights[2] < -1.5)
