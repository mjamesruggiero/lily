import unittest
from context import lily
from lily import regression, utils
from numpy import array

import logging
logging.basicConfig(level=logging.ERROR, format="%(lineno)d\t%(message)s")


class TestRegression(unittest.TestCase):

    def setUp(self):
        self.test_file = "data/test_set.txt"

    def test_gradient_ascent(self):
        """
        regression - gradient_ascent returns accurate weights
        """
        data_array, label_matrix = utils.load_dataset(self.test_file)
        weights = regression.gradient_ascent(data_array, label_matrix)
        weights = [round(e, 2) for e in list(array(weights).reshape(-1,))]

        expected = [4.12, 0.48, -0.62]

        self.assertListEqual(expected, weights)
