import unittest
from context import lily
from lily import regression, utils
import os
from numpy import array

import logging
logging.basicConfig(level=logging.ERROR, format="%(lineno)d\t%(message)s")


class TestRegression(unittest.TestCase):

    def load_dataset(self):
        root_path = os.path.abspath(os.path.join(os.curdir, os.pardir))
        file_path = "{0}/data/test_set.txt".format(root_path)
        return utils.load_dataset(file_path)

    def test_gradient_ascent(self):
        """
        regression - gradient_ascent returns accurate weights
        """
        data_array, label_matrix = self.load_dataset()
        weights = regression.gradient_ascent(data_array, label_matrix)
        weights = [round(e, 2) for e in list(array(weights).reshape(-1,))]

        expected = [4.12, 0.48, -0.62]

        self.assertListEqual(expected, weights)
