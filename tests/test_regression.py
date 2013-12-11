import unittest
from context import lily
from lily import regression
import os
from numpy import array

import logging
logging.basicConfig(level=logging.ERROR, format="%(lineno)d\t%(message)s")


class TestRegression(unittest.TestCase):

    def load_dataset(self):
        """
        TODO file access methods should be extracted
        to utility class
        """
        data_matrix = []
        label_matrix = []

        root_path = os.path.abspath(os.path.join(os.curdir, os.pardir))
        file_path = "{0}/data/test_set.txt".format(root_path)
        fr = open(file_path)
        for line in fr.readlines():
            line_array = line.strip().split()
            data_matrix.append([1.0, float(line_array[0]),
                                float(line_array[1])])
            label_matrix.append(int(line_array[2]))
        return data_matrix, label_matrix

    def test_gradient_ascent(self):
        """
        regression - gradient_ascent returns accurate weights
        """
        data_array, label_matrix = self.load_dataset()
        weights = regression.gradient_ascent(data_array, label_matrix)
        weights = [round(e, 2) for e in list(array(weights).reshape(-1,))]

        expected = [4.12, 0.48, -0.62]

        self.assertListEqual(expected, weights)
