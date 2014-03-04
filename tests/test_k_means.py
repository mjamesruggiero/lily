import unittest
from context import lily
from lily import k_means
import numpy as np
import mock

import logging
logging.basicConfig(level=logging.INFO, format="%(lineno)d\t%(message)s")


class TestKmeans(unittest.TestCase):
    def setUp(self):
        self.dataset = [
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0]
        ]
        self.close_dataset = [
            [0, 0, 2, 0, 0],
            [0, 0, 1, 0, 0]
        ]

    def test_euclidean_distance(self):
        """k_means - euclidean_distance gives correct value"""
        returned = k_means.euclidean_distance(np.mat(self.dataset[0]),
                                              np.mat(self.dataset[1]))
        self.assertEqual(returned, 5.196152422706632)

    def test_euclidean_distance_counts_close_vectors(self):
        """k_means - euclidean_distance gives correct close value"""
        returned = k_means.euclidean_distance(np.mat(self.close_dataset[0]),
                                              np.mat(self.close_dataset[1]))
        self.assertEqual(returned, 1.0)

    @mock.patch('lily.k_means.np.random')
    def test_random_centroid(self, mock_rand):
        """k_means - random_centroid returns set of random centroids"""
        mock_rand.rand.return_value = .5
        expected = np.mat(
            [
                [3.5, 3.5, 3.5, 0., 0.],
                [3.5, 3.5, 3.5, 0., 0.]]
        )

        rand_cent = k_means.random_centroid(np.mat(self.dataset), 2)
        self.assertEqual(rand_cent.all(), expected.all())


if __name__ == '__main__':
    unittest.main()
