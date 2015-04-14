import unittest
from context import lily
from lily import k_means
from lily import utils
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

    def test_k_means(self):
        """k_means - k means should build clusters"""
        data_matrix = np.mat(utils.load_tsv_into_array('data/test_set_3.txt'))
        centroids, cluster_assignment = k_means.k_means(data_matrix, 4)
        expected = np.mat(np.array([[1., 0.45675494],
                              [0., 0.3032197],
                              [3., 1.74481454],
                              [1., 0.80407696],
                              [0., 1.02508049],
                              [3., 2.59648559],
                              [1., 0.42859499],
                              [0., 0.0305198],
                              [3., 2.37924609],
                              [2., 0.],
                              [0., 5.38984416],
                              [3., 0.04519236],
                              [1., 1.23757291],
                              [0., 0.01298907],
                              [3., 3.28350116],
                              [1., 2.33205513],
                              [0., 3.72839989],
                              [3., 0.1398885],
                              [1., 0.03288099],
                              [0., 0.4038706],
                              [3., 1.00363352],
                              [1., 1.16346981],
                              [0., 0.93928783],
                              [3., 0.02261741],
                              [1., 3.42458409],
                              [0., 5.92927609],
                              [3., 0.98873759],
                              [1., 1.83018987],
                              [0., 0.91125974],
                              [3., 1.28677032]]))

        self.assertEqual(cluster_assignment.any(), expected.any())

if __name__ == '__main__':
    unittest.main()
