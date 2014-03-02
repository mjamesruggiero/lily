import unittest
from context import lily
from lily import svd
import numpy as np

import logging
logging.basicConfig(level=logging.INFO, format="%(lineno)d\t%(message)s")


class TestSvd(unittest.TestCase):
    def setUp(self):
        self.dataset = [
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
        ]
        self.data_matrix = np.mat(self.dataset)

    def test_euclidean_similarity(self):
        """
        svd - euclidean similarity
        """
        euclidean_sim = svd.euclidean_similarity(self.data_matrix[:, 0],
                                                 self.data_matrix[:, 4])
        self.assertEqual(euclidean_sim, 0.13367660240019172)

    def test_pearson_similarity(self):
        """
        svd - pearson similarity
        """
        pearson_sim = svd.pearson_similarity(self.data_matrix[:, 0],
                                             self.data_matrix[:, 4])
        self.assertEqual(pearson_sim, 0.23768619407595826)

    def test_cosine_similarity(self):
        """
        svd - cosine similarity
        """
        cosine_sim = svd.cosine_similarity(self.data_matrix[:, 0],
                                           self.data_matrix[:, 4])
        self.assertEqual(cosine_sim, 0.54724555912615336)

    def test_estimated_rating(self):
        """
        svd - estimated rating with euclidean distance
        """
        returned = svd.estimated_rating(self.data_matrix,
                                        3,
                                        svd.euclidean_similarity,
                                        4)
        self.assertEqual(returned, [5.0])

if __name__ == '__main__':
    unittest.main()
