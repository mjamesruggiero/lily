import unittest
from context import lily
import numpy as np

class TestLily(unittest.TestCase):
    def test_classify_0(self):
        """
        lily - classify_0 returns the majority class as the prediction
        """
        large_shape = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
        labels = ['foo', 'bar', 'bz', 'quux']
        result = lily.classify_0(large_shape[1, :], large_shape[2, :], labels, 2)
        assert result == 'foo'

    def create_dataset(self):
        """sample data for shannon entropy test"""
        data_set = [[1, 1, 'yes'],
                    [1, 1, 'yes'],
                    [1, 0, 'no'],
                    [0, 1, 'no'],
                    [0, 1, 'no']]
        labels = ['no surfacing', 'flippers']
        return data_set, labels


    def test_shannon_entropy(self):
        """
        lily - calculate_shannon_entropy returns the shannon entropy
        """
        my_data, labels = self.create_dataset()
        se = lily.calculate_shannon_entropy(my_data)
        self.assertEqual(se, 0.9709505944546686)


    def test_split_dataset(self):
        """
        lily - split_data_set breaks feature vectors on desired vals
        """
        my_data, labels = self.create_dataset()
        returned = lily.split_data_set(my_data, 0, 1)
        self.assertEqual([[1, 'yes'], [1, 'yes'], [0, 'no']], returned)


    def test_choose_best_feature_to_split(self):
        """
        lily - choose_best_feature_to_split discovers best feature
        """
        my_data, labels = self.create_dataset()
        returned = lily.choose_best_feature_to_split(my_data)
        self.assertEqual(0, returned)


if __name__ == '__main__':
    unittest.main()

