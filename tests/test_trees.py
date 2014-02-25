import unittest
from context import lily
from lily import trees
import os

import logging
logging.basicConfig(level=logging.ERROR, format="%(lineno)d\t%(message)s")


class TestTrees(unittest.TestCase):

    def setUp(self):
        data_file = os.path.join('data', 'lenses.txt')
        fr = open(data_file)
        self.lenses = [inst.strip().split('\t') for inst in fr.readlines()]
        self.lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']

    def test_create_tree(self):
        """
        trees - integration test for lenses set
        """
        lensesTree = trees.create_tree(self.lenses, self.lenses_labels)
        logging.info("the lenses tree is {0}".format(lensesTree))

        expected = {
            'tearRate': {
                'reduced': 'no lenses',
                'normal': {
                    'astigmatic': {
                        'yes': {
                            'prescript': {
                                'hyper': {
                                    'age': {
                                        'pre': 'no lenses',
                                        'presbyopic': 'no lenses',
                                        'young': 'hard'}},
                                'myope': 'hard'}
                        },
                        'no': {
                            'age': {
                                'pre': 'soft',
                                'presbyopic': {
                                    'prescript': {
                                        'hyper': 'soft',
                                        'myope': 'no lenses'}
                                },
                                'young': 'soft'
                            }
                        }
                    }
                }
            }
        }
        self.assertEqual(expected, lensesTree)

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
        trees - calculate_shannon_entropy returns the shannon entropy
        """
        my_data, labels = self.create_dataset()
        se = trees.calculate_shannon_entropy(my_data)
        self.assertEqual(se, 0.9709505944546686)

    def test_split_dataset(self):
        """
        trees - split_data_set breaks feature vectors on desired vals
        """
        my_data, labels = self.create_dataset()
        returned = trees.split_data_set(my_data, 0, 1)
        self.assertEqual([[1, 'yes'], [1, 'yes'], [0, 'no']], returned)

    def test_choose_best_feature_to_split(self):
        """
        trees - choose_best_feature_to_split discovers best feature
        """
        my_data, labels = self.create_dataset()
        returned = trees.choose_best_feature_to_split(my_data)
        self.assertEqual(0, returned)

    def test_majority_count(self):
        """trees - majority_count returns most common element """
        animals = ['cow', 'chicken', 'chicken',
                   'hen', 'chicken', 'chicken', 'fox', 'fox',
                   ]
        returned = trees.majority_count(animals)
        self.assertEqual(returned, 'chicken')

if __name__ == '__main__':
    unittest.main()
