import unittest
from context import lily
from lily import core
import numpy as np


class TestLily(unittest.TestCase):
    def test_classify_0(self):
        """
        lily - classify_0 returns the majority class as the prediction
        """
        large_shape = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
        labels = ['foo', 'bar', 'bz', 'quux']
        result = core.classify_0(large_shape[1, :],
                                 large_shape[2, :],
                                 labels, 2)
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
        se = core.calculate_shannon_entropy(my_data)
        self.assertEqual(se, 0.9709505944546686)

    def test_split_dataset(self):
        """
        lily - split_data_set breaks feature vectors on desired vals
        """
        my_data, labels = self.create_dataset()
        returned = core.split_data_set(my_data, 0, 1)
        self.assertEqual([[1, 'yes'], [1, 'yes'], [0, 'no']], returned)

    def test_choose_best_feature_to_split(self):
        """
        lily - choose_best_feature_to_split discovers best feature
        """
        my_data, labels = self.create_dataset()
        returned = core.choose_best_feature_to_split(my_data)
        self.assertEqual(0, returned)

    def load_bayes_data_set(self):
        posting_list = [['my', 'dog', 'has', 'flea',
                         'problems', 'help', 'please'],
                        ['maybe', 'not', 'take', 'him',
                         'to', 'dog', 'park', 'stupid'],
                        ['my', 'dalmation', 'is', 'so',
                         'cute', 'i', 'love', 'him'],
                        ['stop', 'posting', 'stupid',
                         'worthless', 'garbage'],
                        ['mr', 'licks', 'ate', 'my', 'steak',
                         'how', 'to', 'stop', 'him'],
                        ['quit', 'buying', 'worthless',
                         'dog', 'food', 'stupid']]
        # 1 = abusive; 0 = not abusive
        class_vector = [0, 1, 0, 1, 0, 1]
        return posting_list, class_vector

    def test_create_vocabulary_list(self):
        """lily - create_vocabulary_list creates... you know"""
        list_of_posts, list_classes = self.load_bayes_data_set()
        expected = ['cute', 'love', 'help', 'garbage',
                    'quit', 'food', 'problems', 'is',
                    'park', 'stop', 'flea', 'dalmation',
                    'licks', 'not', 'him', 'buying',
                    'posting', 'has', 'worthless', 'ate',
                    'to', 'i', 'maybe', 'please',
                    'dog', 'how', 'stupid', 'so',
                    'take', 'mr', 'steak', 'my']
        self.assertEqual(expected, core.create_vocabulary_list(list_of_posts))

    def test_set_of_words_to_vector(self):
        """lily - set_of_words_to_vector turns words into bit vector"""
        vocabulary = ['you', 'hey', 'monkey',
                      'ape', 'gorilla', 'lump', 'chimpanzee']
        input_set = set({'hey', 'dude', 'chimpanzee'})
        self.assertEqual([0, 1, 0, 0, 0, 0, 1],
                         core.set_of_words_to_vector(vocabulary, input_set))

    def test_naive_bayes(self):
        """
        lily - naive bayes can classify text based upon trained vocabulary
        """
        list_of_posts, list_classes = self.load_bayes_data_set()
        vocabulary = core.create_vocabulary_list(list_of_posts)
        training_matrix = []
        for post in list_of_posts:
            vector = core.set_of_words_to_vector(vocabulary, post)
            training_matrix.append(vector)

        p_0_vector, p_1_vector, p_any_being_abusive = \
            core.train_naive_bayes0(np.array(training_matrix),
                                    np.array(list_classes))

        test_entry = ['love', 'my', 'dalmation']

        vector = core.set_of_words_to_vector(vocabulary, test_entry)
        this_document = np.array(vector)
        result = core.classify_naive_bayes(this_document,
                                           p_0_vector,
                                           p_1_vector,
                                           p_any_being_abusive)
        self.assertEqual(0, result)

        test_entry = ['stupid', 'garbage']
        vector = core.set_of_words_to_vector(vocabulary, test_entry)
        this_document = np.array(vector)
        result = core.classify_naive_bayes(this_document,
                                           p_0_vector,
                                           p_1_vector,
                                           p_any_being_abusive)
        self.assertEqual(1, result)

if __name__ == '__main__':
    unittest.main()
