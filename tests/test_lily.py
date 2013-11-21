import unittest
from context import lily
from lily import core
import numpy as np


class TestLily(unittest.TestCase):
    def test_classify_0(self):
        """
        core - classify_0 returns the majority class as the prediction
        """
        large_shape = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
        labels = ['foo', 'bar', 'bz', 'quux']
        result = core.classify_0(large_shape[1, :],
                                 large_shape[2, :],
                                 labels, 2)
        assert result == 'foo'

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
        """core - create_vocabulary_list creates... you know"""
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
        """core - set_of_words_to_vector turns words into bit vector"""
        vocabulary = ['you', 'hey', 'monkey',
                      'ape', 'gorilla', 'lump', 'chimpanzee']
        input_set = set({'hey', 'dude', 'chimpanzee'})
        self.assertEqual([0, 1, 0, 0, 0, 0, 1],
                         core.set_of_words_to_vector(vocabulary, input_set))

    def test_naive_bayes(self):
        """
        core - naive bayes can classify text based upon trained vocabulary
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
