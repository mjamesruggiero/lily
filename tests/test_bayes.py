import unittest
from context import lily
from lily import core
import random
import os
from numpy import array

import logging
logging.basicConfig(level=logging.ERROR, format="%(lineno)d\t%(message)s")


class TestTestBayes(unittest.TestCase):

    def _bayes_trial(self):
        doc_list = []
        class_list = []
        full_text = []

        root_path = os.path.abspath(os.path.join(os.curdir, os.pardir))

        for i in range(1, 26):
            filepath = "{0}/data/spam/{1}.txt".format(root_path, i)
            word_list = core.text_parse(open(filepath).read())
            doc_list.append(word_list)
            full_text.extend(word_list)
            class_list.append(1)

            filepath = "{0}/data/ham/{1}.txt".format(root_path, i)
            word_list = core.text_parse(open(filepath).read())
            doc_list.append(word_list)
            full_text.extend(word_list)
            class_list.append(0)
        vocabulary = core.create_vocabulary_list(doc_list)
        training_set = range(50)

        test_set = []
        for i in range(10):
            rand_index = int(random.uniform(0, len(training_set)))
            test_set.append(training_set[rand_index])
            del(training_set[rand_index])

        training_matrix = []
        train_classes = []
        for doc_index in training_set:
            training_matrix.append(core.bag_of_words_to_vector(
                                   vocabulary,
                                   doc_list[doc_index]))
            train_classes.append(class_list[doc_index])

        p_0_V, p_1_V, p_spam = core.train_naive_bayes0(array(
                                                       training_matrix),
                                                       array(train_classes))
        error_count = 0
        for doc_index in test_set:
            word_vector = core.bag_of_words_to_vector(vocabulary,
                                                      doc_list[doc_index])
            result = core.classify_naive_bayes(array(word_vector),
                                               p_0_V,
                                               p_1_V,
                                               p_spam)
            if result != class_list[doc_index]:
                error_count += 1
        error_rate = float(error_count)/len(test_set)
        logging.info("errors: {0}\ttest_set_count: {1}".format(error_count,
                                                               len(test_set)))
        logging.info("the error rate is {0}".format(error_rate))
        return error_rate

    def test_bayes(self):
        """
        bayes - run 10 tests, get a non-zero error rate
        """
        rates = []
        for i in range(10):
            rates.append(self._bayes_trial())
        mean_rate = sum(rates)/float(len(rates))
        message = "rate {0} not greater than or equal to 0.01"
        self.assertGreaterEqual(mean_rate,
                                0.01,
                                message.format(mean_rate))


if __name__ == '__main__':
    unittest.main()
