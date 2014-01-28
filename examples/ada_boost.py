#!/usr/bin/env python

import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath('..'))
from lily import ada_boost

import logging
logging.basicConfig(level=logging.INFO, format="%(funcName)s\t%(message)s")


def load_simple_data():
    data_matrix = np.matrix([[1., 2.1],
                             [2., 1.1],
                             [1.3, 1.],
                             [1., 1.],
                             [2., 1.]])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_matrix, class_labels


def main():
    #test_classification()
    #test_horse_colic()
    test_compute_horse_auc()


def test_classification():
    D = np.mat(np.ones((5, 1)) / 5)
    data_matrix, class_labels = load_simple_data()
    stump, min_error, best_estimate = ada_boost.build_stump(data_matrix,
                                                            class_labels,
                                                            D)
    logging.info('stump: {}'.format(stump))
    logging.info('min_error: {}'.format(min_error))
    logging.info('best_estimate: {}'.format(best_estimate))
    classifier_array, aggregated_class_estimates =\
        ada_boost.train_dataset(data_matrix, class_labels, 9)
    data_to_classify = [0, 0]
    classifications = ada_boost.classify(data_to_classify, classifier_array)
    logging.info("classifications: {c}".format(c=classifications))


def load_dataset(filename):
    number_of_features = len(open(filename).readline().split('\t'))
    data_matrix = []
    label_matrix = []
    fr = open(filename)
    for line in fr.readlines():
        pieces = []
        current_line = line.strip().split('\t')
        for i in range(number_of_features - 1):
            pieces.append(float(current_line[i]))
        data_matrix.append(pieces)
        label_matrix.append(float(current_line[-1]))
    return data_matrix, label_matrix


def test_horse_colic():
    data, labels = load_dataset('data/horseColicTraining.txt')
    classifier_array, aggregated_class_estimates =\
        ada_boost.train_dataset(data, labels, 10)

    test_data, test_labels = load_dataset('data/horseColicTest.txt')
    prediction10 = ada_boost.classify(test_data, classifier_array)

    # calculate the error
    elements = 67
    err_array = np.mat(np.ones((elements, 1)))
    error_count = err_array[prediction10 != np.mat(test_labels).T].sum()
    message = "total_errors = {total_errors}; error rate= {rate}".\
        format(total_errors=error_count, rate=(error_count / elements))
    logging.info(message)


def test_compute_horse_auc():
    """plot the AUC for the horse colic data"""
    data, labels = load_dataset('data/horseColicTraining.txt')
    classifier_array, aggregated_class_estimates =\
        ada_boost.train_dataset(data, labels, 40)
    ada_boost.plot_roc(aggregated_class_estimates.T, labels)

if __name__ == '__main__':
    main()
