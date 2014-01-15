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
    D = np.mat(np.ones((5, 1)) / 5)
    data_matrix, class_labels = load_simple_data()
    stump, min_error, best_estimate = ada_boost.build_stump(data_matrix,
                                                            class_labels,
                                                            D)
    logging.info('stump: {}'.format(stump))
    logging.info('min_error: {}'.format(min_error))
    logging.info('best_estimate: {}'.format(best_estimate))
    classifier_array = ada_boost.train_dataset(data_matrix,
                                               class_labels,
                                               9)
    for c in classifier_array:
        logging.info('classifier: {c}'.format(c=c))


if __name__ == '__main__':
    main()
