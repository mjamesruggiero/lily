#!/usr/bin/env python

# k nearest neighbors
# mjamesruggiero
# Thu Nov  7 22:13:21 PST 2013
from numpy import array, zeros
import matplotlib.pyplot as plt
import argparse
import os
import logging
import sys

sys.path.insert(0, os.path.abspath('..'))
from lily import core

def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def dating_class_test(data_file, ho_ratio=0.10):
    dating_data_matrix, dating_labels = core.file_to_matrix(data_file)
    norm_matrix, ranges, min_vals = core.auto_norm(dating_data_matrix)
    m = norm_matrix.shape[0]
    num_test_vecs = int(m * ho_ratio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = core.classify_0(norm_matrix[i, :],
                                            norm_matrix[num_test_vecs:m, :],
                                            dating_labels[num_test_vecs:m],
                                            3)

        pass_fail_msg = 'p'
        if classifier_result != dating_labels[i]:
            error_count += 1.0
            pass_fail_msg = '**FAIL**'

        msg = "\t".join(["[{num}] classifier:",
              "{classifier}",
              "real answer:",
              "{label}",
              "{pass_fail}"]).format(num=i,
                                     classifier=classifier_result,
                                     label=dating_labels[i],
                                     pass_fail=pass_fail_msg)

        logging.info(msg)

    logging.info("the total error rate is {0:.2f}".
                 format(error_count/float(num_test_vecs)))


def build_graph(data_file):
    mode = 'normalized'
    dating_data_matrix, dating_labels = core.file_to_matrix(data_file)

    if 'graphing' == mode:
        roles = ['scatter1', 'scatter2']
        this_role = 'scatter2'

        fig = plt.figure()
        logging.info("the matrix is {}".format(dating_data_matrix))
        ax = fig.add_subplot(111)

        logging.info("calling plt.show() for role '{0}'".format(this_role))
        if this_role == roles[0]:
            ax.scatter(dating_data_matrix[:, 1], dating_data_matrix[:, 2])
        else:
            ax.scatter(dating_data_matrix[:, 1],
                       dating_data_matrix[:, 2],
                       15.0*array(dating_labels))

        plt.show()

    if 'normalized' == mode:
        norm_matrix, ranges, min_vals = core.auto_norm(dating_data_matrix)
        logging.info("norm_matrix is {}".format(norm_matrix))


def classify_person(filepath):
    """
    predict whether or not the features match a passing candidate
    """
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(raw_input(
                         "percentage of time spent playing video games? "))
    f_flier_miles = float(raw_input("frequent flier miles earned per year? "))
    ice_cream = float(raw_input("liters of ice cream consumed per year? "))

    dating_matrix, dating_labels = core.file_to_matrix(filepath)
    normalized_matrix, ranges, min_vals = core.auto_norm(dating_matrix)
    in_arr = array([f_flier_miles, percent_tats, ice_cream])
    classifier_result = core.classify_0((in_arr - min_vals)/ranges,
                                        normalized_matrix,
                                        dating_labels,
                                        3)
    print("You will probably like this person: ",
          result_list[classifier_result - 1])



if __name__ == '__main__':
    DESCRIPTION = 'A small script that demoes kNN'
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    # all required
    parser.add_argument('data_file', action="store")
    parser.add_argument('mode', action="store")
    parser.add_argument('ho_ratio', action="store", default=0.10, type=float)

    # optional logging bit
    parser.add_argument('-l', action="store_true", default=False)

    results = parser.parse_args()

    l_level = logging.ERROR
    if results.l:
        l_level = logging.DEBUG

    FORMAT = "%(filename)s, %(lineno)d %(message)s"
    logging.basicConfig(level=l_level, format=FORMAT)

    logging.info(parser.parse_args())

    ho_ratio = results.ho_ratio
    data_file = results.data_file
    mode = results.mode

    if mode == 'test':
        dating_class_test(data_file, ho_ratio)
    if mode == 'graph':
        build_graph(data_file)
    if mode == 'predict':
        classify_person(data_file)
