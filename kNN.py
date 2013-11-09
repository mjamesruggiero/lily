#!/usr/bin/env python

# k nearest neighbors
# mjamesruggiero
# Thu Nov  7 22:13:21 PST 2013
from numpy import tile, array, zeros, shape
import operator
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.DEBUG, format="%(lineno)d\t%(message)s")


def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify_0(in_x, data_set, labels, k):
    """For every point in our dataset
    calculate the distance between inX and the current point
    sort the distances in increasing order
    take k items with lowest distances to inX
    find the majority class among these items
    return the majority class as our prediction for the class of inX"""
    data_set_size = data_set.shape[0]

    # distance
    diff_mat = tile(in_x, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat**2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances**0.5

    # argsort pulls indices corresponding
    # to sorted array
    sorted_dist_indices = distances.argsort()

    # voting with lowest k indices
    class_count = {}
    for i in range(k):
        vote_i_label = labels[sorted_dist_indices[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1

    sorted_class_count = sorted(class_count.iteritems(),
                                key=operator.itemgetter(1),
                                reverse=True)

    # the most frequent
    return sorted_class_count[0][0]


def file_to_matrix(filename):
    """Create numpy matrix from file"""
    fr = open(filename)
    all_lines = fr.readlines()
    number_of_lines = len(all_lines)
    return_matrix = zeros((number_of_lines, 3))
    class_label_vector = []

    index = 0
    for line in all_lines:
        line = line.strip()
        list_from_line = line.split('\t')
        logging.info("list is {0}".format(list_from_line))
        return_matrix[index, :] = list_from_line[0:3]
        logging.info("index is {0}".format(index))

        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_matrix, class_label_vector


def auto_norm(data_set):
    """
    Get the minimum values of each column
    and place in min_vals. max_vals, too.
    data_set.min(0) allows you to take the minimums
    from the columns, not the rows.
    Then calculate the range of possible
    values seen in our data.
    To get the normalized values,
    you subtract the minimum values
    and then divide by the range."""
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    norm_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - tile(min_vals, (m, 1))
    norm_data_set = norm_data_set / tile(ranges, (m, 1))
    return norm_data_set, ranges, min_vals


def dating_class_test():
    ho_ratio = 0.10
    data_file = 'data/datingTestSet2.txt'
    dating_data_matrix, dating_labels = file_to_matrix(data_file)
    norm_matrix, ranges, min_vals = auto_norm(dating_data_matrix)
    m = norm_matrix.shape[0]
    num_test_vecs = int(m * ho_ratio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify_0(norm_matrix[i, :],
                                       norm_matrix[num_test_vecs:m, :],
                                       dating_labels[num_test_vecs:m], 3)

        display_result = 'P'
        if classifier_result != dating_labels[i]:
            error_count += 1.0
            display_result = '**FAIL**'

        message = "classifier return:\t{0}, real answer:\t{1}\t{2}".\
                  format(classifier_result, dating_labels[i], display_result)
        logging.info(message)

    logging.info("the total error rate is {0:.2f}".
                 format(error_count/float(num_test_vecs)))


def main():
    mode = 'normalized'
    data_file = 'data/datingTestSet2.txt'
    dating_data_matrix, dating_labels = file_to_matrix(data_file)

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
        norm_matrix, ranges, min_vals = auto_norm(dating_data_matrix)
        logging.info("norm_matrix is {}".format(norm_matrix))

if __name__ == '__main__':
    #main()
    dating_class_test()
