#!/usr/bin/env python

# k nearest neighbors
# mjamesruggiero
# Thu Nov  7 22:13:21 PST 2013
from numpy import tile, array, zeros
import operator
import matplotlib
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.ERROR, format="%(lineno)d\t%(message)s")

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


def main():
    roles = ['scatter1', 'scatter2']
    this_role = 'scatter2'
    data_file = 'data/datingTestSet2.txt'

    fig = plt.figure()
    dating_data_matrix, dating_labels = file_to_matrix(data_file)
    logging.info("the matrix is {}".format(dating_data_matrix))
    ax = fig.add_subplot(111)

    logging.info("calling plt.show() for role '{0}'".format(this_role))
    if this_role == roles[0]:
        ax.scatter(dating_data_matrix[:, 1], dating_data_matrix[:,2])
    else:
        ax.scatter(dating_data_matrix[:, 1], dating_data_matrix[:, 2], 15.0*array(dating_labels))

    plt.show()

if __name__ == '__main__':
    main()
