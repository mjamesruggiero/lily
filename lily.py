#!/usr/bin/env python

# lily is a module for statistics functions
from numpy import tile, zeros, shape
from math import log
import operator
import logging

def classify_0(in_x, data_set, labels, k):
    """For every point in our dataset
    calculate the distance between inX and the current point;
    sort the distances in increasing order;
    take k items with lowest distances to inX;
    find the majority class among these items;
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


def calculate_shannon_entropy(data_set):
    """
    calculate a cont of the number of instances;
    create a dict whose keys are the values in the final col;
    if a key was not encountered previously, one is created;
    for each key, keep track of how many times the label occurs;
    finally use the frequency of all the different labels
    to calculate the probablility of that label;
    then sum this up for all the labels
    """
    num_entries = len(data_set)
    label_counts = {}
    for feature_vec in data_set:
        current_label = feature_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1

    shannon_entropy = 0.0
    for key in label_counts:
        prob = float(label_counts[key])/num_entries
        shannon_entropy -= prob * log(prob, 2)
    return shannon_entropy
