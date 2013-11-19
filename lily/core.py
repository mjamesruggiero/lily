#!/usr/bin/env python

# lily is a module for statistics functions
from numpy import tile, zeros, ones, log, shape
import operator
import math

import logging
FORMAT = "%(filename)s, %(funcName)s, %(lineno)d %(message)s"
logging.basicConfig(level=logging.ERROR, format=FORMAT)


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
        shannon_entropy -= prob * math.log(prob, 2)
    return shannon_entropy


def split_data_set(data_set, axis, value):
    """
    Takes: dataset to split, the feature to split on,
    and the value of the feature to return.
    and cut out the feature to split on
    """
    ret_data_set = []
    for feature_vec in data_set:
        if feature_vec[axis] == value:
            reduced_feature_vec = feature_vec[:axis]
            reduced_feature_vec.extend(feature_vec[axis+1:])
            ret_data_set.append(reduced_feature_vec)
    return ret_data_set


def choose_best_feature_to_split(data_set):
    num_features = len(data_set[0]) - 1
    base_entropy = calculate_shannon_entropy(data_set)
    logging.info("base entropy is {0}".format(base_entropy))
    best_information_gain = 0.0
    best_feature = -1

    for i in range(num_features):
        feature_list = [example[i] for example in data_set]
        unique_values = set(feature_list)
        logging.info("unique values are {0}".format(unique_values))

        new_entropy = 0.0
        for value in unique_values:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set)/float(len(data_set))

            new_entropy += prob * calculate_shannon_entropy(sub_data_set)
            logging.info("value is {0}; prob is {1}; new entropy is {2}".
                         format(value, prob, new_entropy))

        information_gain = base_entropy - new_entropy
        logging.info("--> information gain for {0} is {1}".
                     format(i, information_gain))

        if information_gain > best_information_gain:
            best_information_gain = information_gain
            best_feature = i

    return best_feature


def majority_count(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(),
                                key=operator.itemgetter(1),
                                reverse=True)
    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    class_list = [example[-1] for example in data_set]

    # stop when all classes are equal
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    # when there are no more features, return majority
    if len(data_set[0]) == 1:
        return majority_count(class_list)

    best_feature = choose_best_feature_to_split(data_set)
    best_feature_label = labels[best_feature]
    my_tree = {best_feature_label: {}}
    del(labels[best_feature])
    feature_values = [example[best_feature] for example in data_set]
    unique_values = set(feature_values)
    for value in unique_values:
        sub_labels = labels[:]
        my_tree[best_feature_label][value] = \
            create_tree(split_data_set(data_set,
                                       best_feature,
                                       value), sub_labels)

    return my_tree


def train_naive_bayes0(train_matrix, train_category):
    """
    * train_matrix: array of array of strings
    * train_category: array of 0 or 1;
    corresponding to the "class" of each
    array of strings in train matrix
    """
    num_training_docs = len(train_matrix)
    num_words = len(train_matrix[0])

    #initialize probablilities; 0 or 1
    prob_1 = sum(train_category)/float(num_training_docs)
    p0_num = ones(num_words)
    p1_num = ones(num_words)
    p0_denom = 2.0
    p1_denom = 2.0

    for i in range(num_training_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])

    # change to log() to help with underflow
    p1_vector = log(p1_num/p1_denom)
    p0_vector = log(p0_num/p0_denom)
    return p0_vector, p1_vector, prob_1


def classify_naive_bayes(vector_to_classify, p_0_vec, p_1_vec, p_class_1):
    """Using element-wise multiplication"""
    p1 = sum(vector_to_classify * p_1_vec) + log(p_class_1)
    p0 = sum(vector_to_classify * p_0_vec) + log(1.0 - p_class_1)
    if p1 > p0:
        return 1
    else:
        return 0


def create_vocabulary_list(data_set):
    vocabulary_set = set([])
    for document in data_set:
        vocabulary_set = vocabulary_set | set(document)
    return list(vocabulary_set)


def set_of_words_to_vector(vocabulary_list, input_set):
    return_vec = [0]*len(vocabulary_list)
    for word in input_set:
        if word in vocabulary_list:
            return_vec[vocabulary_list.index(word)] = 1
        else:
            logging.warn("word '{0}' is not in known vocabulary!".format(word))
    return return_vec
