#!/usr/bin/env python

# Thu Nov 14 08:28:48 PST 2013
# mjamesruggiero

import logging
FORMAT = "%(lineno)d\t%(message)s"
logging.basicConfig(level=logging.ERROR, format=FORMAT)


def classify(input_tree, feature_labels, test_vector):
    """
    Translate label string into index
    """
    class_label = None
    first_key = input_tree.keys()[0]
    second_dict = input_tree[first_key]
    feature_index = feature_labels.index(first_key)
    for key in second_dict.keys():
        logging.info("key is %s and feature index is %d", key, feature_index)
        if test_vector[feature_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key],
                                       feature_labels,
                                       test_vector)
            else:
                class_label = second_dict[key]
    return class_label


def create_dataset():
    """sample data for shannon entropy test"""
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def store_tree(input_tree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(input_tree, fw)
    fw.close()


def get_tree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    pass
