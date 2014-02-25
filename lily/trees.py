import logging
import math
from collections import Counter

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
    """
    sample data for shannon entropy test
    """
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
            reduced_feature_vec.extend(feature_vec[axis + 1:])
            logging.info("reduced_feature_vec:\t{0}".
                         format(reduced_feature_vec))
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
            prob = len(sub_data_set) / float(len(data_set))

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
    """
    Take a list of class names; build
    a dict whose keys are the unique names.
    Count the frequency, and return the one
    with the greatest frequency
    """
    class_count = Counter(class_list)
    return class_count.most_common()[0][0]


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
        prob = float(label_counts[key]) / num_entries
        shannon_entropy -= prob * math.log(prob, 2)
    return shannon_entropy


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
