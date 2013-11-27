# lily is a module for statistics functions
from numpy import tile, zeros, ones, log, shape
import operator
import logging
import re

FORMAT = "%(filename)s, %(funcName)s, %(lineno)d %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)


def text_parse(big_string):
    """
    This could be generalized further.
    """
    list_of_tokens = re.split(r'\W*', big_string)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


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

    for i, line in enumerate(all_lines):
        line = line.strip()
        list_from_line = line.split('\t')
        return_matrix[i, :] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
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


def create_vocabulary(data_set):
    vocabulary_set = set([])
    for document in data_set:
        try:
            vocabulary_set = vocabulary_set | set(document)
        except TypeError, e:
            print("error {0} with document {0}".format(e, document))
    return list(vocabulary_set)


def bag_of_words_to_vector(vocabulary_list, input_set):
    return_vec = [0]*len(vocabulary_list)
    for word in input_set:
        if word in vocabulary_list:
            return_vec[vocabulary_list.index(word)] += 1
    return return_vec


def calculate_most_frequent(vocabulary, full_text, limit=30):
    """calculate the frequency of occurrence"""
    frequency_dict = {}

    for token in vocabulary:
        frequency_dict[token] = full_text.count(token)

    sorted_frequency = sorted(frequency_dict.iteritems(),
                              key=operator.itemgetter(1),
                              reverse=True)
    return sorted_frequency[:limit]


def save_to_csv(itera, filepath, headers=None):
    """
    itera: list of lists or list of tuples
    filepath: a... filepath
    """
    import csv
    with open(filepath, 'wb') as csvfile:
        _writer = csv.writer(csvfile,
                             delimiter=',',
                             quoting=csv.QUOTE_MINIMAL)
        if headers:
            _writer.writerow(headers)
        for row in itera:
            _writer.writerow(row)


def get_stopwords(stopwords_file):
    fr = open(stopwords_file)
    all_lines = fr.readlines()
    stopwords = []

    for i, line in enumerate(all_lines):
        stopwords.append(line.strip())
    return set(stopwords)


def is_stopword(word, stopwords):
    return word in stopwords
