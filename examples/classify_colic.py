import os
import sys
from numpy import array
sys.path.insert(0, os.path.abspath('..'))
from lily import regression


def colic_test():
    """
    use logistic regression to predict if a horse
    with colic will live or die
    """
    feature_train = open("data/horseColicTraining.txt")
    feature_test = open("data/horseColicTest.txt")
    training_set = []
    training_labels = []
    COLUMNS = 21
    for line in feature_train.readlines():
        current = line.strip().split("\t")
        line_pieces = []
        for i in range(COLUMNS):
            line_pieces.append(float(current[i]))
        training_set.append(line_pieces)
        training_labels.append(float(current[COLUMNS]))

    training_weights = regression.modified_stochastic_gradient_ascent(array(training_set),
                                                                      training_labels,
                                                                      500)
    error_count = 0
    num_test_vectors = 0.0
    for line in feature_test.readlines():
        num_test_vectors += 1.0
        current = line.strip().split("\t")
        line_pieces = []
        for i in range(COLUMNS):
            line_pieces.append(float(current[i]))
        classification = regression.classify_vector(array(line_pieces),
                                                    training_weights)
        if int(classification) != int(current[COLUMNS]):
            error_count += 1

    error_rate = (float(error_count)/num_test_vectors)

    error_for_graph = int(round(error_rate * 100))
    message = "the error rate of this test is {0}"
    print message.format(round(error_rate, 3)), error_for_graph * "|"
    return error_rate


def multi_test(number_of_tests=10):
    error_sum = 0.0
    for k in range(number_of_tests):
        error_sum += colic_test()
    print "after {0} iterations, the average error rate is {1}".\
        format(number_of_tests, error_sum/float(number_of_tests))

if __name__ == '__main__':
    #colic_test()
    multi_test()
