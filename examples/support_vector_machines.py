import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from lily import svm
from lily import optimizer
from lily import utils
from numpy import mat, nonzero, shape, multiply, sign, zeros

import logging
logging.basicConfig(level=logging.INFO, format="%(funcName)s\t%(message)s")


def demo_classification():
    data_array, label_array = utils.load_dataset('data/svm_test_set.txt')
    b, alphas = svm.platt_outer_loop(data_array, label_array, 0.6, 0.001, 40)
    ws = svm.calculate_ws(alphas, data_array, label_array)
    data_matrix = mat(data_array)

    for i in range(15):
        classification = data_matrix[i] * mat(ws) + b
        label = label_array[i]
        message = "{:>8}:\tclassification: {:>12}\tlabel: {:>4}"
        print message.format(i, classification, label)


def optimized_smo():
    data_array, label_array = utils.load_dataset('data/svm_test_set.txt')
    b, alphas = svm.platt_outer_loop(data_array, label_array, 0.6, 0.001, 40)
    return b, alphas


def test_rbf(train_data,
             train_labels,
             test_data,
             test_labels,
             k_tuple=('rbf', 1.3)):
    """
    actually create a classifier
    that can handle datasets that don't cleanly
    divide. In this case, use the radial bias kernel.
    """
    b, alphas = svm.platt_outer_loop(train_data,
                                     train_labels,
                                     200,
                                     0.0001,
                                     10000,
                                     k_tuple)
    data_matrix = mat(train_data)
    label_matrix = mat(train_labels).transpose()

    sv_index = nonzero(alphas.A > 0)[0]
    support_vectors = data_matrix[sv_index]
    label_support_vector = label_matrix[sv_index]
    print "there are {0} support vectors".format(shape(support_vectors)[0])

    m, n = shape(data_matrix)
    error_count = 0
    for i in range(m):
        kernel_evaluation = optimizer.kernel_transform(support_vectors,
                                                       data_matrix[i, :],
                                                       k_tuple)
        prediction = kernel_evaluation.T * multiply(label_support_vector,
                                                    alphas[sv_index]) + b
        if sign(prediction) != sign(train_labels[i]):
            error_count += 1
    print "training error rate: {0}".format(float(error_count) / m)

    error_count = 0
    data_matrix = mat(test_data)
    label_matrix = mat(test_labels).transpose()
    m, n = shape(data_matrix)
    for i in range(m):
        kernel_evaluation = optimizer.kernel_transform(support_vectors,
                                                       data_matrix[i, :],
                                                       k_tuple)
        prediction = kernel_evaluation.T * multiply(label_support_vector,
                                                    alphas[sv_index]) + b
        if sign(prediction) != sign(test_labels[i]):
            error_count += 1
    print "test error rate: {0}".format(float(error_count) / m)


def load_images(directory):
    """load the character files"""
    from os import listdir
    hw_labels = []
    traning_file_list = listdir(directory)
    m = len(traning_file_list)
    training_matrix = zeros((m, 1024))
    for i in range(m):
        file_name = traning_file_list[i]
        file_itself = file_name.split('.')[0]
        class_number = int(file_itself.split('_')[0])
        if 9 == class_number:
            hw_labels.append(-1)
        else:
            hw_labels.append(1)
        file_to_analyze = '{0}/{1}'.format(directory, file_name)
        training_matrix[i, :] = img_to_vector(file_to_analyze)
    return training_matrix, hw_labels


def test_digits(k_tuple=('rbf', 10)):
    training_data, training_labels = load_images('data/training_digits')
    test_data_array, test_labels = load_images('data/test_digits')
    test_rbf(training_data,
             training_labels,
             test_data_array,
             test_labels,
             k_tuple)


def img_to_vector(filename):
    """
    converts an image to a vector.
    create a 1x1024 NumPy array;
    open the file, loop over the first 32 lines;
    store integer of each line's first 32 chars
    """
    return_vector = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vector[0, 32 * i + j] = int(line_str[j])
    return return_vector


def demo_classify():
    print '{:-^30}'.format('demo classification')
    demo_classification()


def demo_digits():
    print '{:-^30}'.format('testing digit recognition (takes a while)')
    test_digits()


def main():
    demo_classify()
    demo_digits()

if __name__ == '__main__':
    main()
