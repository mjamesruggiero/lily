import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from lily import svm
from lily import optimizer
from numpy import mat, nonzero, shape, shape, multiply, sign


def load_dataset(file_name):
    data_matrix = []
    label_matrix = []
    fr = open(file_name)
    for line in fr.readlines():
            line_array = line.strip().split('\t')
            data_matrix.append([float(line_array[0]), float(line_array[1])])
            label_matrix.append(float(line_array[2]))
    return data_matrix, label_matrix


def simple_smo():
    data_array, label_array = load_dataset('data/svm_test_set.txt')
    b, alphas = svm.smo_simple(data_array, label_array, 0.6, 0.001, 40)
    return b, alphas


def demo_classification():
    data_array, label_array = load_dataset('data/svm_test_set.txt')
    b, alphas = svm.platt_outer_loop(data_array, label_array, 0.6, 0.001, 40)
    ws = svm.calculate_ws(alphas, data_array, label_array)
    data_matrix = mat(data_array)

    for i in range(15):
        classification = data_matrix[i] * mat(ws) + b
        label = label_array[i]
        message = "{0}:\tclassification: {1}\tlabel: {2}"
        print message.format(i, classification, label)


def optimized_smo():
    data_array, label_array = load_dataset('data/svm_test_set.txt')
    b, alphas = svm.platt_outer_loop(data_array, label_array, 0.6, 0.001, 40)
    return b, alphas


def test_rbf(k1=1.3):
    """
    actually create a classifier
    that can handle datasets that don't cleanly
    divide. In this case, use the radial bias kernel.
    """
    data_array, label_array = load_dataset('data/test_set_rbf.txt')
    b, alphas = svm.platt_outer_loop(data_array, label_array, 200, 0.0001, 10000, ('rbf', k1))
    data_matrix = mat(data_array)
    label_matrix = mat(label_array).transpose()

    sv_index = nonzero(alphas.A > 0)[0]
    support_vectors = data_matrix[sv_index]
    label_support_vector = label_matrix[sv_index]
    print "there are {0} support vectors".format(shape(support_vectors)[0])

    m, n = shape(data_matrix)
    error_count = 0
    for i in range(m):
        kernel_evaluation = optimizer.kernel_transform(support_vectors, data_matrix[i, :], ('rbf', k1))
        prediction = kernel_evaluation.T * multiply(label_support_vector, alphas[sv_index]) + b
        if sign(prediction) != sign(label_array[i]):
            error_count += 1
    print "training error rate: {0}".format(float(error_count)/m)

    data_array, label_array = load_dataset('data/test_set_rbf_2.txt')
    error_count = 0
    data_matrix = mat(data_array)
    label_matrix = mat(label_array).transpose()
    m, n = shape(data_matrix)
    for i in range(m):
        kernel_evaluation = optimizer.kernel_transform(support_vectors, data_matrix[i, :], ('rbf', k1))
        prediction = kernel_evaluation.T * multiply(label_support_vector, alphas[sv_index]) + b
        if sign(prediction) != sign(label_array[i]):
            error_count += 1
    print "test error rate: {0}".format(float(error_count)/m)



def main():
    print "-"*12
    print "demo classification:"
    demo_classification()

    print "-"*12
    rbf = 0.50
    print "testing support vector machines; rbf is {value}".format(value=rbf)
    test_rbf(rbf)

if __name__ == '__main__':
    main()
