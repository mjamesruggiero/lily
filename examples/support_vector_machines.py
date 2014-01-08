import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from lily import svm
from numpy import mat


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


def main():
    demo_classification()

if __name__ == '__main__':
    main()
