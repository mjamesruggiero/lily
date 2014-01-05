import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from lily import svm

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


def optimized_smo():
    data_array, label_array = load_dataset('data/svm_test_set.txt')
    b, alphas = svm.platt_outer_loop(data_array, label_array, 0.6, 0.001, 40)
    return b, alphas


def main():
    optimized_b, optimized_alphas = optimized_smo()
    print "optimized b: {0}".format(optimized_b)

if __name__ == '__main__':
    main()
