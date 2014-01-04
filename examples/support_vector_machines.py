import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from lily import svm


def simple_smo():
    data_array, label_array = svm.load_dataset('data/svm_test_set.txt')
    b, alphas = svm.smo_simple(data_array, label_array, 0.6, 0.001, 40)
    return b, alphas


def optimized_smo():
    data_array, label_array = svm.load_dataset('data/svm_test_set.txt')
    b, alphas = svm.platt_outer_loop(data_array, label_array, 0.6, 0.001, 40)
    return b, alphas


def main():
    optimized_b, optimized_alphas = optimized_smo()
    print "optimized b: {0}".format(optimized_b)

if __name__ == '__main__':
    main()
