import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from lily import svm

def main():
    data_array, label_array = svm.load_dataset('data/svm_test_set.txt')
    b, alphas = svm.smo_simple(data_array, label_array, 0.6, 0.001, 40)
    print "b is {0}".format(b)

if __name__ == '__main__':
    main()

