import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from lily import regression_trees
import numpy as np
import pprint

import logging
logging.basicConfig(level=logging.INFO, format="%(funcName)s\t%(message)s")


def very_simple_tree():
    my_data = regression_trees.load_dataset('data/ex00.txt')
    my_matrix = np.mat(my_data)
    my_tree = regression_trees.create_tree(my_matrix)
    logging.info("the tree:\n{0}".format(my_tree))


def more_complex_tree():
    my_data = regression_trees.load_dataset('data/ex0.txt')
    my_matrix = np.mat(my_data)
    my_tree = regression_trees.create_tree(my_matrix)
    formatted = pprint.pformat(my_tree)
    logging.info("the tree:\n{0}".format(formatted))


def main():
    #very_simple_tree()
    more_complex_tree()

if __name__ == '__main__':
    main()
