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
    data = regression_trees.load_dataset('data/ex0.txt')
    matrix = np.mat(data)
    tree = regression_trees.create_tree(matrix, ops=(0, 1))
    return tree


def log_formatted_tree(tree, label="formatted tree"):
    formatted = pprint.pformat(tree)
    logging.info("{label}:\n{tree}".format(label=label, tree=formatted))


def pruning_example():
    complex_tree = more_complex_tree()
    log_formatted_tree(complex_tree)

    my_data = regression_trees.load_dataset('data/ex2test.txt')
    my_matrix = np.mat(my_data)
    pruned = regression_trees.prune(complex_tree, my_matrix)

    log_formatted_tree(pruned, 'pruned tree')


def piecewise_linear_solve_example():
    matrix_2 = np.mat(regression_trees.load_dataset('data/exp2.txt'))
    model_tree = regression_trees.create_tree(matrix_2,
                                              regression_trees.model_leaf,
                                              regression_trees.model_error)
    log_formatted_tree(model_tree, 'model tree')


def main():
    very_simple_tree()
    pruning_example()
    piecewise_linear_solve_example()

if __name__ == '__main__':
    main()
