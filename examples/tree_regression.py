import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from lily import regression_trees
import numpy as np
import pprint

import logging
logging.basicConfig(level=logging.INFO, format="%(funcName)s\t%(message)s")


def load_dataset(filepath):
    data_matrix = []
    fr = open(filepath)
    for line in fr.readlines():
        current_line = line.strip().split('\t')
        line_values = map(float, current_line)
        data_matrix.append(line_values)
    return data_matrix


def very_simple_tree():
    data = load_dataset('data/ex00.txt')
    matrix = np.mat(data)
    tree = regression_trees.create_tree(matrix)
    log_formatted_tree(tree, "the tree")


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


def forecasting_models_example():
    """
    Build 3 models and evaluate the performance of model trees,
    regression trees and standard linear regression.
    """
    training_matrix = np.mat(regression_trees.
                             load_dataset('data/bikeSpeedVsIq_train.txt'))
    test_matrix = np.mat(regression_trees.
                         load_dataset('data/bikeSpeedVsIq_test.txt'))

    # training tree
    tree = regression_trees.create_tree(training_matrix, ops=(1, 20))
    y_hat = regression_trees.create_forecast(tree, test_matrix[:, 0])
    accuracy = np.corrcoef(y_hat, test_matrix[:, 1], rowvar=0)[0, 1]
    logging.info("training accuracy = {0}".format(accuracy))

    # model tree
    tree = regression_trees.create_tree(training_matrix,
                                        regression_trees.model_leaf,
                                        regression_trees.model_error,
                                        (1, 20))
    y_hat = regression_trees.create_forecast(tree,
                                             test_matrix[:, 0],
                                             regression_trees.
                                             model_tree_evaluation)

    accuracy = np.corrcoef(y_hat, test_matrix[:, 1], rowvar=0)[0, 1]
    logging.info("model tree accuracy = {0}".format(accuracy))

    weights, x, y = regression_trees.linearly_solve(training_matrix)
    for i in range(np.shape(test_matrix)[0]):
        y_hat[i] = test_matrix[i, 0] * weights[1, 0] + weights[0, 0]
    accuracy = np.corrcoef(y_hat, test_matrix[:, 1], rowvar=0)[0, 1]
    logging.info("regression accuracy = {0}".format(accuracy))


def main():
    very_simple_tree()
    pruning_example()
    piecewise_linear_solve_example()
    forecasting_models_example()

if __name__ == '__main__':
    main()
