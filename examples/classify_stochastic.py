import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import lily
from lily import utils, regression, graphs
from numpy import array

def main():
    """
    Example graph for stochastic gradient ascent
    """
    root_path = os.path.abspath(os.curdir)
    file_path = "data/test_set.txt"
    data_matrix, label_matrix = utils.load_dataset(file_path)
    weights = regression.modified_stochastic_gradient_ascent(array(data_matrix), label_matrix)
    graphs.plot_best_fit(weights, data_matrix, label_matrix)

if __name__ == '__main__':
    main()

