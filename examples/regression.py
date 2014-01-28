import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import lily.regression as regression
import logging
import numpy as np
logging.basicConfig(level=logging.INFO, format="%(funcName)s\t%(message)s")


def load_dataset(filename):
    number_of_features = len(open(filename).readline().split('\t'))
    data_matrix = []
    label_matrix = []
    fr = open(filename)
    for line in fr.readlines():
        pieces = []
        current_line = line.strip().split('\t')
        for i in range(number_of_features - 1):
            pieces.append(float(current_line[i]))
        data_matrix.append(pieces)
        label_matrix.append(float(current_line[-1]))
    return data_matrix, label_matrix


def main():
    x_arr, y_arr = load_dataset('data/ex0.txt')
    logging.info("x_arr[0:2] = {}".format(x_arr[0:2]))
    ws = regression.standard_regression(x_arr, y_arr)
    logging.info("ws looks like {}".format(ws))

    x_matrix = np.mat(x_arr)
    y_matrix = np.mat(y_arr)
    y_hat = x_matrix * ws

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_matrix[:, 1].flatten().A[0],
               y_matrix.T[:, 0].flatten().A[0])
    x_copy = x_matrix.copy()
    x_copy.sort(0)
    y_hat = x_copy * ws
    logging.info("len(x_copy[:, 1]) = {}".format(len(x_copy[:, 1])))
    logging.info("len(y_hat) = {}".format(len(y_hat)))
    ax.plot(x_copy[:, 1], y_hat)
    #plt.show()


if __name__ == '__main__':
    main()
