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


def plot_best_fit_line():
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

    # this doesn't work, matplotlib throws:
    # "RuntimeError: maximum recursion depth exceeded"
    #ax.plot(x_copy[:, 1], y_hat)
    #plt.show()


def lwlr_test(test_arr, x_arr, y_arr, k=1.0):
    m = np.shape(test_arr)[0]
    y_hat = np.zeros(m)
    for i in range(m):
        y_hat[i] = regression.locally_weighted_linear_regression(
            test_arr[i],
            x_arr,
            y_arr,
            k)

    return y_hat


def main():
    ab_x, ab_y = load_dataset('data/abalone.txt')
    y_hat_01 = lwlr_test(ab_x[0:99],
                         ab_x[0:99],
                         ab_y[0:99],
                         0.1)

    y_hat_1 = lwlr_test(ab_x[0:99],
                        ab_x[0:99],
                        ab_y[0:99],
                        1)

    y_hat_10 = lwlr_test(ab_x[0:99],
                         ab_x[0:99],
                         ab_y[0:99],
                         10)

    logging.info("y_hat for kernel of 0.1 = {}".format(y_hat_01))
    logging.info("y_hat for kernel of 1 = {}".format(y_hat_1))
    logging.info("y_hat for kernel of 10 = {}".format(y_hat_10))

    zero_one_error = regression.rss_error(ab_y[0:99], y_hat_01.T)
    one_error = regression.rss_error(ab_y[0:99], y_hat_1.T)
    ten_error = regression.rss_error(ab_y[0:99], y_hat_10.T)

    logging.info("error for kernel of 0.1 = {}".format(zero_one_error))
    logging.info("error for kernel of 1 = {}".format(one_error))
    logging.info("error for kernel of 10 = {}".format(ten_error))

    # simple linear regression
    ws = regression.standard_regression(ab_x[0:99], ab_y[0:99])
    y_hat = np.mat(ab_x[100:199]) * ws
    simple_linear_regression_err = regression.rss_error(ab_y[100:199],
                                                        y_hat.T.A)
    logging.info("error for linear regression = {}".format(simple_linear_regression_err))

if __name__ == '__main__':
    main()
