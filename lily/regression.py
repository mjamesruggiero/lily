# logistic regression
# stochastic gradient descent
# standard linear regression
# locally weighted linear regression
import numpy as np

import logging
logging.basicConfig(level=logging.INFO, format="%(funcName)s\t%(message)s")


def sigmoid(in_x):
    """
    sigmoid: a bounded differentiable real function
    that is defined for all real input values
    and has a positive derivative at each point
    """
    return 1.0 / (1 + np.exp(-in_x))


def gradient_ascent(data_matrix_in, class_labels):
    """
    Start with the weights all set to 1
    repeat R number of times:
        calculate the gradient of the entire set
        update the weights by alpha * gradient
    return the weights vector
    """
    data_matrix = np.mat(data_matrix_in)
    label_matrix = np.mat(class_labels).transpose()
    m, n = np.shape(data_matrix)
    alpha = 0.001
    max_cycles = 500
    weights = np.ones((n, 1))
    for k in range(max_cycles):
        h = sigmoid(data_matrix * weights)
        error = (label_matrix - h)
        weights = weights + alpha * data_matrix.transpose() * error
    return weights


def stochastic_gradient_ascent(data_matrix, class_labels):
    """
    Start with all the weights set to 1
    for each piece of data in the dataset:
        calculate the gradient of one piece of data
        update the weights vector by alpha * gradient
    return the weights vector
    """
    m, n = np.shape(data_matrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(data_matrix[i] * weights))
        error = class_labels[i] - h
        weights = weights + alpha * error * data_matrix[i]
    return weights


def modified_stochastic_gradient_ascent(data_matrix,
                                        class_labels,
                                        iterations=150):
    """
    The alpha changes with each iteration.
    Note that update vectors are randomly selected.
    """
    m, n = np.shape(data_matrix)
    weights = np.ones(n)
    _CONSTANT = 0.0001
    for j in range(iterations):
        data_index = range(m)
        for i in range(m):
            # the alpha decreases with iteration,
            # but never reaches zero because of the constant
            alpha = 4 / (1.0 + i + j) + _CONSTANT
            # update vectors chosen randomly
            rand_index = int(np.random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_matrix[rand_index] * weights))
            error = class_labels[rand_index] - h
            weights = weights + alpha * error * data_matrix[rand_index]
            del(data_index[rand_index])

    return weights


def classify_vector(in_x, weights):
    """
    takes weights and an input vector
    and calculates the sigmoid;
    more than 0.5 are 1, otherwise zero
    """
    probability = sigmoid(sum(in_x * weights))
    if probability > 0.5:
        return 1.0
    else:
        return 0.0


def standard_regression(x_arr, y_arr):
    """
    compute the best fit line.
    first compute X.T * X and test if its
    determinate is zero. If so, you cannot
    get the inverse. If not, compute the w
    values and return
    """
    x_matrix = np.mat(x_arr)
    y_matrix = np.mat(y_arr).T
    xTx = x_matrix.T * x_matrix
    if np.linalg.det(xTx) == 0.0:
        logging.warning("Matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (x_matrix.T * y_matrix)
    return ws


def locally_weighted_linear_regression(test_point, x_arr, y_arr, k=1.0):
    x_matrix = np.mat(x_arr)
    y_matrix = np.mat(y_arr).T
    m = np.shape(x_matrix)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diff_mat = test_point - x_matrix[j, :]
        # populate the weights with exponentially decaying values
        weights[j, j] = np.exp(diff_mat * diff_mat.T / (-2.0 * k ** 2))
    x_t_x = x_matrix.T * (weights * x_matrix)
    if np.linalg.det(x_t_x) == 0.0:
        logging.warning("This matrix is singular, cannot do inverse")
        return
    ws = x_t_x.I * (x_matrix.T * (weights * y_matrix))
    return test_point * ws


def rss_error(y_arr, y_hat_arr):
    """
    returns a single number
    describing the error of our estimate
    """
    return ((y_arr - y_hat_arr) ** 2).sum()


def ridge_regression(x_matrix, y_matrix, lamb=0.2):
    """
    Ridge regression adds an additional matrix
    lambda-if-I to the matrix X^tX. The matrix I
    is a mxm identity matrix where there are 1s in the diagonal
    elements and zeros everywhere else.
    """
    x_t_x = x_matrix.T * x_matrix

    # eye creates the identity matrix
    denominator = x_t_x * np.eye(np.shape(x_matrix)[1]) * lamb
    if np.linalg.det(denominator) == 0.0:
        logging.warning("The matrix is singular, cannot do inverse")
        return
    ws = denominator.I * (x_matrix.T * y_matrix)
    return ws


def run_ridge_regression(x_arr, y_arr):
    """
    run ridge regression over a number of lambda values;
    remember that you need to normalize the date to give
    each feature equal importance regardless of the units
    in which it was measured. In this case, we normalize
    by subtracting the mean from each feature
    and dividing by the variance.
    """
    x_matrix = np.mat(x_arr)
    y_matrix = np.mat(y_arr).T
    y_mean = np.mean(y_matrix, 0)
    y_matrix = y_matrix - y_mean

    x_means = np.mean(x_matrix, 0)
    x_variance = np.var(x_matrix, 0)
    x_matrix = (x_matrix - x_means) / x_variance
    number_test_pts = 30
    w_matrix = np.zeros((number_test_pts, np.shape(x_matrix)[1]))
    for i in range(number_test_pts):
        ws = ridge_regression(x_matrix, y_matrix, np.exp(i - 10))
        w_matrix[i, :] = ws.T
    return w_matrix


def stage_wise_linear_regression(x_arr,
                                 y_arr,
                                 eps=0.01,
                                 number_iterations=100):
    """
    Greedily optimizes by looping all the possible
    features to see how the error changes if you increase
    or decrease that feature.
    """
    x_matrix = np.mat(x_arr)
    y_matrix = np.mat(y_arr).T
    y_mean = np.mean(y_matrix, 0)
    y_matrix = y_matrix - y_mean
    x_matrix = regularize(x_matrix)
    m, n = np.shape(x_matrix)
    return_mat = np.zeros((number_iterations, n))

    ws = np.zeros((n, 1))
    ws_test = ws.copy()
    ws_max = ws.copy()
    for i in range(number_iterations):
        logging.info("ws.T = {0}".format(ws.T))
        lowest_error = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                ws_test = ws.copy()
                ws_test[j] += eps * sign
                y_test = x_matrix * ws_test
                rss_e = rss_error(y_matrix.A, y_test.A)
                if rss_e < lowest_error:
                    lowest_error = rss_e
                    ws_max = ws_test
        ws = ws_max.copy()
        return_mat[i, :] = ws.T
    return return_mat


def regularize(x_matrix):
    """
    Regularize by columns
    """
    in_matrix = x_matrix.copy()

    # calculate the mean, then subtract it away
    in_means = np.mean(in_matrix, 0)
    in_variance = np.var(in_matrix, 0)
    in_matrix = (in_matrix - in_means) / in_variance
    return in_matrix
