# logistic regression and stochastic gradient descent
import os
from numpy import mat, shape, ones, exp, arange, array, random

def load_dataset():
    """
    TODO file access methods should be extracted
    to utility class
    """
    data_matrix = []
    label_matrix = []

    root_path = os.path.abspath(os.curdir)
    file_path = "{0}/data/test_set.txt".format(root_path)
    fr = open(file_path)
    for line in fr.readlines():
        line_array = line.strip().split()
        data_matrix.append([1.0, float(line_array[0]), float(line_array[1])])
        label_matrix.append(int(line_array[2]))
    return data_matrix, label_matrix


def sigmoid(in_x):
    """
    sigmoid: a bounded differentiable real function
    that is defined for all real input values
    and has a positive derivative at each point
    """
    return 1.0/(1 + exp(-in_x))


def gradient_ascent(data_matrix_in, class_labels):
    """
    Start with the weights all set to 1
    repeat R number of times:
        calculate the gradient of the entire set
        update the weights by alpha * gradient
    return the weights vector
    """
    data_matrix = mat(data_matrix_in)
    label_matrix = mat(class_labels).transpose()
    m, n = shape(data_matrix)
    alpha = 0.001
    max_cycles = 500
    weights = ones((n, 1))
    for k in range(max_cycles):
        h = sigmoid(data_matrix * weights)
        error = (label_matrix - h)
        weights = weights + alpha * data_matrix.transpose() * error
    return weights


def plot_best_fit(weights):
    """
    Graph this
    TODO graph methods should be extracted
    to utility class
    """

    import matplotlib.pyplot as plt
    #weights = wei.getA()
    data_matrix, label_matrix = load_dataset()
    data_array = array(data_matrix)
    n = shape(data_array)[0]
    x_coord_1 = []
    y_coord_1 = []
    x_coord_2 = []
    y_coord_2 = []
    for i in range(n):
        if int(label_matrix[i]) == 1:
            x_coord_1.append(data_array[i, 1])
            y_coord_1.append(data_array[i, 2])
        else:
            x_coord_2.append(data_array[i, 1])
            y_coord_2.append(data_array[i, 2])
    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.scatter(x_coord_1, y_coord_1, s=30, c='red', marker='s')
    ax.scatter(x_coord_2, y_coord_2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stochastic_gradient_ascent(data_matrix, class_labels):
    """
    Start with all the weights set to 1
    for each piece of data in the dataset:
        calculate the gradient of one piece of data
        update the weights vector by alpha * gradient
    return the weights vector
    """
    m, n = shape(data_matrix)
    alpha = 0.01
    weights = ones(n)
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
    m, n = shape(data_matrix)
    weights = ones(n)
    _CONSTANT = 0.0001
    for j in range(iterations):
        data_index = range(m)
        for i in range(m):
            # the alpha decreases with iteration,
            # but never reaches zero because of the constant
            alpha = 4/(1.0 + i + j) + _CONSTANT

            # update vectors chosen randomly
            rand_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_matrix[rand_index] * weights))
            error = class_labels[rand_index] - h
            weights = weights + alpha * error * data_matrix[rand_index]
            del(data_index[rand_index])

    return weights
