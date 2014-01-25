# logistic regression and stochastic gradient descent
from numpy import mat, shape, ones, exp, random


def sigmoid(in_x):
    """
    sigmoid: a bounded differentiable real function
    that is defined for all real input values
    and has a positive derivative at each point
    """
    return 1.0 / (1 + exp(-in_x))


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
            alpha = 4 / (1.0 + i + j) + _CONSTANT
            # update vectors chosen randomly
            rand_index = int(random.uniform(0, len(data_index)))
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
