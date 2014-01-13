from numpy import matrix, shape, ones, mat, inf, zeros

import logging
logging.basicConfig(level=logging.WARNING, format="%(funcName)s\t%(message)s")


def load_simple_data():
    data_matrix = matrix([
        [1., 2.1],
        [2., 1.1],
        [1.3, 1.],
        [1., 1.],
        [2., 1.]
        ])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_matrix, class_labels


def stump_classify(data_matrix, dimension, threshold, threshold_ineq):
    """
    performs threshold comparisons
    to classify data.
    """
    classification = ones((shape(data_matrix)[0], 1))
    if threshold_ineq == 'lt':
        classification[data_matrix[:, dimension] <= threshold] = -1.0
    else:
        classification[data_matrix[:, dimension] > threshold] = -1.0
    return classification


def build_stump(data_array, class_labels, D):
    """
    Iterate over all the possible inputs to stump_classify
    and find the best decision stump for the dataset.
    'Best' is decided by the weight vector D.
    """
    data_matrix = mat(data_array)
    label_matrix = mat(class_labels).T
    m, n = shape(data_matrix)

    number_of_steps = 10.0
    best_stump = {}
    best_class_estimate = mat(zeros((m, 1)))

    min_error = inf
    for i in range(m):
        range_min = data_matrix[:, i].min()
        range_max = data_matrix[:, i].max()
        step_size = (range_max - range_min)/number_of_steps
        for j in range(-1, int(number_of_steps) + 1):
            for inequal in ['lt', 'gt']:
                threshold = (range_min + float(j) * step_size)
                predicted_values = stump_classify(data_matrix,
                                                  i,
                                                  threshold,
                                                  inequal)
                _errors = mat(ones((m, 1)))
                _errors[predicted_values == label_matrix] = 0
                weighted_error = D.T * _errors

                message = ', '.join([
                    'split: dim {}',
                    'thresh {:03.2f}',
                    'inequal: {}',
                    'weighted_error: {}'
                    ])
                logging.info(message.format(i,
                                            threshold,
                                            inequal,
                                            weighted_error))
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_class_estimate = predicted_values.copy()
                    best_stump['dim'] = i
                    best_stump['threshold'] = threshold
                    best_stump['inequal'] = inequal
        return best_stump, min_error, best_class_estimate


def main():
    D = mat(ones((5, 1))/5)
    data_matrix, class_labels = load_simple_data()
    stump, min_error, best_estimate = build_stump(data_matrix,
                                                  class_labels,
                                                  D)
    logging.info('stump: {}'.format(stump))
    logging.info('min_error: {}'.format(min_error))
    logging.info('best_estimate: {}'.format(best_estimate))


if __name__ == '__main__':
    main()
