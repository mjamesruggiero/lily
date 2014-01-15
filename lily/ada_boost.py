import numpy as np

import logging
logging.basicConfig(level=logging.INFO, format="%(funcName)s\t%(message)s")


def load_simple_data():
    data_matrix = np.matrix([
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
    classification = np.ones((np.shape(data_matrix)[0], 1))
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
    data_matrix = np.mat(data_array)
    label_matrix = np.mat(class_labels).T
    m, n = np.shape(data_matrix)

    number_of_steps = 10.0
    best_stump = {}
    best_class_estimate = np.mat(np.zeros((m, 1)))

    min_error = np.inf
    logging.info("n is {n}".format(n=n))
    for i in range(n):
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
                _errors = np.mat(np.ones((m, 1)))
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


def ada_boost_train_dataset(data_array, class_labels, iterations=40):
    weak_classifications = []
    m = np.shape(data_array)[0]
    D = np.mat(np.ones((m, 1))/m)
    aggregated_class_estimates = np.mat(np.zeros((m, 1)))
    for i in range(iterations):
        best_stump, error, class_estimate = build_stump(data_array,
                                                        class_labels,
                                                        D)
        logging.info("D is {}".format(D.T))
        alpha = float(0.5 * np.log((1.0 - error)/max(error, 1e-16)))
        best_stump['alpha'] = alpha
        weak_classifications.append(best_stump)
        logging.info("class estimate: {est}".format(est=class_estimate.T))
        exponent = np.multiply(-1 * alpha * np.mat(class_labels).T,
                               class_estimate)
        D = np.multiply(D, np.exp(exponent))
        D = D/D.sum()
        aggregated_class_estimates += alpha * class_estimate
        message = "aggregated_class_estimates: {agg}".\
            format(agg=aggregated_class_estimates.T)
        logging.info(message)
        agg_errors = np.multiply(np.sign(aggregated_class_estimates)
                                 != np.mat(class_labels).T, np.ones((m, 1)))
        error_rate = agg_errors.sum() / m
        logging.info("total error: {err}".format(err=error_rate))
        if error_rate == 0.0:
            break
    return weak_classifications


def main():
    D = np.mat(np.ones((5, 1))/5)
    data_matrix, class_labels = load_simple_data()
    stump, min_error, best_estimate = build_stump(data_matrix,
                                                  class_labels,
                                                  D)
    logging.info('stump: {}'.format(stump))
    logging.info('min_error: {}'.format(min_error))
    logging.info('best_estimate: {}'.format(best_estimate))
    classifier_array = ada_boost_train_dataset(data_matrix, class_labels, 9)
    for c in classifier_array:
        logging.info('classifier: {c}'.format(c=c))


if __name__ == '__main__':
    main()