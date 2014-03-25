import numpy as np

import logging
logging.basicConfig(level=logging.INFO, format="%(funcName)s\t%(message)s")


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

        # how large should your step size be?
        step_size = (range_max - range_min) / number_of_steps
        for j in range(-1, int(number_of_steps) + 1):
            for inequal in ['lt', 'gt']:
                threshold = (range_min + float(j) * step_size)
                predicted_values = stump_classify(data_matrix,
                                                  i,
                                                  threshold,
                                                  inequal)
                _errors = np.mat(np.ones((m, 1)))

                # _errors is 1 for any value in predicted_values
                # that isn't equal to the label
                _errors[predicted_values == label_matrix] = 0
                weighted_error = D.T * _errors

                message = ', '.join(['split: dim {}',
                                     'thresh {:03.2f}',
                                     'inequal: {}',
                                     'weighted_error: {}'])
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


def train_dataset(data_array, class_labels, iterations=40):
    weak_classifications = []
    m = np.shape(data_array)[0]
    D = np.mat(np.ones((m, 1)) / m)
    aggregated_class_estimates = np.mat(np.zeros((m, 1)))

    for i in range(iterations):
        best_stump, err, class_estimate = build_stump(data_array,
                                                      class_labels,
                                                      D)
        logging.info("D is {}".format(D.T))
        alpha = get_alpha(err)
        best_stump['alpha'] = alpha
        weak_classifications.append(best_stump)

        logging.info("class estimate: {est}".format(est=class_estimate.T))
        exponent = np.multiply(-1 * alpha * np.mat(class_labels).T,
                               class_estimate)
        D = np.multiply(D, np.exp(exponent))
        D = D / D.sum()

        aggregated_class_estimates += alpha * class_estimate
        message = "aggregated_class_estimates: {agg}".\
            format(agg=aggregated_class_estimates.T)
        logging.info(message)

        error_rate = aggregated_error_rate(aggregated_class_estimates,
                                           class_labels,
                                           m)
        logging.info("total error: {err}".format(err=error_rate))
        if error_rate == 0.0:
            break
    return weak_classifications, aggregated_class_estimates


def aggregated_error_rate(estimates, class_labels, m):
    agg_errors = np.multiply(np.sign(estimates) != np.mat(class_labels).T,
                             np.ones((m, 1)))
    return agg_errors.sum() / m


def get_alpha(err):
    """
    NB: the 1e-16 is insurance against
    divide-by-zero errors
    """
    return float(0.5 * np.log((1.0 - err) / max(err, 1e-16)))


def classify(data_to_classify, classifiers):
    """classifies with a train of weak classifiers"""
    data_matrix = np.mat(data_to_classify)
    m = np.shape(data_matrix)[0]

    # initialize to zeros
    aggregated_class_estimate = np.mat(np.zeros((m, 1)))
    for i in range(len(classifiers)):
        class_estimate = stump_classify(data_matrix, classifiers[i]['dim'],
                                        classifiers[i]['threshold'],
                                        classifiers[i]['inequal'])
        aggregated_class_estimate += classifiers[i]['alpha'] * class_estimate
    return np.sign(aggregated_class_estimate)


def plot_roc(prediction_strengths, class_labels):
    """
    ROC curve: receiver operating characteristic
    the X-axis is the number of false positives,
    the Y-axis is the number of true positives
    AUC: area under the curve
    """
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)
    y_sum = 0.0
    number_positive_classes = sum(np.array(class_labels) == 1.0)
    y_step = 1 / float(number_positive_classes)
    x_step = 1 / float(len(class_labels) - number_positive_classes)
    sorted_indices = prediction_strengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sorted_indices.tolist()[0]:
        if class_labels[index] == 1.0:
            del_x = 0
            del_y = y_step
        else:
            del_x = x_step
            del_y = 0
            y_sum += cur[1]
        ax.plot([cur[0], cur[0] - del_x], [cur[1], cur[1] - del_y], c='b')
        cur = (cur[0] - del_x, cur[1] - del_y)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0, 1, 0, 1])
    plt.show()
    logging.info("the area under the curve (AUC) is {}".format(y_sum * x_step))
