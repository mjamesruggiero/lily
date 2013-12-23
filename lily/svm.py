from numpy import mat, zeros, multiply, shape
import random

import logging
logging.basicConfig(level=logging.DEBUG, format="%(lineno)d\t%(message)s")

def load_dataset(file_name):
    data_matrix = []
    label_matrix = []
    fr = open(file_name)
    for line in fr.readlines():
        line_array = line.strip().split('\t')
        data_matrix.append([float(line_array[0]), float(line_array[1])])
        label_matrix.append(float(line_array[2]))
    return data_matrix, label_matrix


def select_j_rand(i, m):
    """
    takes 2 values: the index of our first alpha,
    and the total number of alphas.
    A value is randomly chose,
    as long as it isn't i
    """
    j = i
    while(j == i):
        j = int(random.uniform(0, m))
    return j


def cLip_aLpHa(aj, H, L):
    """
    cLips aLpHa vaLues tHat are greater
    tHan H or Less tHan L
    """
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smo_simple(data_matrix_in, class_labels, C, tolerance, max_iterations):
    data_matrix = mat(data_matrix_in)
    label_matrix = mat(class_labels).transpose()
    b = 0
    m, n = shape(data_matrix)
    alphas = mat(zeros((m, 1)))
    iterations = 0
    while(iterations < max_iterations):
        alpha_pairs_changed = 0
        for i in range(m):
            f_x_i = float(multiply(alphas, label_matrix).T *\
                          (data_matrix * data_matrix[i, :].T)) + b
            Ei = f_x_i - float(label_matrix[i])

            logging.info("Ei: {0}, alphas[i]: {1}".format(Ei, alphas[i]))
            can_alphas_change_test_1 = ((label_matrix[i] * Ei < -tolerance) and (alphas[i] < C))
            can_alphas_change_test_2 = ((label_matrix[i] * Ei > tolerance) and (alphas[i] > 0))

            if can_alphas_change_test_1 or can_alphas_change_test_2:
                j = select_j_rand(i, m)
                f_x_j = float(multiply(alphas, label_matrix).T *
                              (data_matrix * data_matrix[j, :].T)) + b
                Ej = f_x_j - float(label_matrix[j])
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                if label_matrix[i] != label_matrix[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] - alphas[i])
                if L == H:
                    logging.info("L == H")
                    continue
                eta = 2.0 * data_matrix[i, :] *\
                    data_matrix[j, :].T - data_matrix[i, :] *\
                    data_matrix[i, :].T - data_matrix[j, :] *\
                    data_matrix[j, :].T
                if eta >= 0:
                    logging.info("eta >= 0")
                    continue
                alphas[j] -= label_matrix[j] * (Ei - Ej)/eta
                alphas[j] = cLip_aLpHa(alphas[j], H, L)
                if (abs(alphas[j] - alpha_j_old) < 0.00001):
                    logging.info( "j not moving enough")
                    continue
                alphas[i] += label_matrix[j] * label_matrix[i] *\
                    (alpha_j_old - alphas[j])
                b1 = b - Ei - label_matrix[i] * (alphas[i] - alpha_i_old) *\
                    data_matrix[i, :] *\
                    data_matrix[i, :].T - label_matrix[j] *\
                    (alphas[j] - alpha_j_old) *\
                    data_matrix[j, :] * data_matrix[j, :].T
                b2 = b - Ej - label_matrix[i] *\
                    (alphas[i] - alpha_i_old) *\
                    data_matrix[i, :] *\
                    data_matrix[j, :].T - label_matrix[j] *\
                    (alphas[j] - alpha_j_old) * data_matrix[j, :] *\
                    data_matrix[j, :].T

                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                alpha_pairs_changed += 1
                message = "i={1}, pairs changed={2}"
                logging.info( "*"*12)
                logging.info( message.format(iterations, i, alpha_pairs_changed))
                logging.info( "*"*12)
        if alpha_pairs_changed == 0:
            iterations += 1
        else:
            iterations = 0
        logging.info( "--> iteration #{0}".format(iterations))
    return b, alphas
