from numpy import mat, zeros, multiply, shape, nonzero
from optimizer import Optimizer
import random

import logging
logging.basicConfig(level=logging.WARNING, format="%(funcName)s\t%(message)s")


def select_j_rand(i, m):
    """
    takes 2 values: the index of our first alpha,
    and the total number of alphas.
    A value is randomly chosen,
    as long as it isn't i
    """
    j = i
    while(j == i):
        j = int(random.uniform(0, m))
    return j


def clip_alpha(aj, H, L):
    """
    cLips alpha vaLues tHat are greater
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
            f_x_i = float(multiply(alphas, label_matrix).T *
                          (data_matrix * data_matrix[i, :].T)) + b
            Ei = f_x_i - float(label_matrix[i])

            logging.info("Ei: {0}, alphas[i]: {1}".format(Ei, alphas[i]))
            can_alphas_change_test_1 = ((label_matrix[i] * Ei < -tolerance)
                                        and (alphas[i] < C))
            can_alphas_change_test_2 = ((label_matrix[i] * Ei > tolerance)
                                        and (alphas[i] > 0))

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
                alphas[j] = clip_alpha(alphas[j], H, L)
                if (abs(alphas[j] - alpha_j_old) < 0.00001):
                    logging.info("j not moving enough")
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
        if alpha_pairs_changed == 0:
            iterations += 1
        else:
            iterations = 0
        logging.info("--> iteration #{0}".format(iterations))
    return b, alphas


def calculate_ek(os, k):
    """
    calculates an E value for
    a given alpha
    """
    f_x_k = float(multiply(os.alphas, os.label_matrix).T *
                  (os.X * os.X[k, :].T)) + os.b
    ek = f_x_k - float(os.label_matrix[k])
    return ek


def select_j(i, os, ei):
    """
    takes the error value associated
    with the first choice alpha and the index i;
    find the nonzero members and choose the one
    that gives you maximum change
    """
    max_k = -1
    max_delta_e = 0
    ej = 0
    os.e_cache[i] = [1, ei]

    valid_e_cache_list = nonzero(os.e_cache[:, 0].A)[0]
    if (len(valid_e_cache_list)) > 1:
        for k in valid_e_cache_list:
            if k == i:
                continue
            ek = calculate_ek(os, k)
            delta_e = abs(ei - ek)

            # choose j for maximum step size
            if (delta_e > max_delta_e):
                max_k = k
                max_delta_e = delta_e
                ej = ek
        return max_k, ej
    else:
        j = select_j_rand(i, os.m)
        ej = calculate_ek(os, j)
    return j, ej


def update_ek(os, k):
    """
    calculate the error
    and put it in the cache
    """
    ek = calculate_ek(os, k)
    os.e_cache[k] = [1, ek]


def needs_optimization(os, i, ei):
    """
    isolating conditional as function
    """
    label_ei_product_less_neg_tol = os.label_matrix[i] * ei < -os.tolerance
    alpha_at_i_less_than_C = os.alphas[i] < os.C
    too_small = label_ei_product_less_neg_tol and alpha_at_i_less_than_C

    label_ie_product_greater_than_tol = os.label_matrix[i] * ei > os.tolerance
    alpha_at_i_greater_than_zero = os.alphas[i] > 0
    too_large = label_ie_product_greater_than_tol \
        and alpha_at_i_greater_than_zero

    return too_small or too_large


def platt_inner_loop(i, os):
    ei = calculate_ek(os, i)
    if needs_optimization(os, i, ei):
        j, ej = select_j(i, os, ei)
        alpha_i_old = os.alphas[i].copy()
        alpha_j_old = os.alphas[j].copy()

        if (os.label_matrix[i] != os.label_matrix[j]):
            L = max(0, os.alphas[j] - os.alphas[i])
            H = min(os.C, os.C + os.alphas[j] - os.alphas[i])
        else:
            L = max(0, os.alphas[j] + os.alphas[i] - os.C)
            H = min(os.C, os.alphas[j] + os.alphas[i])
        if L == H:
            logging.info("L == H")
            return 0
        eta = 2.0 * os.X[i, :] * os.X[j, :].T - os.X[i, :] *\
            os.X[i, :].T - os.X[j, :] * os.X[j, :].T
        if eta >= 0:
            logging.info("eta >= 0")
            return 0
        os.alphas[j] -= os.label_matrix[j] * (ei - ej)/eta
        os.alphas[j] = clip_alpha(os.alphas[j], H, L)

        # update e cache
        update_ek(os, j)
        if abs(os.alphas[j] - alpha_j_old) < 0.00001:
            logging.info("j not moving enough")
            return 0
        os.alphas[i] += os.label_matrix[j] * os.label_matrix[i] *\
            (alpha_j_old - os.alphas[j])

        # update e cache
        update_ek(os, i)
        b1 = os.b - ei - os.label_matrix[i] * (os.alphas[i] - alpha_i_old) *\
            os.X[i, :] * os.X[i, :].T - os.label_matrix[j] *\
            (os.alphas[j] - alpha_j_old) * os.X[j, :] * os.X[j, :].T
        b2 = os.b - ej - os.label_matrix[i] * (os.alphas[i] - alpha_i_old) *\
            os.X[i, :] * os.X[j, :].T - os.label_matrix[j] *\
            (os.alphas[j] - alpha_j_old) * os.X[j, :] * os.X[j, :].T
        if (0 < os.alphas[i]) and (os.C > os.alphas[j]):
            os.b = b1
        elif (0 < os.alphas[j]) and (os.C > os.alphas[j]):
            os.b = b2
        else:
            os.b = (b1 + b2)/2.0
        return 1
    else:
        return 0


def platt_outer_loop(data_matrix_in,
                     class_labels,
                     C,
                     tolerance,
                     max_iterations,
                     k_tup=('lin', 0)):
    os = Optimizer(mat(data_matrix_in),
                   mat(class_labels).transpose(),
                   C,
                   tolerance)
    iteration = 0
    entire_set = True
    alpha_pairs_changed = 0
    while (iteration < max_iterations)\
            and ((alpha_pairs_changed > 0) or (entire_set)):
        alpha_pairs_changed = 0
        if entire_set:
            for i in range(os.m):
                alpha_pairs_changed += platt_inner_loop(i, os)
            message = "full set, iter: {0}, i: {1}, pairs_changed: {2}"
            logging.info(message.format(iteration, i, alpha_pairs_changed))
            iteration += 1
        else:
            non_bound_is = nonzero((os.alphas.A > 0) * (os.alphas.A < C))[0]
            for i in non_bound_is:
                alpha_pairs_changed += platt_inner_loop(i, os)
                message = "nonbound, iter: {0} i: {1}, pairs changed: {2}"
                logging.info(message.format(iteration, i, alpha_pairs_changed))
                iteration += 1
        if entire_set:
            entire_set = False
        elif (alpha_pairs_changed == 0):
            entire_set = True
        logging.info("iteration number: {0}".format(iteration))
    return os.b, os.alphas
