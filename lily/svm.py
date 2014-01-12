from numpy import mat, zeros, multiply, shape, nonzero
from optimizer import Optimizer
import random

import logging
logging.basicConfig(level=logging.WARNING, format="%(funcName)s\t%(message)s")

"""
Sequential Minimal Optimization

Breaks large optimzation problems into smaller problems.
The smaller problems can easily be solved, and solving them sequentially
will give the same answer as trying to solve everything together.

SMO works to find a set of alphas and b.
Once we have a set of alphas, we can easily compute our weights w
and get the separating hyperplane.

SMO chooses two alphas to optimize on each cycle.
Once a suitable pair of alphas is found, one is increased
and one is decreased. To be suitable, a set of alphas must
meet certain criteria.

One: both of the alphas have to be outside their margin boundary.
Two: the alphas aren't already clamped or bounded.
"""


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


def calculate_ek(os, k):
    """
    calculates an E value for
    a given alpha
    """
    f_x_k = float(multiply(os.alphas, os.label_matrix).T * os.K[:, k] + os.b)
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
            message = "L == H; i = {i_val}, ei = {ei}".format(i_val=i, ei=ei)
            logging.info(message)
            return 0
        eta = 2.0 * os.K[i, j] - os.K[i, i] - os.K[j, j]
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
            os.K[i, i] - os.label_matrix[j] * (os.alphas[j] - alpha_j_old) *\
            os.K[i, j]
        b2 = os.b - ej - os.label_matrix[i] * (os.alphas[i] - alpha_i_old) *\
            os.K[i, j] - os.label_matrix[j] * (os.alphas[j] - alpha_j_old) *\
            os.K[j, j]
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
                     k_tuple=('lin', 0)):
    os = Optimizer(mat(data_matrix_in),
                   mat(class_labels).transpose(),
                   C,
                   tolerance, k_tuple)
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


def calculate_ws(alphas, data_array, class_labels):
    """
    get the hpyerplane from the alphas
    by computing the w value.
    note that if the alphas are zero, they don't
    "matter"
    """
    X = mat(data_array)
    label_matrix = mat(class_labels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * label_matrix[i], X[i, :].T)
    return w
