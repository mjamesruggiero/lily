import numpy as np
from numpy import linalg as la
import logging
logging.basicConfig(level=logging.DEBUG, format="%(lineno)d\t%(message)s")


def euclidean_similarity(a_group, b_group):
    return 1.0 / (1.0 + la.norm(a_group - b_group))


def pearson_similarity(a_group, b_group):
    if len(a_group) < 3:
        return 1.0
    return 0.5 + 0.5 * np.corrcoef(a_group, b_group, rowvar=0)[0][1]


def cosine_similarity(a_group, b_group):
    numerator = float(a_group.T * b_group)
    denominator = la.norm(a_group) * la.norm(b_group)
    return 0.5 + 0.5 * (numerator / denominator)


def estimated_rating(data_matrix, user, similarity_measure, item):
    similarity_total = 0.0
    rat_similarity_total = 0.0
    n = np.shape(data_matrix)[1]
    for j in range(n):
        user_rating = data_matrix[user, j]
        if user_rating == 0:
            continue
        overlap = np.nonzero(np.logical_and(data_matrix[:, item].A > 0,
                                            data_matrix[:, j].A > 0))[0]
        if len(overlap) == 0:
            similarity = 0
        else:
            similarity = similarity_measure(data_matrix[overlap, item],
                                            data_matrix[overlap, j])

        message = "the {i} and {j} similarity is {sim}"
        logging.info(message.format(i=item,
                                    j=j,
                                    sim=similarity))
        similarity_total += similarity
        rat_similarity_total += similarity * user_rating

    if similarity_total == 0:
        return 0
    else:
        return rat_similarity_total / similarity_total


def recommend(data_matrix,
              user,
              n=3,
              similarity_measure=cosine_similarity,
              estimated_method=estimated_rating):

    unrated_items = np.nonzero(data_matrix[user, :].A == 0)[1]
    if len(unrated_items) == 0:
        logging.info('you rated everything')
        return None

    item_scores = []
    for item in unrated_items:
        estimated_score = estimated_method(data_matrix,
                                           user,
                                           similarity_measure,
                                           item)
        item_scores.append((item, estimated_score))
    return sorted(item_scores, key=lambda jj: jj[1], reverse=True)[:n]


def svd_estimator(data_matrix, user, similarity_measure, item):
    similarity_total = 0.0
    rat_similarity_total = 0.0
    n = np.shape(data_matrix)[1]

    u, sigma, vt = la.svd(data_matrix)
    sigma_4 = np.mat(np.eye(4) * sigma[:4])
    transformed_items = data_matrix.T * u[:, :4] & sigma_4.I

    for j in range(n):
        user_rating = data_matrix[user, j]
        if user_rating == 0 or j == item:
            continue
        similarity = similarity_measure(transformed_items[item, :].T,
                                        transformed_items[j, :].T)
        message = "the {i} and {j} similarity is: {sim}"
        logging.info(message.format(i=item, j=j, sim=similarity))

        similarity_total += similarity
        rat_similarity_total += similarity * user_rating

    if similarity_total == 0:
        return 0
    else:
        return rat_similarity_total / similarity_total
