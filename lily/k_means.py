import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format="%(funcName)s\t%(message)s")


def euclidean_distance(vec_a, vec_b):
    squares = np.power(vec_a - vec_b, 2)
    sum_of_squares = squares.sum(axis=-1)
    return np.sqrt(sum_of_squares)[0, 0]


def random_centroid(dataset, k):
    n = np.shape(dataset)[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        min_j = min(dataset[:, j])
        range_j = float(max(dataset[:, j]) - min_j)
        centroids[:, j] = min_j + range_j * np.random.rand(k, 1)
    return centroids


def k_means(dataset,
            k,
            distance_measure=euclidean_distance,
            create_centroids=random_centroid):
    """
    Create k centroids, then assign each point to the closest
    centroid. Then re-calculate the centroids. Repeat until the
    points stop changing clusters.
    """
    m = np.shape(dataset)[0]
    cluster_assessment = np.mat(np.zeros((m, 2)))
    centroids = create_centroids(dataset, k)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_distance = np.inf
            min_index = -1
            for j in range(k):
                dist_j_i = distance_measure(centroids[j, :], dataset[i, :])
                if dist_j_i < min_distance:
                    min_distance = dist_j_i
                    min_index = j
            if cluster_assessment[i, 0] != min_index:
                cluster_changed = True
            cluster_assessment[i, :] = min_index, min_distance ** 2
        logging.info("centroids: {}".format(centroids))
        for cent in range(k):
            points_in_cluster = dataset[np.nonzero(cluster_assessment[:, 0].
                                                   A == cent)[0]]
            centroids[cent, :] = np.mean(points_in_cluster, axis=0)
    return centroids, cluster_assessment


def bisect_k_means(dataset, k, distance_measure=euclidean_distance):
    """
    Choose the cluster with the largest SSE and split it.
    Then repeat until you get the k number of clusters.
    For every cluster, you measure the total error, perform
    k-means with k=2, measure the error after k-means has split
    the cluster in two, choose to keep the cluster split that gives
    the lowest error
    """
    m = np.shape(dataset)[0]
    cluster_assessment = np.mat(np.zeros((m, 2)))
    centroid_0 = np.mean(dataset, axis=0).tolist()[0]
    centroid_list = [centroid_0]

    # set the original error
    for j in range(m):
        cluster_assessment[j, 1] = distance_measure(np.mat(centroid_0),
                                                    dataset[j, :]) ** 2

    while (len(centroid_list) < k):
        lowest_sse = np.inf
        for i in range(len(centroid_list)):
            points_in_current_cluster =\
                dataset[np.nonzero(cluster_assessment[:, 0].A == i)[0], :]
            centroid_matrix, split_cluster_assessment =\
                k_means(points_in_current_cluster, 2, distance_measure)

            # compare the SSE to the current minimum
            sse_split = sum(split_cluster_assessment[:, 1])
            nosplit_index = np.nonzero(cluster_assessment[:, 0].A != i)
            sse_not_split =\
                sum(cluster_assessment[nosplit_index[0], 1])

            logging.info("sse split:\n{split}".format(split=sse_split))
            logging.info("sse NOT split:\n{not_split}".
                         format(not_split=sse_not_split))

            if (sse_split + sse_not_split) < lowest_sse:
                best_centroid_to_split = i
                best_new_centroids = centroid_matrix
                best_cluster_assessmnt = split_cluster_assessment.copy()
                lowest_sse = sse_split + sse_not_split

        best_cluster_index = np.nonzero(best_cluster_assessmnt[:, 0].A == 1)[0]
        best_cluster_assessmnt[best_cluster_index, 0] = len(centroid_list)

        best_cluster_index = np.nonzero(best_cluster_assessmnt[:, 0].A == 0)[0]
        best_cluster_assessmnt[best_cluster_index, 0] = best_centroid_to_split

        logging.info("the best centroid on which to split = {best}".
                     format(best=best_centroid_to_split))
        logging.info("the length of best_cluster_assessmnt = {cluster_len}".
                     format(cluster_len=len(best_cluster_assessmnt)))

        centroid_list[best_centroid_to_split] =\
            best_new_centroids[0, :].tolist()[0]
        centroid_list.append(best_new_centroids[1, :])

        asssigning_cluster_index =\
            np.nonzero(cluster_assessment[:, 0].A == best_centroid_to_split)[0]
        cluster_assessment[asssigning_cluster_index, :] =\
            best_cluster_assessmnt

    return np.mat(centroid_list), cluster_assessment
