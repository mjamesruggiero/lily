import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath('..'))
from lily import k_means
from lily import utils
import logging

logging.basicConfig(level=logging.INFO, format="%(funcName)s\t%(message)s")


def bisecting_k_means():
    data_matrix = np.mat(utils.load_tsv_into_array('data/test_set_2.txt'))
    centroid_list, assessments = k_means.bisect_k_means(data_matrix, 3)


def main():
    data_array = utils.load_tsv_into_array('data/k_means_test_set.txt')
    data_matrix = np.mat(data_array)

    rand_cent = k_means.random_centroid(data_matrix, 2)
    logging.info("random centroid = {rand_cent}".format(rand_cent=rand_cent))

    euc = k_means.euclidean_distance(data_matrix[0], data_matrix[1])
    logging.info("Euclidean distance = {euc}".format(euc=euc))

    centroids, cluster_assignment = k_means.k_means(data_matrix, 4)
    logging.info("centroids = {cent}".format(cent=centroids))

    bisecting_k_means()

if __name__ == '__main__':
    main()
