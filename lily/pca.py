import numpy as np


def load_dataset(filepath, delim='\t'):
    fr = open(filepath)
    strings = [line.strip().split(delim) for line in fr.readlines()]
    data = [map(float, line) for line in strings]
    return np.mat(data)


def pca(data_matrix, top_n_features=9999999):
    """the PCA algorithm"""
    mean_values = np.mean(data_matrix, axis=0)
    mean_removed = data_matrix - mean_values
    covariant_matrix = np.cov(mean_removed, rowvar=0)
    eigen_vals, eigen_vects = np.linalg.eig(np.mat(covariant_matrix))
    eigen_val_indices = np.argsort(eigen_vals)
    eigen_val_indices = eigen_val_indices[:-(top_n_features + 1):-1]
    red_eigen_vects = eigen_vects[:, eigen_val_indices]
    low_d_data = mean_removed * red_eigen_vects
    reconstituted_matrix = (low_d_data * red_eigen_vects.T) + mean_values
    return low_d_data, reconstituted_matrix
