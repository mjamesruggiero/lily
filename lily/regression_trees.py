import numpy as np


def load_dataset(filepath):
    data_matrix = []
    fr = open(filepath)
    for line in fr.readlines():
        current_line = line.strip().split('\t')
        line_values = map(float, current_line)
        data_matrix.append(line_values)
    return data_matrix


def binary_split_dataset(data_set, feature, value):
    """
    use array filtering to partition data on the given
    feature and value
    """
    matrix_0 = data_set[np.nonzero(data_set[:, feature] > value)[0], :][0]
    matrix_1 = data_set[np.nonzero(data_set[:, feature] <= value)[0], :][0]
    return matrix_0, matrix_1


def reg_leaf(data_set):
    return np.mean(data_set[:, -1])


def reg_err(data_set):
    return np.var(data_set[:, 1]) * np.shape(data_set)[0]


def create_tree(data_set,
                leaf_type=reg_leaf,
                error_type=reg_err,
                ops=(1, 4)):
    """
    First attempts to split the dataset into 2 parts,
    as determined by choose_best_split. If choose_best_split
    hits a stopping condition, it will return None and the value
    for a model type. In the case of regression trees, the model is a
    constant value; in the case of model trees, this is a linear equation.
    If the stopping condition isn't hit, you create a new dict
    and split the dataset into 2 portions, calling create_tree
    recursively on those portions.
    """
    feature, value = choose_best_split(data_set,
                                       leaf_type,
                                       error_type,
                                       ops)
    if feature is None:
        return value
    return_tree = {}
    return_tree['spInd'] = feature
    return_tree['spVal'] = value
    left_set, right_set = binary_split_dataset(data_set, feature, value)
    return_tree['left'] = create_tree(left_set, leaf_type, error_type, ops)
    return_tree['right'] = create_tree(right_set, leaf_type, error_type, ops)
    return return_tree


def choose_best_split(data_set,
                      leaf_type=reg_leaf,
                      error_type=reg_err,
                      ops=(1, 4)):
    tol_s = ops[0]
    tol_n = ops[1]
    if len(set(data_set[:, -1].T.tolist()[0])) == 1:
        return None, leaf_type(data_set)
    m, n = np.shape(data_set)
    s = error_type(data_set)
    best_s = np.inf
    best_index = 0
    best_value = 0

    for feature_index in range(n - 1):
        for split_value in set(data_set[:, feature_index]):
            matrix_0, matrix_1 = binary_split_dataset(data_set,
                                                      feature_index,
                                                      split_value)
            if (np.shape(matrix_0)[0] < tol_n) or \
                    (np.shape(matrix_1)[0] < tol_n):
                continue
            new_s = error_type(matrix_0) + error_type(matrix_1)
            if new_s < best_s:
                best_index = feature_index
                best_value = split_value
                best_s = new_s
    if (s - best_s) < tol_s:
        return None, leaf_type(data_set)
    matrix_0, matrix_1 = binary_split_dataset(data_set,
                                              best_index,
                                              best_value)
    if (np.shape(matrix_0)[0] < tol_n) or (np.shape(matrix_1)[0] < tol_n):
        return None, leaf_type(data_set)
    return best_index, best_value
