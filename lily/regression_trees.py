import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format="%(funcName)s\t%(message)s")


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


def is_tree(obj):
    return type(obj).__name__ == 'dict'


def get_mean(tree):
    """
    Descends a tree until it hits only leaf nodes. When it
    finds two leaf nodes it takes the average of those two nodes.
    """
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])
    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, test_data):
    """
    Collapse the tree if there is no test data
    """
    if np.shape(test_data)[0] == 0:
        return get_mean(tree)
    if is_tree(tree['right']) or is_tree(tree['left']):
        l_set, r_set = binary_split_dataset(test_data,
                                            tree['spInd'],
                                            tree['spVal'])
    if is_tree(tree['left']):
        tree['left'] = prune(tree['left'], l_set)
    if is_tree(tree['right']):
        tree['right'] = prune(tree['right'], r_set)

    # neither one are trees; now you can merge
    if not is_tree(tree['left']) and not is_tree(tree['right']):
        l_set, r_set = binary_split_dataset(test_data,
                                            tree['spInd'],
                                            tree['spVal'])
        left_merge_err = sum(np.power(l_set[:, -1] - tree['left'], 2))
        right_merge_err = sum(np.power(r_set[:, -1] - tree['right'], 2))
        error_no_merge = left_merge_err + right_merge_err
        tree_mean = (tree['left'] + tree['right']) / 2.0
        error_merge = sum(np.power(test_data[:, -1] - tree_mean, 2))
        if error_merge < error_no_merge:
            message = "merging and returning tree_mean: {0}"
            logging.info(message.format(tree_mean))
            return tree_mean
        else:
            return tree
    else:
        return tree


def linearly_solve(data_set):
    """
    Format the dataset into the target variable Y
    and the independent variable X.
    Perform some simple linear regression.
    """
    m, n = np.shape(data_set)
    x = np.mat(np.ones((m, n)))
    y = np.mat(np.ones((m, 1)))
    x[:, 1:n] = data_set[:, 0:n - 1]
    y = data_set[:, -1]
    x_t_x = x.T * x
    if np.linalg.det(x_t_x) == 0.0:
        raise NameError("This matrix is singular, cannot do inverse;\n\
                        try increasing the second value of 'ops'")
    weights = x_t_x.I * (x.T * y)
    return weights, x, y


def model_leaf(data_set):
    weights, x, y = linearly_solve(data_set)
    return weights


def model_error(data_set):
    weights, x, y = linearly_solve(data_set)
    y_hat = x * weights
    return sum(np.power(y - y_hat, 2))


def tree_evaluation(model, _):
    return float(model)


def model_tree_evaluation(model, input_data):
    n = np.shape(input_data)[1]
    x = np.mat((1, n + 1))
    x[:, 1: n + 1] = input_data
    return float(x * model)


def tree_forecast(tree, input_data, model_evaluator=tree_evaluation):
    """
    Gives one forecast for one data point, for a given tree.
    """
    if not is_tree(tree):
        return model_evaluator(tree, input_data)
    if input_data[tree['spInd']] > tree['spVal']:
        if is_tree(tree['left']):
            return tree_forecast(tree['left'], input_data, model_evaluator)
        else:
            return model_evaluator(tree['left'], input_data)
    else:
        if is_tree(tree['right']):
            return tree_forecast(tree['right'], input_data, model_evaluator)
        else:
            return model_evaluator(tree['right'], input_data)


def create_forecast(tree, test_data, model_evaluator=tree_evaluation):
    m = len(test_data)
    y_hat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        y_hat[i, 0] = tree_forecast(tree,
                                    np.mat(test_data[i]),
                                    model_evaluator)
    return y_hat
