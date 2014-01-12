from numpy import mat, shape, zeros, exp

def kernel_transform(X, A, k_tuple):
    """
    use element-wise division
    """
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if k_tuple[0] == 'lin':
        K = X * A.T
    elif k_tuple[0] == 'rbf':
        for j in range(m):
            delta_row = X[j, :] - A
            K[j] = delta_row * delta_row.T
        K = exp(K / (-1 * k_tuple[1]**2))
    else:
        raise NameError("ERROR: kernel not recognized")
    return K


class Optimizer(object):
    """
    Data structure that represents
    SMO values.
    Called an 'optimizer' because it makes
    optimizing the algorithm easier
    """
    def __init__(self, data_matrix_in, label_matrix, C, tolerance, k_tuple):
        self.X = data_matrix_in
        self.label_matrix = label_matrix
        self.C = C
        self.tolerance = tolerance
        self.m = shape(data_matrix_in)[0]
        self.alphas = mat(zeros((self.m, 1)))

        # error cache
        self.e_cache = mat(zeros((self.m, 2)))
        self.b = 0

        #kernel
        self.K = mat(zeros((self.m, self.m)))

        for i in range(self.m):
            self.K[:, i] = kernel_transform(self.X, self.X[i, :], k_tuple)
