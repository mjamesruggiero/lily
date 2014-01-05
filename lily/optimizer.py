from numpy import mat, shape, zeros

class Optimizer(object):
    """
    data structure that represents
    all the important SMO values
    """
    def __init__(self, data_matrix_in, label_matrix, C, tolerance):
        self.X = data_matrix_in
        self.label_matrix = label_matrix
        self.C = C
        self.tolerance = tolerance
        self.m = shape(data_matrix_in)[0]
        self.alphas = mat(zeros((self.m, 1)))

        # error cache
        self.e_cache = mat(zeros((self.m, 2)))
        self.b = 0
