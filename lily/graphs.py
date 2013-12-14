import matplotlib.pyplot as plt
import utils
from numpy import array, shape, arange

def plot_best_fit(weights, data_matrix, label_matrix):
    """
    Graph this
    """
    data_array = array(data_matrix)
    n = shape(data_array)[0]
    x_coord_1 = []
    y_coord_1 = []
    x_coord_2 = []
    y_coord_2 = []
    for i in range(n):
        if int(label_matrix[i]) == 1:
            x_coord_1.append(data_array[i, 1])
            y_coord_1.append(data_array[i, 2])
        else:
            x_coord_2.append(data_array[i, 1])
            y_coord_2.append(data_array[i, 2])
    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.scatter(x_coord_1, y_coord_1, s=30, c='red', marker='s')
    ax.scatter(x_coord_2, y_coord_2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
