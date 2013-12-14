
def load_dataset(file_path):
    data_matrix = []
    label_matrix = []

    fr = open(file_path)
    for line in fr.readlines():
        line_array = line.strip().split()
        data_matrix.append([1.0, float(line_array[0]), float(line_array[1])])
        label_matrix.append(int(line_array[2]))
    return data_matrix, label_matrix
