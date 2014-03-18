import os


def load_dataset(file_path):
    data_matrix = []
    label_matrix = []

    fr = open(file_path)
    for line in fr.readlines():
        line_array = line.strip().split()
        data_matrix.append([1.0, float(line_array[0]), float(line_array[1])])
        label_matrix.append(int(line_array[2]))
    return data_matrix, label_matrix


def load_tsv_datafile(filename):
    """
    load TSV datafile into list of data
    and corresponding labels;
    feature labels sit
    in the last (farthest-right) column
    (i.e. a 1 or a 0)
    """
    number_of_features = len(open(filename).readline().split('\t'))
    data_matrix = []
    label_matrix = []
    fr = open(filename)
    for line in fr.readlines():
        pieces = []
        current_line = line.strip().split('\t')
        for i in range(number_of_features - 1):
            pieces.append(float(current_line[i]))
        data_matrix.append(pieces)
        label_matrix.append(float(current_line[-1]))
    return data_matrix, label_matrix


def get_env_config(key='VOTESMART_API_KEY'):
    """
    Grab environment variable
    """
    try:
        return os.environ[key]
    except KeyError:
        message = "Please set the {k} environment variable"
        raise RuntimeError(message.format(k=key))
