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


def get_env_config(key='VOTESMART_API_KEY'):
    """
    Grab environment variable
    """
    try:
        return os.environ[key]
    except KeyError:
        message = "Please set the {k} environment variable"
        raise RuntimeError(message.format(k=key))
