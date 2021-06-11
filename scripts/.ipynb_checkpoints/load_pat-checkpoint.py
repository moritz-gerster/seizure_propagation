import numpy as np 

def load_pat(patient):

    path = "../data/normed_topologies/"

    file = open(path + patient + "_normed.txt", "r")

    lines = file.read().split("\n")[:-1]
    file.close()

    values = []
    for line in lines:
        line_split = line.split(", ")
        line_values = [float(value_string) for value_string in line_split]
        values.append(line_values)

    empi_matrix = np.array(values)

    N = empi_matrix.shape[0]

    J = empi_matrix
    J *= 5
    J[np.diag_indices(N)] = 20  # das ist Selbstkopplung

    return J