import os
import numpy as np
from typing import Tuple
def load_dataset(relative_path: str = 'iris_data/iris_dataset.csv', verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Get the directory path of the current Python script (such that it also works when using debugger)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path_to_data = os.path.join(current_dir, relative_path)

    data = np.genfromtxt(path_to_data, delimiter=',', dtype=[('sepal_length', float),
                                                             ('sepal_width', float),
                                                             ('petal_length', float),
                                                             ('petal_width', float),
                                                             ('class', 'U20')])
    # Initialize matrices for each class
    setosa_matrix = np.zeros((50, 4))
    versicolor_matrix = np.zeros((50, 4))
    virginica_matrix = np.zeros((50, 4))

    # Populate matrices
    for i, entry in enumerate(data):
        if entry[4] == 'Iris-setosa':
            setosa_matrix[i % 50] = (entry[0], entry[1], entry[2], entry[3])
        elif entry[4] == 'Iris-versicolor':
            versicolor_matrix[i % 50] = (entry[0], entry[1], entry[2], entry[3])
        elif entry[4] == 'Iris-virginica':
            virginica_matrix[i % 50] = (entry[0], entry[1], entry[2], entry[3])
        else:
            print("Error: Unknown class")

    if verbose:
        print("Setosa Matrix:")
        print(setosa_matrix)
        print("\n Length of Setosa Matrix: ", len(setosa_matrix))

        print("\nVersicolor Matrix:")
        print(versicolor_matrix)
        print("\n Length of Versicolor Matrix: ", len(versicolor_matrix))

        print("\nVirginica Matrix:")
        print(virginica_matrix)
        print("\n Length of Virginica Matrix: ", len(virginica_matrix))

    return setosa_matrix, versicolor_matrix, virginica_matrix

load_dataset()