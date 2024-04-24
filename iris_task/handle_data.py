import os
import numpy as np
from typing import Tuple


def load_dataset(
    relative_path: str = "iris_data/iris_dataset.csv", verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the Iris dataset from a CSV file to RAM.

    Parameters:
    relative_path (str): The relative path to the CSV file. Default is 'iris_data/iris_dataset.csv'.
    verbose (bool): If True, print additional information about the loaded dataset. Default is False.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing three numpy arrays representing the setosa, versicolor, and virginica matrices respectively.
    """

    # Get the directory path of the current Python script (such that it also works when using debugger)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path_to_data = os.path.join(current_dir, relative_path)

    data = np.genfromtxt(
        path_to_data,
        delimiter=",",
        dtype=[
            ("sepal_length", float),
            ("sepal_width", float),
            ("petal_length", float),
            ("petal_width", float),
            ("class", "U20"),
        ],
    )
    # Initialize matrices for each class
    setosa_matrix = np.zeros((50, 4))
    versicolor_matrix = np.zeros((50, 4))
    virginica_matrix = np.zeros((50, 4))

    # Populate matrices
    for i, entry in enumerate(data):
        class_name = entry[4]
        if class_name == "Iris-setosa":
            setosa_matrix[i % 50] = (entry[0], entry[1], entry[2], entry[3])
        elif class_name == "Iris-versicolor":
            versicolor_matrix[i % 50] = (entry[0], entry[1], entry[2], entry[3])
        elif class_name == "Iris-virginica":
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


def split_dataset(
    iris_matrix: np.ndarray, split_index: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits the given iris_matrix into two parts based on the split_index.

    Parameters:
        iris_matrix (np.ndarray): The input iris matrix.
        split_index (int): The index at which to split the iris matrix.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two parts of the iris matrix,
        where the first part contains the rows from 0 to split_index-1, and the second part
        contains the rows from split_index to the end.
    """

    if split_index >= len(iris_matrix):
        raise ValueError(
            "split_index must be smaller than the number of rows in iris_matrix"
        )
    return iris_matrix[:split_index], iris_matrix[split_index:]


def create_train_and_test(
    setosa_matrix, versicolor_matrix, virginica_matrix, split_index: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates a train and test set from the given matrices.

    Parameters:
        setosa_matrix (np.ndarray): The matrix containing the setosa data.
        versicolor_matrix (np.ndarray): The matrix containing the versicolor data.
        virginica_matrix (np.ndarray): The matrix containing the virginica data.
        split_index (int): The index at which to split the matrices.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the train and test data and labels for each class.
    """

    train_setosa, test_setosa = split_dataset(setosa_matrix, split_index)
    train_setosa_labels, test_setosa_labels = np.zeros(split_index), np.zeros(
        50 - split_index
    )

    train_versicolor, test_versicolor = split_dataset(versicolor_matrix, split_index)
    train_versicolor_labels, test_versicolor_labels = np.ones(split_index), np.ones(
        50 - split_index
    )

    train_virginica, test_virginica = split_dataset(virginica_matrix, split_index)
    train_virginica_labels, test_virginica_labels = np.full(split_index, 2), np.full(
        50 - split_index, 2
    )

    train_data = np.concatenate((train_setosa, train_versicolor, train_virginica))
    train_labels = np.concatenate(
        (train_setosa_labels, train_versicolor_labels, train_virginica_labels)
    )

    test_data = np.concatenate((test_setosa, test_versicolor, test_virginica))
    test_labels = np.concatenate(
        (test_setosa_labels, test_versicolor_labels, test_virginica_labels)
    )

    return train_data, train_labels, test_data, test_labels
