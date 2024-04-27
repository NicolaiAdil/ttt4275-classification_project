import numpy as np
from typing import Tuple

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
