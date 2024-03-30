import matplotlib.pyplot as plt
import numpy as np


def plot_characteristics_separability(setosa_matrix: np.ndarray, versicolor_matrix: np.ndarray, virginica_matrix: np.ndarray) -> None:
    """
    Plots the characteristics of the iris dataset for each class.

    Parameters:
        setosa_matrix (np.ndarray): The matrix containing the characteristics of the setosa class.
        versicolor_matrix (np.ndarray): The matrix containing the characteristics of the versicolor class.
        virginica_matrix (np.ndarray): The matrix containing the characteristics of the virginica class.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 8))

    # Setosa
    axs[0].scatter(setosa_matrix[:, 0], setosa_matrix[:, 1], color='r', label='Setosa')
    # Versicolor
    axs[0].scatter(versicolor_matrix[:, 0], versicolor_matrix[:, 1], color='g', label='Versicolor')
    # Virginica
    axs[0].scatter(virginica_matrix[:, 0], virginica_matrix[:, 1], color='b', label='Virginica')

    axs[0].set_xlabel('Sepal Length')
    axs[0].set_ylabel('Sepal Width')
    axs[0].set_title('Sepal Characteristics of Iris Dataset')
    axs[0].legend()

    # Plot petal length and width
    # Setosa
    axs[1].scatter(setosa_matrix[:, 2], setosa_matrix[:, 3], color='r', label='Setosa')
    # Versicolor
    axs[1].scatter(versicolor_matrix[:, 2], versicolor_matrix[:, 3], color='g', label='Versicolor')
    # Virginica
    axs[1].scatter(virginica_matrix[:, 2], virginica_matrix[:, 3], color='b', label='Virginica')

    axs[1].set_xlabel('Petal Length')
    axs[1].set_ylabel('Petal Width')
    axs[1].set_title('Petal Characteristics of Iris Dataset')
    axs[1].legend()

    plt.tight_layout()

    plt.show()
