import sys
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Internal imports
from handle_data import load_dataset, split_dataset, create_train_and_test
from plot import (
    plot_characteristics_separability,
)


def main(verbose=False):

    # Load the Iris dataset
    iris = load_iris()
    iris_data = iris.data
    setosa_matrix, versicolor_matrix, virginica_matrix = (
        iris_data[:50],
        iris_data[50:100],
        iris_data[100:150],
    )

    # Plot the characteristics of the iris dataset for each class
    plot_characteristics_separability(
        setosa_matrix, versicolor_matrix, virginica_matrix, iris.feature_names
    )

    


if __name__ == "__main__":
    verbose = False
    if "-v" in sys.argv or "--verbose" in sys.argv:
        verbose = True

    main(verbose)
