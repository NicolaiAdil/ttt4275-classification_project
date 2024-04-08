import sys
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Internal imports
from handle_data import load_dataset, split_dataset, create_train_and_test
from plot import (
    plot_characteristics_separability,
    plot_error_rate,
    plot_confusion_matrix,
)
from train import LinearClassifier, train_and_plot_MSE


def main(verbose=False):

    # Load the Iris dataset
    iris = load_iris()
    iris_data, iris_labels = iris.data, iris.target
    setosa_matrix, versicolor_matrix, virginica_matrix = (
        iris_data[:50],
        iris_data[50:100],
        iris_data[100:150],
    )

    # Plot the characteristics of the iris dataset for each class
    plot_characteristics_separability(
        setosa_matrix, versicolor_matrix, virginica_matrix, iris.feature_names
    )

    # --- Task 1a --- #

    # Split the dataset by class
    train_data, train_labels, test_data, test_labels = create_train_and_test(
        setosa_matrix, versicolor_matrix, virginica_matrix, 30
    )

    # --- Task 1b --- #

    # Train the linear classifier for different step lengths (alpha)
    train_and_plot_MSE(
        train_data,
        train_labels,
        test_data,
        test_labels,
        [0.0025, 0.005, 0.0075, 0.01],
        verbose,
    )

    # --- Task 1c --- #

    # Train the linear classifier with the best step length found from previous attempt
    best_classifier = LinearClassifier(alpha=0.005)

    _, error_rate_vector, error_rate_test_vector = best_classifier.train(
        train_data, train_labels, test_data, test_labels, verbose
    )

    plot_error_rate(
        error_rate_vector,
        error_rate_test_vector,
        "Error rate of the training and test data",
    )

    # Plot the confusion matrix for the training and test data
    plot_confusion_matrix(best_classifier, iris, "train")
    plot_confusion_matrix(best_classifier, iris, "test")

    # --- Task 1d --- #
    test_data, test_labels, train_data, train_labels = create_train_and_test(
        setosa_matrix, versicolor_matrix, virginica_matrix, 20
    )
    train_and_plot_MSE(
        train_data,
        train_labels,
        test_data,
        test_labels,
        [0.0025, 0.005, 0.0075, 0.01],
        verbose,
    )
    best_classifier = LinearClassifier(alpha=0.005)

    _, error_rate_vector, error_rate_test_vector = best_classifier.train(
        train_data, train_labels, test_data, test_labels, verbose
    )

    plot_error_rate(
        error_rate_vector,
        error_rate_test_vector,
        "Error rate of the training and test data",
    )

    # Plot the confusion matrix for the training and test data
    plot_confusion_matrix(best_classifier, iris, "train")
    plot_confusion_matrix(best_classifier, iris, "test")


if __name__ == "__main__":
    verbose = False
    if "-v" in sys.argv or "--verbose" in sys.argv:
        verbose = True

    main(verbose)
