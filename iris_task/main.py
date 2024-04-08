import sys
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Internal imports
from handle_data import load_dataset, split_dataset, create_train_and_test
from plot import plot_characteristics_separability, plot_error_rate
from train import LinearClassifier


def main(verbose=False):

    # Load the Iris dataset
    iris = load_iris()
    iris_data, iris_labels = iris.data, iris.target
    setosa_matrix, versicolor_matrix, virginica_matrix = (
        iris_data[:50],
        iris_data[50:100],
        iris_data[100:150],
    )
    if verbose:
        print(
            "Dimensions of data and labels respectively:",
            iris_data.shape,
            iris_labels.shape,
        )
        # Plot the characteristics of the iris dataset for each class
        plot_characteristics_separability(
            setosa_matrix, versicolor_matrix, virginica_matrix, iris.feature_names
        )

    # --- Task 1a --- #

    # Split the dataset by class
    # TODO: to this better
    train_data, train_labels, test_data, test_labels = create_train_and_test(
        setosa_matrix, versicolor_matrix, virginica_matrix, 30
    )

    # --- Task 1b --- #

    # Train the linear classifier for different step lengths (alpha)
    alphas = [0.0025, 0.005, 0.0075, 0.01]
    for alpha in alphas:
        classifier = LinearClassifier(alpha=alpha)
        loss_vector, _, _ = classifier.train(
            train_data, train_labels, test_data, test_labels, verbose
        )
        plt.plot(loss_vector, label=f"Alpha: {alpha}")
    # Plot the MSE for each step length
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    # plt.title("MSE of the training data")
    plt.legend()
    plt.show()

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
    for train_or_test in ["train", "test"]:
        confusion_matrix = best_classifier.get_confusion_matrix(
            train_or_test=train_or_test
        )
        # Plot the confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix for {train_or_test} data")
        plt.show()

    # --- Task 1d --- #


if __name__ == "__main__":
    verbose = False
    if "-v" in sys.argv or "--verbose" in sys.argv:
        verbose = True

    main(verbose)
