import sys
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

# Internal imports
from handle_data import create_train_and_test
from plot import (
    plot_characteristics_separability,
    plot_error_rate,
    plot_confusion_matrix,
    plot_characteristics_separability_histogram,
)
from train import (
    LinearClassifier,
    train_and_plot_MSE,
)


def main(verbose=False):

    while True:
        print("\nMenu:")
        print("1: Plot the characteristics of the iris dataset for each class")
        print("2: First 30 samples for training, last 20 samples for testing")
        print("3: First 20 samples for training, last 30 samples for testing")
        print("4: First 30 samples for training, remove sepal width")
        print("5: First 30 samples for training, remove setal width and sepal length")
        print(
            "6: First 30 samples for training, remove sepal width, setal length and petal length"
        )
        print("7: Plot histograms for each feature and class")
        print("q: Quit")

        choice = input("Your choice: ")

        if choice.lower() == "q":
            break

        # Load the Iris dataset
        iris = load_iris()
        iris_data, iris_labels = iris.data, iris.target
        setosa_matrix, versicolor_matrix, virginica_matrix = (
            iris_data[:50],
            iris_data[50:100],
            iris_data[100:150],
        )

        alphas = [0.0025, 0.005, 0.0075, 0.01]

        # Plot the characteristics of the iris dataset for each class
        if choice == "1":
            # Plot the characteristics of the iris dataset for each class
            plot_characteristics_separability(
                setosa_matrix, versicolor_matrix, virginica_matrix, iris.feature_names
            )

        # Use first 30 samples for training and last 20 samples for testing
        if choice == "2":

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
                alphas,
                "First 30 samples for training, last 20 samples for testing",
                verbose,
            )

            # --- Task 1c --- #

            # Train the linear classifier with the best step length found from previous attempt
            best_alpha = 0.005
            best_classifier = LinearClassifier(alpha=best_alpha)

            _, error_rate_vector, error_rate_test_vector = best_classifier.train(
                train_data, train_labels, test_data, test_labels, verbose
            )

            plot_error_rate(
                error_rate_vector,
                error_rate_test_vector,
                f"First 30 samples for training, last 20 samples for testing\nα = {best_alpha}",
            )

            # Plot the confusion matrix for the training and test data
            plot_confusion_matrix(
                best_classifier,
                iris,
                "train",
                f"First 30 samples for training, last 20 samples for testing\nα = {best_alpha}",
                error_rate_train=error_rate_vector[-1],
            )
            plot_confusion_matrix(
                best_classifier,
                iris,
                "test",
                f"First 30 samples for training, last 20 samples for testing\nα = {best_alpha}",
                error_rate_test=error_rate_test_vector[-1],
            )

        # Use first 20 samples for training and last 30 samples for testing
        if choice == "3":

            test_condition = "Last 30 samples for training, first 20 samples for testing"

            # --- Task 1d --- #
            test_data, test_labels, train_data, train_labels = create_train_and_test(
                setosa_matrix, versicolor_matrix, virginica_matrix, 20
            )
            train_and_plot_MSE(
                train_data,
                train_labels,
                test_data,
                test_labels,
                alphas,
                test_condition,
                verbose,
            )
            best_alpha = 0.005
            best_classifier = LinearClassifier(alpha=best_alpha)

            _, error_rate_vector, error_rate_test_vector = best_classifier.train(
                train_data, train_labels, test_data, test_labels, verbose
            )

            plot_error_rate(
                error_rate_vector,
                error_rate_test_vector,
                f"{test_condition}\nα = {best_alpha}",
            )

            # Plot the confusion matrix for the training and test data
            plot_confusion_matrix(
                best_classifier,
                iris,
                "train",
                f"{test_condition}\nα = {best_alpha}",
                error_rate_train=error_rate_vector[-1],
            )
            plot_confusion_matrix(
                best_classifier,
                iris,
                "test",
                f"{test_condition}\nα = {best_alpha}",
                error_rate_test=error_rate_test_vector[-1],
            )

        # Remove features
        if choice in ["4", "5", "6"]:

            if choice == "4":
                # --- Task 2a --- #
                indices_to_remove = [1]
                removed_features = ["Sepal Width"]
            elif choice == "5":
                # --- Task 2b --- #
                indices_to_remove = [0, 1]
                removed_features = ["Sepal Length", "Sepal Width"]
            elif choice == "6":
                # --- Task 2b --- #
                indices_to_remove = [0, 1, 2]
                removed_features = ["Sepal Length", "Sepal Width", "Petal Length"]

            reduced_data = np.delete(iris_data, indices_to_remove, axis=1)
            reduced_setosa = reduced_data[:50]
            reduced_versicolor = reduced_data[50:100]
            reduced_virginica = reduced_data[100:150]

            train_data, train_labels, test_data, test_labels = create_train_and_test(
                reduced_setosa, reduced_versicolor, reduced_virginica, 30
            )
            best_alpha = 0.005
            best_classifier = LinearClassifier(alpha=best_alpha)
            _, error_rate_vector, error_rate_test_vector = best_classifier.train(
                train_data, train_labels, test_data, test_labels, verbose
            )

            plot_error_rate(
                error_rate_vector,
                error_rate_test_vector,
                f"Training and testing after removing features: {removed_features}\nα = {best_alpha}",
            )

            # --- Task 2c --- #
            plot_confusion_matrix(
                best_classifier,
                iris,
                "train",
                f"After removing features: {removed_features}\nα = {best_alpha}",
                error_rate_train=error_rate_vector[-1],
            )
            plot_confusion_matrix(
                best_classifier,
                iris,
                "test",
                f"After removing features: {removed_features}\nα = {best_alpha}",
                error_rate_test=error_rate_test_vector[-1],
            )

        if choice == "7":
            # Creating histograms for each feature with all classes in the same plot
            data = iris_data
            labels = iris.target
            features = iris.feature_names
            class_names = iris.target_names
            plot_characteristics_separability_histogram(data, labels, features, class_names)


if __name__ == "__main__":
    verbose = False
    if "-v" in sys.argv or "--verbose" in sys.argv:
        verbose = True

    main(verbose)
