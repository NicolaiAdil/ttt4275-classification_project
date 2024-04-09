import matplotlib.pyplot as plt
import sklearn.utils
import numpy as np
import seaborn as sns


def plot_characteristics_separability(setosa_matrix: np.ndarray, versicolor_matrix: np.ndarray, virginica_matrix: np.ndarray, iris_feature_names: list) -> None:
    """
    Plots the characteristics of the iris dataset for each class.

    Parameters:
        iris (sklearn.utils.Bunch): Object containing the iris dataset (data, labels, feature names, etc.).
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 8))

    # Setosa
    axs[0].scatter(setosa_matrix[:, 0], setosa_matrix[:, 1], color='r', label='Setosa')
    # Versicolor
    axs[0].scatter(versicolor_matrix[:, 0], versicolor_matrix[:, 1], color='g', label='Versicolor')
    # Virginica
    axs[0].scatter(virginica_matrix[:, 0], virginica_matrix[:, 1], color='b', label='Virginica')

    axs[0].set_xlabel(iris_feature_names[0])
    axs[0].set_ylabel(iris_feature_names[1])
    axs[0].set_title('Sepal Characteristics of Iris Dataset')
    axs[0].legend()

    # Plot petal length and width
    # Setosa
    axs[1].scatter(setosa_matrix[:, 2], setosa_matrix[:, 3], color='r', label='Setosa')
    # Versicolor
    axs[1].scatter(versicolor_matrix[:, 2], versicolor_matrix[:, 3], color='g', label='Versicolor')
    # Virginica
    axs[1].scatter(virginica_matrix[:, 2], virginica_matrix[:, 3], color='b', label='Virginica')

    axs[1].set_xlabel(iris_feature_names[2])
    axs[1].set_ylabel(iris_feature_names[3])
    axs[1].set_title('Petal Characteristics of Iris Dataset')
    axs[1].legend()

    plt.tight_layout()

    plt.show()

def plot_error_rate(error_rate_vector, error_rate_test_vector, title):
    plt.plot(error_rate_vector, label="Training data")
    plt.plot(error_rate_test_vector, label="Test data")
    plt.xlabel("Iteration")
    plt.ylabel("Error rate")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_confusion_matrix(classifier, class_names, train_or_test):
    confusion_matrix = classifier.get_confusion_matrix(
        train_or_test=train_or_test
    )
    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for {train_or_test} data")
    plt.show()
