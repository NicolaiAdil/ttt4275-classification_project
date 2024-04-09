# Internal Imports
from plot import plot_error_rate, plot_confusion_matrix
from train import LinearClassifier, train_and_plot_MSE
from handle_data import load_dataset, split_dataset, create_train_and_test



def train_test_and_plot(iris, setosa_matrix, versicolor_matrix, virginica_matrix, verbose):

    # Split the dataset by class
    train_data, train_labels, test_data, test_labels = create_train_and_test(
        setosa_matrix, versicolor_matrix, virginica_matrix, 30
    )

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
    plot_confusion_matrix(best_classifier, iris.target_names, "train")
    plot_confusion_matrix(best_classifier, iris.target_names, "test")

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
    plot_confusion_matrix(best_classifier, iris.target_names, "train")
    plot_confusion_matrix(best_classifier, iris.target_names, "test")
