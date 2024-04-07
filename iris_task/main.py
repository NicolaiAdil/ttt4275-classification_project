import sys
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

# Internal imports
from handle_data import load_dataset, split_dataset
from plot import plot_characteristics_separability
from train import LinearClassifier

def main(verbose=False):

    # Load the Iris dataset
    iris = load_iris()
    iris_data, iris_labels = iris.data, iris.target
    setosa_matrix, versicolor_matrix, virginica_matrix = iris_data[:50], iris_data[50:100], iris_data[100:150]
    if verbose:
        print("Dimensions of data and labels respectively:", iris_data.shape, iris_labels.shape)
        # Plot the characteristics of the iris dataset for each class
        plot_characteristics_separability(setosa_matrix, versicolor_matrix, virginica_matrix, iris.feature_names)
    
    # --- Task 1a --- #
    # Split the dataset into training and testing sets
    # TODO: to this better
    split_index = 30
    train_setosa, test_setosa = split_dataset(setosa_matrix, split_index)
    train_setosa_labels, test_setosa_labels = np.zeros(split_index), np.zeros(20)

    train_versicolor, test_versicolor = split_dataset(versicolor_matrix, split_index)
    train_versicolor_labels, test_versicolor_labels = np.ones(split_index), np.ones(20)

    train_virginica, test_virginica = split_dataset(virginica_matrix, split_index)
    train_virginica_labels, test_virginica_labels = np.full(split_index, 2), np.full(20, 2)

    train_data = np.concatenate((train_setosa, train_versicolor, train_virginica))
    train_labels = np.concatenate((train_setosa_labels, train_versicolor_labels, train_virginica_labels))

    # --- Task 1b --- #
    # Train the linear classifier
    classifier = LinearClassifier()
    loss_vector = classifier.train(train_data, train_labels, verbose)
    plt.plot(loss_vector)






if __name__ == '__main__':
    verbose = False
    if "-v" in sys.argv or "--verbose" in sys.argv:
        verbose = True

    main(verbose)
