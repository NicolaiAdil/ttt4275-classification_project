import sys
from sklearn.datasets import load_iris

# Internal imports
from handle_data import load_dataset, split_dataset
from plot import plot_characteristics_separability

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
    split_index = 30
    train_setosa, test_setosa = split_dataset(setosa_matrix, split_index)
    train_versicolor, test_versicolor = split_dataset(versicolor_matrix, split_index)
    train_virginica, test_virginica = split_dataset(virginica_matrix, split_index)






if __name__ == '__main__':
    verbose = False
    if "-v" in sys.argv or "--verbose" in sys.argv:
        verbose = True

    main(verbose)
