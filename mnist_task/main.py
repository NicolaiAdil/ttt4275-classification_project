
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Internal imports
from train import Classifier
from plotting import plot_confusion_matrix

def get_data_from_mat(relative_path: str) -> dict:
    # Get the absolute path of the script (i.e., where the script is located)
    script_dir = os.path.dirname(__file__)
    
    # Construct the absolute file path to the .mat file
    mat_file_path = os.path.join(script_dir, relative_path)
    
    return scipy.io.loadmat(mat_file_path)

def plot_histogram(test_labels, train_labels, title="Histogram of class distribution"):
    # Extract labels and their counts for the histograms
    unique_test, counts_test = np.unique(test_labels, return_counts=True)
    unique_train, counts_train = np.unique(train_labels, return_counts=True)
    
    # Creating a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns, figure size 12x6 inches
    
    # Plotting the test data histogram
    ax1.bar(unique_test, counts_test, color='blue')
    ax1.set_title('Test Data Class Distribution')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Frequency')
    
    # Plotting the training data histogram
    ax2.bar(unique_train, counts_train, color='green')
    ax2.set_title('Training Data Class Distribution')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Frequency')
    
    # Setting a main title for the figure
    plt.suptitle(title)
    
    plt.show()


def main():
    
    while True:
        print("\nMenu:")
        print("1: Nearest neighboor classifier using euclidean distance")
        print("2: K-Nearest neighboor classifier using euclidean distance")
        print("q: Quit")

        choice = input("Your choice: ")

        mnist_data = get_data_from_mat('mnist_data/data_all.mat')

        if choice == "q":
            break

        if choice == "1":
            TRAIN_SAMPLE_SIZE = len(mnist_data['trainlab'])  # Train with smaller subset to save time
            TEST_SAMPLE_SIZE = 1000 #len(mnist_data['testlab'])  # Test with smaller subset to save time

            # Create the object and load data into the member variables
            mnist_classifier = Classifier()
            mnist_classifier.load_data(mnist_data, TEST_SAMPLE_SIZE, TRAIN_SAMPLE_SIZE)

            plot_histogram(mnist_classifier.test_labels, mnist_classifier.train_labels, "Histogram of Class Distribution in Training Data")
            
            # This is the same as solving K-means with K=1
            predictions = mnist_classifier.k_means(mnist_classifier.test_data)

            error_rate = mnist_classifier.get_error_rate(predictions)
            confusion_matrix = mnist_classifier.get_confusion_matrix(predictions, "test")
            plot_confusion_matrix(confusion_matrix, mnist_classifier.num_classes, error_rate, "test")

            

if __name__ == "__main__":
    main()