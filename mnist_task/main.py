
import os
import numpy as np
import scipy.io
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
            train_sample_size = 10000  # Train with smaller subset to save time
            test_sample_size = 500  # Test with smaller subset to save time

            # Create the object and load data into the member variables

            mnist_classifier = Classifier()
            mnist_classifier.load_data(mnist_data, test_sample_size, train_sample_size)
            

            predictions = mnist_classifier.predict(mnist_classifier.test_data)

            error_rate = mnist_classifier.get_error_rate(predictions, mnist_classifier.test_labels)
            print(f"Error Rate: {error_rate:.2f}%")

            confusion_matrix = mnist_classifier.get_confusion_matrix(predictions, "test")
            plot_confusion_matrix(confusion_matrix, mnist_classifier.num_classes, "test")

            

if __name__ == "__main__":
    main()