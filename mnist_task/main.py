
import os
import scipy.io
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Internal imports
from train import Classifier
from plotting import (
    plot_confusion_matrix,
    plot_histogram,
    display_classification,
    plot_templates,
)


def get_data_from_mat(relative_path: str) -> dict:
    # Get the absolute path of the script (i.e., where the script is located)
    script_dir = os.path.dirname(__file__)
    
    # Construct the absolute file path to the .mat file
    mat_file_path = os.path.join(script_dir, relative_path)
    
    return scipy.io.loadmat(mat_file_path)


def main():
    
    while True:
        print("\nMenu:")
        print("1: Nearest neighboor classifier using euclidean distance (with batching)")
        print("2: Nearest neighboor classifier: Compare time used with or without batching")
        print("3: K-Nearest neighboor classifier using euclidean distance")
        print("q: Quit")

        choice = input("Your choice: ")
        print("\n")

        if choice == "q":
            break

        mnist_data = get_data_from_mat('mnist_data/data_all.mat')
        # Choose how large of a subset of the training and test data you want to train with
        TRAIN_SAMPLE_SIZE = len(mnist_data['trainlab'])  
        TEST_SAMPLE_SIZE = len(mnist_data['testlab']) // 20 # 5% of the test data, 500 samples

        # Create the object and load data into the member variables
        mnist_classifier = Classifier()
        mnist_classifier.load_data(mnist_data, TEST_SAMPLE_SIZE, TRAIN_SAMPLE_SIZE)

        # Task 1
        if choice in ['1', '2']:
        
            plot_histogram(mnist_classifier.test_labels, mnist_classifier.train_labels, "Histogram of Class Distribution in Training Data")
            
            # Task 1a, 1b
            if choice == '1':

                # We look at just the nearest neighbor (k=1) to determine the class of the test data
                predictions = mnist_classifier.process_in_batches(k=1, batch_size=1000)

                error_rate = mnist_classifier.get_error_rate(predictions)
                confusion_matrix = mnist_classifier.get_confusion_matrix(predictions, "test")
                plot_confusion_matrix(confusion_matrix, mnist_classifier.num_classes, error_rate, "test")

                # Display the classification results
                images_reshaped = mnist_classifier.test_data.reshape(-1, 28, 28)  # Assuming test_data is flattened
                display_classification(images_reshaped, mnist_classifier.test_labels, predictions, num_images=10)
            
            # Extra
            if choice == '2':
                # We look at just the nearest neighbor (k=1) to determine the class of the test data
                start_time_batch = time.time()
                predictions_batch = mnist_classifier.process_in_batches(k=1, batch_size=1000)
                end_time_batch = time.time()
                print(f"Time taken to process test data in batches: {end_time_batch - start_time_batch:.2f} seconds")

                start_time = time.time()
                predictions = mnist_classifier.predict(mnist_classifier.test_data, k=1)
                end_time = time.time()
                print(f"Time taken to predict test data: {end_time - start_time:.2f} seconds")

            
        if choice == '3':
            # Perform clustering
            templates = mnist_classifier.k_means_cluster_by_class(num_clusters=64)
            
            plot_templates(templates, nTemplates=6)

        else:
            print("Invalid choice. Please try again.")
            continue
            

if __name__ == "__main__":
    main()