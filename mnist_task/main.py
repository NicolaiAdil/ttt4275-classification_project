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
        print("1: All data: K-Nearest neighboor classifier using euclidean distance k=1")
        print("2: All data: K-Nearest neighboor classifier using euclidean distance k=5")
        print("3: All data: K-Nearest neighboor classifier: Compare time used with or without batching k=1 REMEMBER TO CHANGE TEST SAMPLE SIZE!")
        print("4: Clustering: K-Nearest neighboor classifier using euclidean distance k=1")
        print("5: Clustering: K-Nearest neighboor classifier using euclidean distance k=7")
        print("q: Quit")

        choice = input("Your choice: ")
        print("\n")

        if choice == "q":
            break

        mnist_data = get_data_from_mat("mnist_data/data_all.mat")

        # Choose how large of a subset of the training and test data you want to train with
        TRAIN_SAMPLE_SIZE = TRAIN_SAMPLE_SIZE = len(mnist_data["trainlab"])
        TEST_SAMPLE_SIZE = (len(mnist_data["testlab"]) // 20)  # 5% of the test data, 500 samples #len(mnist_data["testlab"])

        # Task 1
        if choice in ["1", "2", "3"]:
            # Task 1a
            if choice in ["1", "3"]:
                k = 1
            # Extra
            if choice == "2":
                k = 5

            # Create the object and load data into the member variables
            mnist_classifier = Classifier()
            mnist_classifier.load_data(mnist_data, TEST_SAMPLE_SIZE, TRAIN_SAMPLE_SIZE)

            plot_histogram(
                mnist_classifier.test_labels,
                mnist_classifier.train_labels,
                "Histogram of Class Distribution in Test and Train Data",
            )

            # Task 1
            if choice in ["1", "2"]:
                # We look at just the nearest neighbor (k=1) to determine the class of the test data
                predictions = mnist_classifier.process_in_batches(
                    mnist_classifier.test_data,
                    mnist_classifier.train_data,
                    mnist_classifier.train_labels,
                    k=k,
                    batch_size=1000,
                )

                error_rate = mnist_classifier.get_error_rate(predictions)
                confusion_matrix = mnist_classifier.get_confusion_matrix(
                    predictions, "test"
                )
                plot_confusion_matrix(
                    confusion_matrix,
                    mnist_classifier.num_classes,
                    error_rate,
                    k,
                    "test",
                )

                # Task 1b
                images_reshaped = mnist_classifier.test_data.reshape(
                    -1, 28, 28
                )  # Assuming test_data is flattened
                display_classification(
                    images_reshaped,
                    mnist_classifier.test_labels,
                    predictions,
                    num_images=10,
                    title=f"Classification Results\nK={k}",
                )

            # Extra. 
            # To see real results you should change the TEST_SAMPLE_SIZE to a larger number (e.g. 10,000)
            # This is due to the fact that a computer has no problem with 1000 samples in memory, but struggles with for example 10,000 samples.
            elif choice == "3":
                start_time_batch = time.time()
                predictions_batch = mnist_classifier.process_in_batches(
                    mnist_classifier.test_data,
                    mnist_classifier.train_data,
                    mnist_classifier.train_labels,
                    k=k,
                    batch_size=1000,
                )
                end_time_batch = time.time()
                print(
                    f"---\nTime taken to process test data in batches: {end_time_batch - start_time_batch:.2f} seconds\nk={k}\nTrain data samples = {TRAIN_SAMPLE_SIZE}\nTest data samples = {TEST_SAMPLE_SIZE}\n---"
                )

                start_time = time.time()
                predictions = mnist_classifier.knn_predict(
                    mnist_classifier.test_data,
                    mnist_classifier.train_data,
                    mnist_classifier.train_labels,
                    k=k,
                )
                end_time = time.time()
                print(
                    f"---\nTime taken to predict test data without batching: {end_time - start_time:.2f} seconds\nk={k}\nTrain data samples = {TRAIN_SAMPLE_SIZE}\nTest data samples = {TEST_SAMPLE_SIZE}\n---"
                )

        # Task 2
        elif choice in ["4", "5"]:
            if choice == "4":
                k = 1
            # Task 2c
            if choice == "5":
                k = 7

            # Create the object and load data into the member variables
            mnist_classifier = Classifier()
            mnist_classifier.load_data(mnist_data, TEST_SAMPLE_SIZE, TRAIN_SAMPLE_SIZE)

            plot_histogram(
                mnist_classifier.test_labels,
                mnist_classifier.train_labels,
                "Histogram of Class Distribution in Test and Train Data",
            )

            # Task 2a
            start_time = time.time()
            templates_data, template_labels = mnist_classifier.cluster_by_class(num_clusters=64)
            end_time = time.time()
            print(
                f"---\nTime taken to cluster training data: {end_time - start_time:.2f} seconds\nTrain data samples = {TRAIN_SAMPLE_SIZE}\nTest data samples = {TEST_SAMPLE_SIZE}\n---"
            )
            plot_templates(templates_data, template_labels, 64, nTemplates=6)

            # Perform classification
            start_time = time.time()
            predictions = mnist_classifier.knn_predict(
                mnist_classifier.test_data, templates_data, template_labels, k=k
            )
            end_time = time.time()

            # Task 2b
            print(
                f"---\nTime taken to predict test data: {end_time - start_time:.2f} seconds\nk={k}\nTrain data samples = {TRAIN_SAMPLE_SIZE}\nTest data samples = {TEST_SAMPLE_SIZE}\n---"
            )
            error_rate = mnist_classifier.get_error_rate(predictions)
            confusion_matrix = mnist_classifier.get_confusion_matrix(
                predictions, "test"
            )
            plot_confusion_matrix(
                confusion_matrix, mnist_classifier.num_classes, error_rate, k, "test"
            )

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
