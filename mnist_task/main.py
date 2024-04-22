
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist  # For loading MNIST data

# Internal imports
from train import Classifier
from plotting import plot_confusion_matrix

def main():
    
    while True:
        print("\nMenu:")
        print("1: Nearest neighboor classifier using euclidean distance")
        print("2: K-Nearest neighboor classifier using euclidean distance")
        print("q: Quit")

        choice = input("Your choice: ")

        if choice == "q":
            break

        if choice == "1":
            train_sample_size = 10000  # Train with smaller subset to save time
            test_sample_size = 100  # Test with smaller subset to save time

            # Create the object and load data into the member variables
            mnist_classifier = Classifier()
            mnist_classifier.load_data(mnist.load_data(), test_sample_size, train_sample_size)

            predictions = mnist_classifier.predict(mnist_classifier.test_data)

            error_rate = mnist_classifier.get_error_rate(predictions, mnist_classifier.test_labels)
            print(f"Error Rate: {error_rate:.2f}%")

            confusion_matrix = mnist_classifier.get_confusion_matrix(predictions, "test")
            plot_confusion_matrix(confusion_matrix, mnist_classifier.num_classes, "test")

            

if __name__ == "__main__":
    main()