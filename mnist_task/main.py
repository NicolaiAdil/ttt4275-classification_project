
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist  # For loading MNIST data


# Internal imports
from train import Classifier

def main():
    
    while True:
        print("\nMenu:")
        print("1: Nearest neighboor classifier using euclidean distance")
        print("2: First 30 samples for training, last 20 samples for testing")
        print("q: Quit")

        choice = input("Your choice: ")

        if choice == "q":
            break

        if choice == "1":

            mnist_classifier = Classifier()
            mnist_classifier.load_data(mnist.load_data())

            # Prediction and evaluation
            test_sample_size = 1000  # Test with smaller subset to save time

            predictions = mnist_classifier.predict(mnist_classifier.test_data[:test_sample_size])
            error_rate = mnist_classifier.get_error_rate(predictions, mnist_classifier.test_labels[:test_sample_size])
            mnist_classifier.get_confusion_matrix("test")  # Plot confusion matrix for test data

            print("Error Rate:", error_rate)

if __name__ == "__main__":
    main()