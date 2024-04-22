
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist  # For loading MNIST data


# Internal imports
from train import Classifier

def main():
    
    classifier = Classifier()
    classifier.load_data(mnist.load_data())

    # Prediction and evaluation
    test_sample_size = 1000  # Test with smaller subset to save time
    predictions = classifier.predict(classifier.test_data[:test_sample_size])
    error_rate = classifier.get_error_rate(predictions, classifier.test_labels[:test_sample_size])
    confusion_mtx = classifier.get_confusion_matrix(predictions, classifier.test_labels[:test_sample_size])

    print("Error Rate:", error_rate)
    print("Confusion Matrix:\n", confusion_mtx)

if __name__ == "__main__":
    main()