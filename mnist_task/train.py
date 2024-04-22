import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from scipy.spatial import distance  # For Euclidean distance calculation
from tqdm import tqdm

class Classifier:
    def __init__(self):
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None

    def load_data(self, data):
        (self.train_data, self.train_labels), (self.test_data, self.test_labels) = data
        self.train_data = self.train_data.reshape(-1, 28*28) / 255.0  # Flatten and normalize
        self.test_data = self.test_data.reshape(-1, 28*28) / 255.0  # Flatten and normalize

    def predict(self, data):

        predictions = np.empty(data.shape[0], dtype=self.train_labels.dtype)
        for i in tqdm(range(data.shape[0]), desc="Predicting"):
            distances = distance.cdist([data[i]], self.train_data, 'euclidean')
            nearest_index = np.argmin(distances)
            predictions[i] = self.train_labels[nearest_index]
        return predictions

    def get_confusion_matrix(self, train_or_test="test"):
        """
        Computes and plots the confusion matrix for specified data.

        Args:
            train_or_test (str): 'train' to use training data, 'test' to use testing data.
        """
        if train_or_test == "train":
            data, labels = self.train_data, self.train_labels
        else:
            data, labels = self.test_data, self.test_labels

        predictions = self.predict(data)
        num_classes = 10  # Number of classes in MNIST
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

        for true, pred in zip(labels, predictions):
            confusion_matrix[true][pred] += 1

        # Plot the confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[str(i) for i in range(num_classes)],
            yticklabels=[str(i) for i in range(num_classes)]
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix for {train_or_test.capitalize()} Data")
        plt.show()

    def get_error_rate(self, predictions, ground_truth):
        return np.mean(predictions != ground_truth)
    
